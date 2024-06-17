import os
import numpy as np
import torch, random
from PIL import Image
import random

from eval_dloaders.voc_dataset_utils import VOC_CLASSES
from eval_dloaders.coco_dataset_utils import COCO_SYNSET_CATEGORIES
import json
from utils.tools import find_sublist_index, calculate_overlap_percentage

class TextTokenizerCollator(object):
    def __init__(self, tokenizer, mode):
        self.tokenizer = tokenizer
        self.mode = mode

    def __call__(self, data):

        vision_inputs, texts, mask_length, bbox_gt, bbox_grid_gt = zip(*data)
        vision_inputs = torch.concatenate(vision_inputs, dim=0).unsqueeze(1)
        
        # untangle texts from list in list to list
        texts = [item for sublist in texts for item in sublist]
        self.tokenizer.padding_side = 'right' if self.mode == 'train' else 'left'
        language_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        mask_length = torch.concatenate(mask_length, dim=0)
        bbox_gt = torch.concatenate(bbox_gt, dim=0)
        bbox_grid_gt = torch.concatenate(bbox_grid_gt, dim=0)
        
        return vision_inputs, language_inputs, mask_length, texts, bbox_gt, bbox_grid_gt

class XPaste(torch.utils.data.Dataset):
    def __init__(self, args, split='train'):
        """Initializes the dataset with configuration parameters."""
        self.grid_size = args.grid_size
        self.split = split
        self.root_dir = args.dataset_root
        self.processor = args.image_processor
        self.bckg_path = args.background_url
        self.val_size = args.val_size
        self.train_size = args.train_size
        self.test_size = args.test_size
        self.base_image_size = args.base_image_size
        self.rare_cases_cutoff = args.rare_cases_cutoff
        self.zero_shot_classes = args.zero_shot_classes
        self.max_count_sampling = args.max_count_sampling
        self.reconstruct_obj_name = args.reconstruct_obj_name
        self.same_object_likelihood = args.same_object_likelihood
        self.pos_img_referral= args.pos_img_referral

        self.synonyms_description = args.synonyms_description
        if args.synonyms_description:
            with open(args.synonyms_path, 'r') as file:
                self.synonyms = json.load(file)
            self.synonyms['baguet'] = self.synonyms['baguette']
            self.synonyms['icecream'] = self.synonyms['ice cream']

        # xpasting flags
        self.num_objects = args.num_objects_train if split == 'train' else args.num_objects
        self.num_fixed = args.num_objects_fixed
        self.outside_img_ratio = args.outside_img_ratio
        self.overlap_ratio = args.overlap_ratio
        self.overlap_constraint_for_both = args.overlap_constraint_for_both
             
        self.grid_min_size = [round(fac* self.grid_size) for fac in args.grid_min_size_factor]
        self.grid_max_size = [round(fac* self.grid_size) for fac in args.grid_max_size_factor]

        assert len(self.grid_min_size) >= self.num_objects + 1, "grid_min_size_factor should have length num_objects + 1"
        assert len(self.grid_max_size) >= self.num_objects + 1, "grid_max_size_factor should have length num_objects + 1"
        for  min, max in zip(self.grid_min_size, self.grid_max_size):
            assert min > 0 and max <= self.grid_size, "grid_min_size_factor and grid_max_size_factor should be between 0 and grid_size"
            assert min <= max, "grid_min_size_factor should be smaller than grid_max_size_factor"
            
        self.codec = args.codec
        self.text_constructer = args.text_constructer
        self.tokenizer = args.tokenizer
        assert self.codec.grid_size == self.grid_size, "The definition of grid size in the dataloader should match the one in the codec"
        
        # load background image dataset
        path = self.bckg_path + 'train/' if split == 'train' else self.bckg_path + 'testval/'
        self.base_image = [os.path.join(path, f) for f in os.listdir(path)]

        # open json file with info for objects
        with open(os.path.join(args.dataset_root[:-7], 'LVIS_instance_pools.json')) as f:
            info = json.load(f)
      
        # sort out rare classes < rare_cases_cutoff
        self.info = {}
        self.info_test = {}

        for key, val in info.items():
            self.info[key.replace('_', ' ').split(' (')[0]] = val  
        
        del self.info['projectile'] # unrelastic generations for this class
        lvis_coco = []
        for v in COCO_SYNSET_CATEGORIES:
            name = v['synset'].split('.')[0].replace('_', ' ')
            self.info_test[name] = self.info[name] # append those names to the zero-shot classes
            del self.info[name]
            lvis_coco.append(v['synset'])

        assert len(lvis_coco) == 80
        print("Removed successfully all coco classes from lvis xpaste train dataset")
        
        # remove all categories with in test categories for pvoc
        for tc in VOC_CLASSES:
            try:
                cl = args.re_synonyms[tc]
                self.info_test[cl] = self.info[cl] # append those names to the zero-shot classes
                del self.info[cl]
            except KeyError:
                pass
        
        self.label_names = [name for name in self.info.keys()]
        
        del_cls = []
        for key, val in info.items():
            if len(val) < self.rare_cases_cutoff:
                del_cls.append(key.replace('_', ' ').split(' (')[0])

        def dic2data(dic):
            data = []
            i = 0
            for key, values in dic.items():
                for val in values:
                    data.append([key, val])
                i += 1
            return data 
        
        diff = diff = self.zero_shot_classes - len(self.info_test.keys())  
        if diff > 0:
            # select k classes for zero shot evaluation
            keys = list(self.info.keys())
            random.seed(10)
            random.shuffle(keys) # seed is fixed so same classes for train, val and test
            keys = keys[:diff]
            
            for k in keys:
                self.info_test[k] = self.info[k]
                del self.info[k] # remove this key from self.info
            
        if split == 'test':
            self.data = dic2data(self.info_test)
            self.data_size = self.test_size 
        else:
            self.data = dic2data(self.info)
            # shuffle data 
            random.seed(10)
            random.shuffle(self.data)

            # split in train and eval
            split_train_eval = args.val_size/len(self.data)
            assert args.val_size < len(self.data)
            
            if self.split == 'train':
                self.data = self.data[:int(len(self.data)*(1-split_train_eval))]
                self.data_size = self.train_size
            else:
                self.data = self.data[int(len(self.data)*(1-split_train_eval)):]
                self.data_size = self.val_size

    def __len__(self):
        return self.data_size

    def get_text(self, top_left, bottom_right):
        return self.codec.encode(top_left+bottom_right)
    
    def sample_size(self, ratio, grid_min, grid_max):
        if ratio >= 1:
            size_x = random.randint(grid_min, grid_max -1)
            size_y = max(1, round(size_x/ratio))
        else:
            size_y = random.randint(grid_min, grid_max -1)
            size_x = max(1, round(size_y * ratio))

        assert size_x > 0 and size_y > 0, "sampled bbox size should be > 0"
        return size_x, size_y
    
    def overlay_image(self, base, path_to_overlay, bboxs, grid_min, grid_max):
        
        overlay = Image.open(self.root_dir + path_to_overlay.split('images')[-1]) # mask out old data path in th json    
        ratio = overlay.size[0] / overlay.size[1]
        
        grid_cell_size = self.base_image_size[0] // self.grid_size
        assert grid_cell_size == self.base_image_size[1] // self.grid_size
        
        if len(bboxs) > 0:
            # in case there is an existing object, sample size and position
            locations = []
            count = 0
            while len(locations) == 0 and count < self.max_count_sampling: 
                bbox_size_x, bbox_size_y = self.sample_size(ratio, grid_min, grid_max)
                locations = self.possible_locations([bbox_size_x, bbox_size_y], bboxs)
                count += 1
            
            if len(locations) == 0:
                # didn't find suitable size just take smallest one
                bbox_size_x, bbox_size_y = self.sample_size(ratio, grid_min, grid_min+1)
                start_grid_x = random.randint(0, self.grid_size - bbox_size_x -1)
                start_grid_y = random.randint(0, self.grid_size - bbox_size_y -1)
            else:
                start_grid_x, start_grid_y = random.choice(locations)[:2]            
        else:
            # in case there is no existing object, just sample random size and position
            bbox_size_x, bbox_size_y = self.sample_size(ratio, grid_min, grid_max)
            start_grid_x = random.randint(0, self.grid_size - bbox_size_x -1)
            start_grid_y = random.randint(0, self.grid_size - bbox_size_y -1)
        
        end_grid_x = start_grid_x + bbox_size_x # clip
        end_grid_y = start_grid_y + bbox_size_y # clip
        
        overlay_width = grid_cell_size * (bbox_size_x + 1)
        overlay_height = grid_cell_size * (bbox_size_y + 1)

        overlay = overlay.resize((overlay_width, overlay_height))
        top_left_x = start_grid_x * grid_cell_size
        top_left_y = start_grid_y * grid_cell_size

        # Use the mask to paste the overlay image on base image
        base.paste(overlay, (top_left_x, top_left_y), overlay)

        return base, [max(0, start_grid_x), max(0, start_grid_y)], [min(self.grid_size -1, end_grid_x), min(self.grid_size -1, end_grid_y)] 
    
    def determine_relative_position_to_others(self, object_names, bounding_boxes, preselected_object, preselected_coord, threshold=0.70):
        """
        Determine the relative position of a preselected object to other objects in the image.

        Args:
        object_names (list): List of object names.
        bounding_boxes (list): List of bounding boxes corresponding to the objects.
        preselected_object (str): The name of the preselected object.
        iou_threshold (float): The IoU threshold for considering overlap.

        Returns:
        str: Relative position of the preselected object to others (left, right, top, bottom, middle, before, behind).
        """
        # Extract the bounding box of the preselected object and other objects
        preselected_bbox = preselected_coord
        other_bboxes = []

        for name, bbox in zip(object_names, bounding_boxes):
            if name == preselected_object and all(a == b for a, b in zip(bbox, preselected_coord)) == False:
                other_bboxes.append(bbox)

        if len(other_bboxes) == 0:
            return None

        # Calculate the center of the preselected object
        preselected_center_x, preselected_center_y = 0.5 * (preselected_bbox[0] + preselected_bbox[2]), 0.5 * (preselected_bbox[1] + preselected_bbox[3])

        # Initialize variables to calculate differences
        diff_x, diff_y = [], []

        for bbox in other_bboxes:
            other_center_x, other_center_y = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])

            # Check IoU/overlap percentage for before/behind
            overlap = calculate_overlap_percentage(preselected_bbox, bbox)
            if overlap > threshold:
                index_preselected = find_sublist_index(bounding_boxes, preselected_bbox)
                index_other = find_sublist_index(bounding_boxes, bbox)
                
                if index_preselected > index_other:
                    return random.choice(["before", 'in front'])
                else:
                    return "behind"

            # Accumulate differences
            diff_x.append((preselected_center_x - other_center_x, other_center_x))
            diff_y.append((preselected_center_y - other_center_y, other_center_y))

        # Check if the preselected object is in the middle for three objects
        if len(other_bboxes) == 2:
            sorted_x = sorted(diff_x, key=lambda x: x[1])
            sorted_y = sorted(diff_y, key=lambda y: y[1])
            if sorted_x[0][0] < 0 < sorted_x[1][0] and sorted_y[0][0] < 0 < sorted_y[1][0]:
                return random.choice(["middle", 'center'])

        # Determine the direction with the highest difference
        max_diff_x = max(diff_x, key=lambda x: abs(x[0]))[0]
        max_diff_y = max(diff_y, key=lambda y: abs(y[0]))[0]

        if abs(max_diff_x) > abs(max_diff_y):
            return "left" if max_diff_x < 0 else "right"
        else:
            return random.choice(["above", 'top']) if max_diff_y < 0 else "below"
                

    def infer_object_position(self, bounding_box):
        """
        Infer the position of an object in an image based on its bounding box location.

        Args:
        bounding_box (tuple): The bounding box of the object, defined as (x1, y1, x2, y2).
        image_width (int): Width of the image.
        image_height (int): Height of the image.

        Returns:
        str: Position of the object in the image (left, right, middle, top, bottom).
        """
        # Calculate the center point of the bounding box
        center_x, center_y = (0.5 * (bounding_box[0] + bounding_box[2]), 0.5 * (bounding_box[1] + bounding_box[3]))

        # Determine the relative position based on the center point
        horizontal_section = int(center_x / (self.base_image_size[0] / 3))
        vertical_section = int(center_y / (self.base_image_size[1] / 3))

        if horizontal_section == 1 and vertical_section == 1:
            return random.choice(["middle", 'center'])
        elif horizontal_section == 2:
            return "right"
        elif horizontal_section == 0:
            return "left"
        elif vertical_section == 2:
            return "bottom"
        else:
            return "top"
    
    def compute_overlap_ratio(self, boxA, boxB):
        """
        Compute the overlap ratio of box1 with respect to box2.
        """
        x1, y1, x2, y2 = [boxA[0] * self.base_image_size[0] // self.grid_size, boxA[1] * self.base_image_size[1] // self.grid_size, \
                (boxA[2] + 1)* self.base_image_size[0] // self.grid_size, (boxA[3] + 1) * self.base_image_size[1] // self.grid_size]   
        x1b, y1b, x2b, y2b = [boxB[0] * self.base_image_size[0] // self.grid_size, boxB[1] * self.base_image_size[1] // self.grid_size, \
                (boxB[2] + 1)* self.base_image_size[0] // self.grid_size, (boxB[3] + 1) * self.base_image_size[1] // self.grid_size]
        

        # Compute the area of intersection
        interArea = max(0, min(x2, x2b) - max(x1, x1b)) * max(0, min(y2, y2b) - max(y1, y1b))
        area_box2 = (x2b - x1b) * (y2b - y1b)
        area_box1 = (x2 - x1) * (y2 - y1)

        # Avoid divide by zero
        if area_box2 == 0:
            return 0

        # Overlap ratio
        if self.overlap_constraint_for_both:
            overlap_ratio = max(interArea / area_box2, interArea / area_box1)
        else:
            overlap_ratio = interArea / area_box2

        return overlap_ratio

    def possible_locations(self, new_box_size, existing_boxes):
        """
        Compute possible locations for the new bounding box.
        image_dim: (width, height) of the image.
        new_box_size: (width, height) of the new bounding box.
        existing_boxes: List of existing bounding boxes in the image.
        """
        width, height = self.grid_size -1, self.grid_size -1
        new_width, new_height = new_box_size
        valid_boxes = [] 
        
        step_size1 = 1 if self.grid_size < 25 else random.randint(1,10)
        step_size2 = 1 if self.grid_size < 25 else random.randint(1,10)
        for x in range(-new_width , width, step_size1):  # Step size of 1 for fine-grained positions.
            for y in range(-new_height, height, step_size2):
                proposed_box = [x, y, x + new_width, y + new_height]
                total_overlap = sum(self.compute_overlap_ratio(proposed_box, e_box) for e_box in existing_boxes)

                # Check for max 50% occlusion with existing boxes
                if total_overlap <= self.overlap_ratio:
                    outside_right = max(0, proposed_box[2] - width)
                    outside_left = max(0, -proposed_box[0])
                    outside_bottom = max(0, proposed_box[3] - height)
                    outside_top = max(0, -proposed_box[1])
                    
                    outside_area =  (outside_right + outside_left) * new_height + \
                                    (outside_top + outside_bottom) * new_width - \
                                    (outside_right * outside_top + outside_left * outside_bottom + 
                                     outside_left * outside_top + outside_right * outside_bottom)
                    total_area = new_width * new_height

                    # Check for max x% outside the image boundary
                    if outside_area / total_area <= self.outside_img_ratio:
                        valid_boxes.append(proposed_box)

        return valid_boxes

    def find_and_select_random(self, target_object):
        # Filter the list to only include lists containing the target object
        filtered_list = [item for item in self.data if target_object in item]

        # Randomly select an item from the filtered list
        selected_item = random.choice(filtered_list)

        # Extract the object name and info
        object_name, img = selected_item
        return object_name, img
    
    def __getitem__(self, i):
        
        # prompt to reconstruct with next token prediction        
        obj_names = []; obj_coords = []; obj_rel_names = []
        
        #load background image
        image = Image.open(random.choice(self.base_image)).resize(self.base_image_size).copy()

        # determine how many objects to paste
        num_objects = self.num_objects if self.num_fixed else random.randint(1, self.num_objects)
        
        # define min and max size to sample from       
        grid_min = self.grid_min_size[num_objects-1]; grid_max = self.grid_max_size[num_objects-1]
        
        for _ in range(num_objects):
            
            # get object to overlay
            label, object = random.choice(self.data)
            
            # with a random choice select same object again.
            if random.random() < self.same_object_likelihood:
                if len(obj_names) > 0:
                    count = 0
                    for i in range(len(obj_names)-1):
                        if obj_names[i] == obj_names[-1]:
                            count += 1
                    if count == 0:
                        label = obj_names[-1]                    
                        label, object = self.find_and_select_random(label)
            
            # overlay image on the background
            image, x, y = self.overlay_image(image, object, obj_coords, grid_min, grid_max)
            # log information
            obj_names.append(label)
            obj_coords.extend([x + y])
        
        # preprocess vision data
        vision_x = self.processor(image).unsqueeze(0).unsqueeze(1)

        for i in range(len(obj_names)):
            label = obj_names[i]
            if self.same_object_likelihood > 0: # CHANGE TO DIFF FLAG FOR REFFERAL FINETUNING!!
                # check if the same object is in the image multiple times
                rel_pos = self.determine_relative_position_to_others(obj_names, obj_coords, label, obj_coords[i], threshold=0.75)
                object = label if not self.synonyms_description else np.random.choice(self.synonyms[label])
                if rel_pos is not None:
                    object = random.choice([rel_pos + ' ' + object, object + ' ' + rel_pos, object + ' on ' + rel_pos])
                else:
                    if random.random() < self.pos_img_referral:
                        abs_pos = self.infer_object_position(obj_coords[i])
                        object = random.choice([abs_pos + ' ' + object, object + ' ' + abs_pos, object + ' on ' + abs_pos, object + ' at the ' + abs_pos])
            else:
                object = label
            obj_rel_names.append(object)
            
        
        if self.split == 'train':
            # select random object and generate text for it for training
            idx = random.randint(0, len(obj_names)-1) 
            x, y = obj_coords[idx][:2], obj_coords[idx][2:]
            coord = self.get_text(x, y)
            obj_name = obj_rel_names[idx]

            text, text_target = self.text_constructer.construct_prompt_train(obj_name, coord, self.reconstruct_obj_name)
            
            # calculate the number of tokens where loss for next-token prediction should NOT be applied
            mask_length = torch.tensor([len(self.tokenizer(text_target, return_tensors="pt").input_ids[0])])
            return vision_x, [text], mask_length, torch.tensor([]), torch.tensor([])
        else:
            
            ## prepare things for validation
            # transfer grid codec positions to pixel value position of bboxs
            obj_bboxs = torch.tensor([self.codec.gc_to_bboxes_default_image_size([coords], self.grid_size)[0] for coords in obj_coords]) 
            obj_bboxs_grid = obj_bboxs # already pasted on exact grid locations
            # generate prompts for all objects pasted
            prompts = [self.text_constructer.construct_prompt(obj_n) for obj_n in obj_rel_names]
            # repeat images to match number of prompts
            vision_x = torch.repeat_interleave(vision_x, len(prompts), dim=0)
            
            # calculate the number of tokens where loss for next-token prediction should NOT be applied, for validation loss
            mask_length = torch.tensor([len(self.tokenizer(txt, return_tensors="pt").input_ids[0]) for txt in prompts])
            return vision_x, prompts, mask_length, obj_bboxs, obj_bboxs_grid
    

