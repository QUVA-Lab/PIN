import os
import torch
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from lvis import LVIS
from PIL import Image

from eval_dloaders.coco_dataset_utils import Resize, coco_get_gc, coco_get_bboxes, coco_gc_to_bboxes    

def lvis_calculate_filter_mask_and_mapping(dataset, num_obj):
    labels = [dataset.load_anns(dataset.get_ann_ids([img_id])) for img_id in dataset.get_img_ids()]
    if num_obj:
        max_len = num_obj
        filter_mask = np.array([len(label) <= max_len and len(label)>0 for label in labels], dtype=bool)
    else:
        filter_mask = np.ones(len(dataset.ids), dtype=bool)
    cumulative_mask_sum = np.cumsum(filter_mask)
    filter_mapping = {cumulative_mask_sum[i] - 1: i for i, value in enumerate(filter_mask) if value}
    return filter_mask, filter_mapping, sum(filter_mask)

class LVISTest:
    def __init__(self, args):
        self.imags_root = args.coco_dataset_root
        annFile = os.path.join(args.lvis_root, 'lvis_v1_val.json')
        self.lvis_dataset = LVIS(annFile)
        self.image_ids = self.lvis_dataset.get_img_ids()
        self.class_names = {item['id']: item['name'].replace('_', ' ').split(' (')[0] for item in self.lvis_dataset.load_cats(ids=None)}
        
        self.resize_processor = Resize(args.image_size)
        self.image_size = (args.image_size, args.image_size)
        self.image_processor = args.image_processor
        self.grid_size = args.grid_size
        
        self.num_objects = args.num_objects
        self.filter_mask, self.filter_mapping, self.effective_length = lvis_calculate_filter_mask_and_mapping(self.lvis_dataset, self.num_objects)
        
        self.text_generator = args.text_constructer
    
    def _get_lvis_object_names(self, label):
        object_names = []
        for annotation in label:
            object_name = self.class_names[annotation['category_id']]
            object_names.append(object_name)
        return object_names

    def __getitem__(self, idx):

        idx = self.filter_mapping[idx]
        image_id = self.image_ids[idx]
        label = self.lvis_dataset.load_anns(self.lvis_dataset.get_ann_ids([image_id]))
        coco_url = self.lvis_dataset.load_imgs([image_id])[0]['coco_url']
        
        coco_split = coco_url.split('/')[-2]
        img_id = coco_url.split('/')[-1]
        img_path = os.path.join(self.imags_root, coco_split, img_id)
        image = Image.open(img_path).convert("RGB")        

        image, annotation = self.resize_processor(image, label)
        vision_input = torch.cat([self.image_processor(image).unsqueeze(0)], dim=0).unsqueeze(1)

        class_labels = self._get_lvis_object_names(annotation)
        gt = torch.tensor(coco_get_bboxes(annotation, self.image_size))
        grid_gt = torch.tensor(coco_gc_to_bboxes(coco_get_gc(annotation, self.grid_size, self.image_size) , self.grid_size, self.image_size))

        vision_input = torch.repeat_interleave(vision_input, len(class_labels), dim=0)
        text = [self.text_generator.construct_prompt(cls) for cls in class_labels]
        return vision_input, text, torch.tensor([10]), gt, grid_gt
    
    def __len__(self):
        return self.effective_length