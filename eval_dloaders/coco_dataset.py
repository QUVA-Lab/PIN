from torchvision.datasets import CocoDetection
import numpy as np
import os
import torch
from eval_dloaders.coco_dataset_utils import Resize, coco_get_bboxes, coco_gc_to_bboxes, coco_get_gc, coco_get_objects_names


def calculate_filter_mask_and_mapping(coco, num_obj):
    labels = [coco.coco.loadAnns(coco.coco.getAnnIds(imgIds=[img_id])) for img_id in coco.ids]
    if num_obj:
        max_len = num_obj
        filter_mask = np.array([len(label) <= max_len and len(label)>0 for label in labels], dtype=bool)
    else:
        filter_mask = np.ones(len(coco.ids), dtype=bool)
    cumulative_mask_sum = np.cumsum(filter_mask)
    filter_mapping = {cumulative_mask_sum[i] - 1: i for i, value in enumerate(filter_mask) if value}
    return filter_mask, filter_mapping, sum(filter_mask)


class COCOTestSet(CocoDetection):
    def __init__(self, args):
        self.root = os.path.join(args.coco_dataset_root, 'val2017')
        annFile = os.path.join(args.coco_dataset_root, 'annotations/instances_val2017.json')
        
        super(COCOTestSet, self).__init__(self.root, annFile)
        self.image_size = (args.image_size, args.image_size)
        self.resize_processor = Resize(args.image_size)
        self.image_processor = args.image_processor
        self.grid_size = args.grid_size
        self.filter_mask, self.filter_mapping, self.effective_length = calculate_filter_mask_and_mapping(self, args.num_objects)
        self.text_generator = args.text_constructer

        
    def __getitem__(self, index):
        index = self.filter_mapping[index]
        image, label = super(COCOTestSet, self).__getitem__(index)
        image, annotation = self.resize_processor(image, label)
        vision_input = torch.cat([self.image_processor(image).unsqueeze(0)], dim=0).unsqueeze(1)
        
        gt = torch.tensor(coco_get_bboxes(annotation, self.image_size))
        grid_gt = torch.tensor(coco_gc_to_bboxes(coco_get_gc(annotation, self.grid_size, self.image_size) , self.grid_size, self.image_size))
        class_labels = coco_get_objects_names(annotation)

        vision_input = torch.repeat_interleave(vision_input, len(class_labels), dim=0)
        text = [self.text_generator.construct_prompt(cls) for cls in class_labels]
        return vision_input, text, torch.tensor([10]), gt, grid_gt
    
    def __len__(self):
        return self.effective_length
