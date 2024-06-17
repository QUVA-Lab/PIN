import torch
from torchvision import datasets
import numpy as np
import os
from eval_dloaders.voc_dataset_utils import voc_get_bboxes, voc_get_gc, voc_gc_to_bboxes, voc_get_objects_names, Resize
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse


def calculate_filter_mask_and_mapping(target_files, num_obj):
    if num_obj:
        labels = [datasets.VOCDetection.parse_voc_xml(ET_parse(t).getroot()) for t in target_files]
        max_len = num_obj
        filter_mask = np.array([not isinstance(label['annotation']['object'], list) or
                                len(label['annotation']['object']) <= max_len for label in labels], dtype=bool)
    else:
        filter_mask = np.ones(len(target_files), dtype=bool)
    cumulative_mask_sum = np.cumsum(filter_mask)
    filter_mapping = {cumulative_mask_sum[i] - 1: i for i, value in enumerate(filter_mask) if value}
    return filter_mask, filter_mapping, sum(filter_mask)

class VOCTestSet(datasets.VOCDetection):
    def __init__(self, args):
        self.root = os.path.join(args.pvoc_dataset_root, 'test')
        super(VOCTestSet, self).__init__(root=self.root, year='2007', image_set='test', download=args.download)
        self.resize_processor = Resize(args.image_size)
        self.image_processor = args.image_processor
        self.grid_size = args.grid_size
        self.filter_mask, self.filter_mapping, self.effective_length = calculate_filter_mask_and_mapping(self.targets, args.num_objects)
        self.text_generator = args.text_constructer

    def __getitem__(self, index):
        index = self.filter_mapping[index]
        image, target = super(VOCTestSet, self).__getitem__(index)
        image, full_target = self.resize_processor(image, target)
        vision_input = torch.cat([self.image_processor(image).unsqueeze(0)], dim=0).unsqueeze(1)
        
        gt = torch.tensor(voc_get_bboxes(full_target))
        grid_gt = torch.tensor(voc_gc_to_bboxes(voc_get_gc(full_target, self.grid_size), full_target['annotation']['size'], self.grid_size))
        class_labels = voc_get_objects_names(full_target)
        
        vision_input = torch.repeat_interleave(vision_input, len(class_labels), dim=0)
        text = [self.text_generator.construct_prompt(cls) for cls in class_labels]
        return vision_input, text, torch.tensor([10]), gt, grid_gt

    def __len__(self):
        return self.effective_length