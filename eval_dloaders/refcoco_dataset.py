import os
from PIL import Image
import torch
from eval_dloaders.coco_dataset_utils import coco_gc_to_bboxes
from eval_dloaders.refcoco_utils import REFER, get_bboxes, get_gc, Resize


class RefCOCOTestSet():
    def __init__(self, args, dataset, splitBy, split):
        self.refer = REFER(data_root=args.refcoco_data_root, dataset=dataset, splitBy=splitBy)
        self.ref_ids = self.refer.getRefIds(split=split) 
        
        self.resize_processor = Resize(args.image_size)
        self.image_processor = args.image_processor
        self.image_size = (args.image_size, args.image_size)
        self.grid_size = args.grid_size
        self.text_generator = args.text_constructer
        
    def __getitem__(self, index):
        ref_id = self.ref_ids[index]
        ref = self.refer.loadRefs(ref_id)[0]

        curr_bbox = self.refer.getRefBox(ref_id)

        image = self.refer.Imgs[ref['image_id']]
        image = Image.open(os.path.join(self.refer.IMAGE_DIR, image['file_name'])).convert("RGB")
        image, adjusted_bbox = self.resize_processor(image, curr_bbox)

        expression = ref['sentences'][-1]['raw'] 
        vision_input = torch.cat([self.image_processor(image).unsqueeze(0)], dim=0).unsqueeze(1)
        
        gt = torch.tensor(get_bboxes(adjusted_bbox, self.image_size))
        grid_gt = torch.tensor(coco_gc_to_bboxes(get_gc(adjusted_bbox, self.grid_size, self.image_size) , self.grid_size, self.image_size))
        class_labels = [expression]
        
        vision_input = torch.repeat_interleave(vision_input, len(class_labels), dim=0)
        text = [self.text_generator.construct_prompt(cls) for cls in class_labels]
        return vision_input, text, torch.tensor([10]), gt, grid_gt
    
    def __len__(self):
        return len(self.ref_ids)