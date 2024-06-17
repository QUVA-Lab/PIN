import json
import os
import torch
from pytorch_lightning.callbacks import Callback
import numpy as np

# move outside of dataloader as it takes long to load it, otherwise it will be loaded for train, val and test
def load_synonyms(path):
    with open(os.path.join(path)) as f:
        synonyms = json.load(f)['categories']
        
    synonyms = {item['name'].replace('_', ' ').split(' (')[0]:item['synonyms'] for item in synonyms}
    for key, val in synonyms.items():
        synonyms[key] = [v.replace('_', ' ').split(' (')[0] for v in val]
    return synonyms

def find_sublist_index(outer_list, element):
    for index, sublist in enumerate(outer_list):
        if all(a == b for a, b in zip(sublist, element)):
            return index

def categorize_boxes(boxes, img_size):
    """ Categorize boxes as small, medium, and large based on their areas. """
    if len(boxes.shape) > 1:
      areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # width * height
    else:
      areas = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])
    small_mask = areas < (32 * (img_size/224))**2
    medium_mask = (areas >= (32* (img_size/224))**2) & (areas < (96* (img_size/224))**2)
    large_mask = areas >= (96* (img_size/224))**2
    return small_mask, medium_mask, large_mask

def load_model(model, args,):
    '''
    Load different adaptions from a checkpoint file.
    '''
    if args.prompt_algo == 'PIN':
        weights = torch.load(args.load_ckpt_path, map_location='cuda:0')
        model.MLP.load_state_dict(weights['mlp'], strict=True)
        model.pos_encoding = weights['pos_enc']
    elif args.prompt_algo == 'ViT_LoRA':
        assert args.vlm == 'openflamingo', 'LoRA is only implemented for OpenFlamingo'
        weights = torch.load(args.load_ckpt_path, map_location='cuda:0')
        model.vision_encoder.load_state_dict(weights['encoder'], strict=True)
    elif args.prompt_algo == 'ViT_VPT':
        weights = torch.load(args.load_ckpt_path, map_location='cuda:0')
        enc_type = 'vision_encoder' if args.vlm=='openflamingo' else 'visual_encoder'
        encoder = getattr(model, enc_type)
        encoder.prompt = weights['prompt'] 
    return model

class UpdateDataLoaderCallback(Callback):
    def __init__(self, update_epoch):
        """
        update_epoch: The epoch at which the update should happen.
        new_dataset_getter: A function that returns the new dataset.
        """
        self.update_epoch = update_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.update_epoch:
            trainer.train_dataloader.dataset.reconstruct_obj_name = False
            trainer.train_dataloader.dataset.num_fixed = False

def denormalize(tensor):
    # OPENAI mean and std used for normalization taken from:
    # https://github.com/mlfoundations/open_clip/blob/f190703d847b7b234dbeb8265d7a69d7e7e4e996/src/open_clip/constants.py#L1
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def get_sinusoid_encoding_table(n_position, d_hid): 
    # sin-cos position encoding code taken from:
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 

def calculate_overlap_percentage(box1, box2):
    """
    Calculate the maximum overlap percentage between two bounding boxes.

    Args:
    box1, box2 (tuple): Bounding boxes defined as (x1, y1, x2, y2).

    Returns:
    float: The maximum overlap percentage between the two bounding boxes.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the overlap percentage for each box
    overlap_percentage_box1 = (interArea / box1Area) if box1Area > 0 else 0
    overlap_percentage_box2 = (interArea / box2Area) if box2Area > 0 else 0

    # Return the maximum overlap percentage
    return max(overlap_percentage_box1, overlap_percentage_box2)
