from utils.tools import categorize_boxes, denormalize
from torch.optim import Adam
import pytorch_lightning as pl
import torch
import numpy as np
import os, torchvision
import copy
import torch.nn.functional as F


class PIN(pl.LightningModule):

    def __init__(self, model, tokenizer, args, logger):
        '''method used to define our model parameters'''
        super().__init__()

        self.model = model     
        self.tokenizer = tokenizer
        
        self.lr = args.lr
        self.milestones = args.milestones_lr
        self.wd = args.wd
        self.vlm = args.vlm
        
        self.grid_size = args.grid_size
        self.codec = args.codec
        
        self.media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        self.max_new_tokens = args.max_new_tokens
        self.num_beams = args.num_beams
        
        self.wlogger = logger
        self.prompt_algo = args.prompt_algo
        
        self.image_size = args.image_size
        self.save_path = args.save_path
        self.save_freq = args.save_freq
        self.freq_eval_objdet = args.freq_eval_objdet
        self.test_mode = args.test_mode
        self.dloaders = ['val', 'zero_shot'] if not args.test_mode else ['pvoc', 'coco', 'lvis']

        self.save_hyperparameters()
        
    def _get_loss(self, batch, batch_idx, mode='train'):
        '''get loss in forward pass'''
        x, t, l, *_ = batch
        loss, _ = self.forward(x, t, l)

        self.log(f'{mode}/loss', loss, add_dataloader_idx=False, prog_bar=True)
        return loss

    def forward(self, vision, txt, mask_length):
        '''forward pass of model'''
        
        # mask out special tokens for loss computation
        labels = txt["input_ids"].clone()

        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[labels == self.tokenizer.eos_token] = -100
        labels[labels == self.media_token_id] = -100

        #mask out token unrelated to location prediction
        for i in range(labels.shape[0]):
            labels[i, :mask_length[i]] = -100
        
        logits = self.model(
            vision,
            txt.input_ids,
            txt.attention_mask.bool(),
        )[0]
        
        labels = torch.roll(labels, shifts=-1)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.view(-1))
        return loss, logits 
    
    def on_train_epoch_end(self):
        if self.prompt_algo.startswith('PIN'):
            if (self.current_epoch + 1) % self.save_freq == 0:
                mlp_dic = self.model.MLP.state_dict()
                pos_enc = self.model.pos_encoding 
                torch.save({'mlp': mlp_dic, 'pos_enc': pos_enc}, os.path.join(self.save_path, f'model_epoch{self.current_epoch+1}.pt'))
        elif self.prompt_algo.startswith('ViT_VPT'):
            prompt = self.model.vision_encoder.prompt
            torch.save({'prompt': prompt}, os.path.join(self.save_path, f'model_epoch{self.current_epoch+1}.pt'))
        elif self.prompt_algo.startswith('ViT_LoRA'):
            encoder = self.model.vision_encoder.state_dict()
            torch.save({'encoder': encoder}, os.path.join(self.save_path, f'model_epoch{self.current_epoch+1}.pt'))
    
    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        loss = self._get_loss(batch, batch_idx, mode='train')
        return loss
        
    def validation_step(self, batch, batch_idx, dataloader_idx):
        '''used for logging metrics'''

        if dataloader_idx == 0:
            self.evaluate(batch, batch_idx, dataloader_idx)
            if not self.test_mode:
                _ = self._get_loss(batch, batch_idx, mode='val_nexttoken')
        elif dataloader_idx == 1:
            self.evaluate(batch, batch_idx, dataloader_idx)
            if not self.test_mode:
                _ = self._get_loss(batch, batch_idx, mode='zero_shot_nexttoken')
        elif dataloader_idx >1 and (self.current_epoch + 1)% self.freq_eval_objdet == 0:
            self.evaluate(batch, batch_idx, dataloader_idx)
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        self.validation_step(batch, batch_idx, dataloader_idx)
        
    def configure_optimizers(self):
        '''defines model optimizer'''
        optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=0.5)
        return [optimizer], [scheduler]

    def visualize_bbox_text(self, image, bbox_p, bbox_gt, save_name, prd_txt):
        
        # Convert torch tensor images to wandb.Image format
        # This assumes your images are of shape [batch_size, channels, frames, height, width]
        upload_imgs = 16 if image.shape[0] > 16 else image.shape[0]
        images = copy.deepcopy(image)[:upload_imgs, -1, 0, :, :, :]
        images = [(denormalize(img).cpu()* 255).type(torch.uint8) for img in images]
        width = 2

        bboxs_gt = bbox_gt[:upload_imgs]
        images = [torchvision.utils.draw_bounding_boxes(img, box.unsqueeze(0), colors='#00EDFF', width=width) for img, box in zip(images, bboxs_gt)]
        
        bbox_pred = bbox_p[:upload_imgs]
        images = [torchvision.utils.draw_bounding_boxes(img, box.unsqueeze(0), colors='#FF5733', width=width) for img, box in zip(images, bbox_pred)]
        
        images = [img.permute(1, 2, 0).numpy() for img in images]
        self.wlogger.log_image(key=f'{save_name}_samples', images=images, caption=prd_txt[:upload_imgs])
    
    def run_inference(self, vision, prompt):
        
        # Generate TEXT (location description [,,,] for given object name in prompt)
        output = self.model.generate(
            vision_x=vision,
            lang_x=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
        ).cpu()
        
        
        # Process generated text
        bboxs = torch.cat([self.codec.decode(self.tokenizer.decode(token))[1] for token in output], dim=0)
        texts = [self.tokenizer.decode(token) for token in output]
        return bboxs, texts
    
    def evaluate(self, batch, batch_idx, dataloader_idx):
        '''
        evaluate model on given dataset
        '''
        ds_name = self.dloaders[dataloader_idx]
        vision, prompt, _, _, bbox_gt, bbox_grid_gt = batch

        bbox_pred, text_pred = self.run_inference(vision, prompt)
        
        # only visualize once at the beginning of the evaluation to not log too much images.
        if batch_idx == 0:
            self.visualize_bbox_text(vision, bbox_pred, bbox_gt, ds_name + '_visualization', text_pred)
            
        iou, iou_grid = [], []
        iou_small, iou_medium, iou_large = [], [], []
        
        # compute iou between ground truth and predicted bounding boxes based on groud truth obj names!
        iou.extend(torchvision.ops.box_iou(bbox_pred, bbox_gt.cpu()).diagonal().tolist()) # look at it 1 
        iou_grid.extend(torchvision.ops.box_iou(bbox_pred,  bbox_grid_gt.cpu()).diagonal().tolist()) # look at it 1 
        
        #compute iou for each size
        small_mask, medium_mask, large_mask = categorize_boxes(bbox_gt.cpu(), self.image_size)
        
        for i in range(len(bbox_gt)):
            # Calculate IoU for each category
            if small_mask[i].item() is True:
                iou_s = torchvision.ops.box_iou(bbox_pred[i].unsqueeze(0), bbox_gt[i].cpu().unsqueeze(0)).mean()
                iou_small.append(iou_s.item())
            if medium_mask[i].item() is True:
                iou_m = torchvision.ops.box_iou(bbox_pred[i].unsqueeze(0), bbox_gt[i].cpu().unsqueeze(0)).mean()
                iou_medium.append(iou_m.item())
            if large_mask[i].item() is True:
                iou_l = torchvision.ops.box_iou(bbox_pred[i].unsqueeze(0), bbox_gt[i].cpu().unsqueeze(0)).mean()
                iou_large.append(iou_l.item())
            
        # Log metrics
        self.log(f'{ds_name}/IoU', np.mean(iou), add_dataloader_idx=False)
        self.log(f'{ds_name}/IoU_grid', np.mean(iou_grid), add_dataloader_idx=False)

        if len(iou_small) > 0:
            self.log(f'{ds_name}/iou_gt_obj_small', np.mean(iou_small), add_dataloader_idx=False)
        if len(iou_medium) > 0:
            self.log(f'{ds_name}/iou_gt_obj_medium', np.mean(iou_medium), add_dataloader_idx=False)
        if len(iou_large) > 0:
            self.log(f'{ds_name}/iou_gt_obj_large', np.mean(iou_large), add_dataloader_idx=False)

        if ds_name == 'zero_shot' or ds_name == 'val':
            bbox_gt = [i.std(dim=0).mean().cpu() if torch.isnan(i.std(dim=0).mean()).any()==False else 0 for i in bbox_gt]
            self.log(f'{ds_name}/std_bbox_gt', np.array(bbox_gt).mean(), add_dataloader_idx=False)
            bbox_pred = [i.std(dim=0).mean().cpu() if torch.isnan(i.std(dim=0).mean()).any()==False else 0 for i in bbox_pred]
            self.log(f'{ds_name}/std_bbox_pred', np.array(bbox_pred).mean(), add_dataloader_idx=False)