import torch
import pytorch_lightning as pl
import os 
import argparse
from synthetic_dataloader import XPaste
import datetime 
from pytorch_lightning.callbacks import LearningRateMonitor

from huggingface_hub import hf_hub_download

from trainer import PIN
from synthetic_dataloader import TextTokenizerCollator as TextTokenizer

from VLM_PIN_adaption import VLM_adaption
from utils.tools import load_model, UpdateDataLoaderCallback, load_synonyms
from utils.grid_codec import GridCodec
from utils.text_constructor import TextConstructor

from eval_dloaders.voc_dataset import VOCTestSet
from eval_dloaders.coco_dataset import COCOTestSet 
from eval_dloaders.lvis_dataset import LVISTest
from eval_dloaders.refcoco_dataset import RefCOCOTestSet

parser = argparse.ArgumentParser()

parser.add_argument('--run_name', default='debug', type=str)
parser.add_argument('--save_path', default='./', type=str)
parser.add_argument('--load_ckpt_path', default='', type=str)
parser.add_argument('--save_freq', default=5, type=int)
parser.add_argument('--freq_eval_objdet', default=20, type=int)

# Data
parser.add_argument('--dataset_root', default='path_to_pvoc', type=str)
parser.add_argument('--lvis_config_path', default='utils/lvis_v1_train.json', type=str)
parser.add_argument('--synonyms_path', default='utils/object_categories_synonyms.json', type=str)
parser.add_argument('--background_url', default='path_to_BG20k', type=str)

# Dataloader
parser.add_argument('--rare_cases_cutoff', default=20, type=int, help='Only use classes with this amount of images generated from '
                    'StableDiffusion in the synthetic dataloder')
parser.add_argument('--zero_shot_classes', default=50, type=int, help='Number of classes to leave out for zero-shot evaluation on synthetic data')
parser.add_argument('--train_size', default=200, type=int, help='Number of images used in one epoch during training for synthetic dataloader')
parser.add_argument('--val_size', default=20, type=int, help='Number of images used in one epoch during validation for synthetic dataloader')
parser.add_argument('--test_size', default=20, type=int, help='Number of images used in one epoch during testing for synthetic dataloader')
parser.add_argument('--same_object_likelihood', default=0.0, type=float)
parser.add_argument('--pos_img_referral', default=0.0, type=float)
parser.add_argument('--synonyms_description', default=False, action="store_true")
parser.add_argument('--grid_min_size_factor', nargs="*", default=[0.3, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1], type=float)
parser.add_argument('--grid_max_size_factor', nargs="*", default=[1.0, 1.0, 1.0, 1.0, 0.5, 0.4, 0.4, 0.4, 0.4], type=float)
parser.add_argument('--num_objects', default=5, type=int)
parser.add_argument('--num_objects_train', default=5, type=int)
parser.add_argument("--num_objects_fixed", action='store_true', default=False, help="same number of objects instead of sampling from a range")
parser.add_argument('--outside_img_ratio', default=0.0, type=float)
parser.add_argument('--overlap_ratio', default=0.2, type=float)
parser.add_argument("--overlap_constraint_for_both", default=False, action="store_true", help="Will consider the overlap constraint for every pair for every object")
parser.add_argument('--max_count_sampling', default=5, type=int)

# Grid definition
parser.add_argument('--grid_size', default=16, type=int)

# Optimization
parser.add_argument('--test_mode', default=False, action="store_true")
parser.add_argument('--train_batch', default=4, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument("--seed", type=int, default=56, help="seed for initializing training")
parser.add_argument('--wd', default=1e-5, type=float)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--milestones_lr',  nargs="*", type=int, default=[200])

# PIN
parser.add_argument('--MLP_hidden_dim',  nargs="*", type=int, default=[512, 768])
parser.add_argument("--embed_channel", type=str, default="sinus", help="add or concat")
parser.add_argument('--embed_channel_dim', default=64, type=int)

#Prompt
parser.add_argument("--start_prompt", type=str, default="In the image is", help="the language encoder")
parser.add_argument("--middle_prompt", type=str, default='located at', help="the language encoder")
parser.add_argument('--range_max_value', default=224, type=int, help="the maximal coordinate when using zero-one codec")
parser.add_argument("--reconstruct_obj_name", default=False, action="store_true", help="Also reconstruct obj name for next token prediction")
parser.add_argument('--epoch_change_reconstruct', default=200, type=int, help="the epoch to change reconstruct_obj_name value to False")

#Prompt Algorithms
parser.add_argument("--prompt_algo", type=str, default="pin",
                    choices=[
                        "ViT_VPT", 
                        "ViT_LoRA", # only supported for openflamingo
                        "PIN", 
                    ], help="Which method to use")
parser.add_argument('--prompt_dropout', default=0.1, type=float, help="if using vpt, the dropout probaility for the prompt")
parser.add_argument('--prompt_num_tokens', default=20, type=int, help="if using vpt or CoOp, th number of token prompts")

# VLM Model
parser.add_argument('--vlm', default='openflamingo', type=str)
parser.add_argument("--lang_encoder", type=str, default="anas-awadalla/mpt-1b-redpajama-200b-dolly", help="the language encoder")
parser.add_argument("--checkpoint", type=str, default="openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", help="model hugginface checkpoint path")
parser.add_argument("--blip_model_name", type=str, default="pretrain_opt2.7b", help="model name for blip2")

parser.add_argument("--max_new_tokens", type=int, default=20, help="the maximal number of token the model can generate")
parser.add_argument("--num_beams", type=int, default=3, help="The number of beams for the open falmingo model")
parser.add_argument("--image_size", type=int, default=224, help="The size images are resized to")
parser.add_argument("--vit_patch_size", type=int, default=14, help="The size of a patch in the ViT")

# For VOC/COCO/LVIS
parser.add_argument("--pvoc_dataset_root", type=str, default="./pascal-voc", help="The directory where the dataset is located")
parser.add_argument("--coco_dataset_root", type=str, default="/coco/", help="The directory where the coco dataset is located, with both train and validation datasets")
parser.add_argument("--lvis_root", type=str, default="", help="The directory where lvis json files are located")
parser.add_argument("--refcoco_data_root", type=str, default="", help="The directory where refcoco data is: including annotations and images folder")
parser.add_argument("--download", default=False, action="store_true", help="downloading the dataset")


def main():
    args = parser.parse_args()
    args.codec = GridCodec(args)
    pl.seed_everything(1, workers=True)

    args.save_path = f'{args.save_path}/model_{args.run_name}/'
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    project_name = 'debug' if args.run_name == 'debug' else "PromptLearning"    
    wandb_logger = pl.loggers.WandbLogger(name=args.run_name + "_" + time, project=project_name, save_dir=args.save_path)

    # load lvis synonyms used later in synthetic data loader to ensure none of the classes are used in the zero-shot evaluation
    args.synonyms_list = load_synonyms(args.lvis_config_path)
    args.re_synonyms = {}
    for key, val in args.synonyms_list.items():
        for v in val:
            args.re_synonyms[v] = key

    # Load specified VLM
    if args.vlm == 'openflamingo':
        from utils.create_OF import create_model_and_transforms

        if args.checkpoint.startswith("openflamingo/OpenFlamingo-3B"):
            cross_attn_every_n_layers = 1
        elif args.checkpoint.startswith("openflamingo/OpenFlamingo-4B"):
            cross_attn_every_n_layers = 2
        else:
            cross_attn_every_n_layers = 4
            
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=args.lang_encoder,
            tokenizer_path=args.lang_encoder,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            image_size=args.image_size,
            vit_patch_size=args.vit_patch_size
        )
        
        model = VLM_adaption(args, model)
        args.image_processor = image_processor
        args.tokenizer = tokenizer
        checkpoint_path = hf_hub_download(args.checkpoint, "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        
    elif args.vlm == 'blip2':
        from lavis.models import load_model_and_preprocess

        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", 
            model_type=args.blip_model_name
        )
        image_processor, tokenizer = vis_processors["eval"], model.opt_tokenizer
        args.image_processor = image_processor
        args.tokenizer = tokenizer
        model = VLM_adaption(args, model)
        
    else:
        raise(NotImplemented, f'{args.vlm} not implemented')
    
    if len(args.load_ckpt_path) > 0: 
        print(f'Reload our learnable parameters from checkpoint path {args.load_ckpt_path}')
        model = load_model(model, args)
    
    args.synonyms_list = load_synonyms(args.lvis_config_path)
    args.text_constructer = TextConstructor(args)

    # Create synthetic datasets for different splits
    args.base_image_size = (args.image_size, args.image_size)
    val_batch_size = args.train_batch // args.num_objects_train + 2

    model = PIN(model, tokenizer, args, wandb_logger)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    update_loader_callback = UpdateDataLoaderCallback(update_epoch=args.epoch_change_reconstruct)
    
    trainer = pl.Trainer(
        max_epochs = args.epoch, 
        accelerator="gpu", 
        devices=1 if args.run_name == 'debug' else "-1",
        precision=16,
        logger=wandb_logger, 
        check_val_every_n_epoch=10,
        callbacks=[lr_monitor, update_loader_callback],
        strategy= "ddp" if torch.cuda.device_count() > 1 else 'auto',
        num_sanity_val_steps=1 if args.run_name == 'debug' else 1,
        enable_checkpointing=False
    )    

    # Choose the the split to evaluate
    if not args.test_mode:
        collator_train = TextTokenizer(tokenizer, mode='train')
        dataset_train, dataset_val, dataset_zero_shot = XPaste(args, 'train'), XPaste(args, 'val'), XPaste(args, 'test')

        # Load data loaders
        train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=args.train_batch, num_workers=args.workers, drop_last=True, collate_fn=collator_train)
        
        collator_val = TextTokenizer(tokenizer, mode='val') # left padding instead of right padding
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=val_batch_size, num_workers=args.workers, collate_fn=collator_val)
        zero_shot_loader = torch.utils.data.DataLoader(dataset_zero_shot, batch_size=val_batch_size, num_workers=args.workers, collate_fn=collator_val)

        trainer.fit(model, train_loader, val_dataloaders=[val_loader, zero_shot_loader])     

    else:
        #Load evaluation loaders
        collator_val = TextTokenizer(tokenizer, mode='val') # left padding instead of right padding
        pvoc_loader = torch.utils.data.DataLoader(VOCTestSet(args), batch_size=val_batch_size, collate_fn=collator_val)
        coco_loader = torch.utils.data.DataLoader(COCOTestSet(args), batch_size=val_batch_size, collate_fn=collator_val)
        lvis_loader = torch.utils.data.DataLoader(LVISTest(args), batch_size=val_batch_size, collate_fn=collator_val)

        trainer.test(model, [pvoc_loader ,coco_loader, lvis_loader])

if __name__ == '__main__':
    main()