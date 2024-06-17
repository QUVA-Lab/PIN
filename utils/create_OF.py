"""
Code taken and minimally adapted from:
https://github.com/mlfoundations/open_flamingo
"""

from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip
from openflamingo.flamingo import Flamingo
from openflamingo.flamingo_lm import FlamingoLMMixin
from openflamingo.utils import extend_instance

import torch, math
from torchvision import transforms


import torch
from torch import nn
from transformers import AutoModelForCausalLM

class MosaicGPT(nn.Module):
    def __init__(self, config):
        super(MosaicGPT, self).__init__()
        self.lang_encoder = AutoModelForCausalLM.from_pretrained(config['lang_encoder_path'],
                                                                 local_files_only=config['use_local_files'],
                                                                 trust_remote_code=True,
                                                                 cache_dir=config['cache_dir'])
        self._init_weights()

    def _init_weights(self):
        """
        Custom weight initialization.
        """
        if hasattr(self.lang_encoder, 'transformer'):
            # Assuming the language model has a 'transformer' attribute
            model_embeds = self.lang_encoder.transformer.wte
            nn.init.xavier_uniform_(model_embeds.weight)
        else:
            # Apply your custom initialization logic here
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask=None):
        return self.lang_encoder(input_ids, attention_mask=attention_mask)

    def resize_token_embeddings(self, new_num_tokens):
        """
        Resize the token embeddings and re-initialize weights.
        """
        old_embeddings = self.lang_encoder.get_input_embeddings()
        new_embeddings = self.lang_encoder.resize_token_embeddings(new_num_tokens)
        self._init_weights(new_embeddings)  # Re-initialize weights for new embeddings

    def _init_weights(self, new_embeddings):
        """
        Initialize weights for new embeddings specifically.
        """
        nn.init.xavier_uniform_(new_embeddings.weight)



def interpolate_pos_encoding(pos_embed, w, h, patch_size):
        N, dim = pos_embed.shape  # Original size N and embedding dimension dim

        # Calculate the new grid dimensions in terms of patches
        w0 = w // patch_size
        h0 = h // patch_size
        new_N = w0 * h0  # New total number of patches
        if N == new_N+1:
            return pos_embed  # No change if the number of patches remains

        
        class_pos_embed = pos_embed[0, :]
        patch_pos_embed = pos_embed[1:, :]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=0)
    
    
def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = False,
    cache_dir: Optional[str] = None,
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
        freeze_lm_embeddings (bool, optional): whether to freeze LM input embeddings when configuring Perceiver.
        cache_dir (str, optional): path to cache directory for downloading OpenClip/HF weights.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path,
        pretrained=clip_vision_encoder_pretrained,
        cache_dir=cache_dir,
    )
    
    # Adapt positional embeddings to allow for different image resolutions
    image_size = flamingo_kwargs['image_size']; flamingo_kwargs.pop('image_size')
    patch_size = flamingo_kwargs['vit_patch_size']; flamingo_kwargs.pop('vit_patch_size')
    if image_size!=224:
        new_transforms = []
        for transform in image_processor.transforms:
            if isinstance(transform, transforms.Resize):
                new_transforms.append(transforms.Resize(image_size, interpolation=Image.BICUBIC))
            elif isinstance(transform, transforms.CenterCrop):
                new_transforms.append(transforms.CenterCrop(size=(image_size,image_size)))
            else:
                new_transforms.append(transform)
        image_processor = transforms.Compose(new_transforms)

        # interpolation
        pos_embed = vision_encoder.visual.positional_embedding
        pos_embed_interpolate = interpolate_pos_encoding(pos_embed, image_size, image_size, patch_size)
        vision_encoder.visual.positional_embedding = torch.nn.Parameter(pos_embed_interpolate)

    # set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    # hacks for MPT-1B, which doesn't have a get_input_embeddings method
    if "mpt-1b-redpajama-200b" in lang_encoder_path:

        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings

        extend_instance(lang_encoder, EmbeddingFnMixin)

    # convert LM to FlamingoLM
    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    model = Flamingo(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
            "width"
        ],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        **flamingo_kwargs,
    )
    
    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    model.perceiver.requires_grad_(True)
    model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    if not freeze_lm_embeddings:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
        # TODO: investigate also training the output embeddings when untied

    # print(
    #     f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    # )

    return model, image_processor, text_tokenizer


def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
}