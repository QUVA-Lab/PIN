import torch.nn as nn
import torch
import math
from types import MethodType
from einops import rearrange
from functools import reduce
from operator import mul

from utils.ViT_VPT_forward import new_ViT_VPT_forward_OF, new_ViT_VPT_forward_BLIP
from utils.tools import get_sinusoid_encoding_table
from utils.clip_lora import PlainMultiHeadAttention


# adapted vision forward pass from OpenFlamingo: https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/factory.py
def OF_adapted_encode_vision(self, vision_x: torch.Tensor):
    """
    Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
    Args:
        vision_x (torch.Tensor): Vision input
            shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

    rearrange code based on https://github.com/dhansmair/flamingo-mini
    """

    assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
    b, T, F = vision_x.shape[:3]
    assert F == 1, "Only single frame supported"
    vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
    
    if self.prompt_algo == 'ViT_VPT' or self.prompt_algo == 'ViT_LoRA': # enable grad for those methods in encoder =)
        vision_x = self.vision_encoder(vision_x)[1]
    else:
        with torch.no_grad():
            vision_x = self.vision_encoder(vision_x)[1]
    vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)    
    
    if self.prompt_algo == 'PIN':
        prompt = self.MLP(self.pos_encoding.to(device=vision_x.device, dtype=vision_x.dtype))
        vision_x = vision_x + prompt.reshape(1,1,1,-1,1024).repeat(b,T,F,1,1)
        
    vision_x = self.perceiver(vision_x)

    for layer in self.lang_encoder._get_decoder_layers():
        layer.condition_vis_x(vision_x)
        

# adapted forward pass from blip opt: https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_opt.py
def BLIP_adapted_forward(self, image, lang, attn):
    # adapted forward for BLIP, now also receiving tokens and targets
    with self.maybe_autocast():
        image = image.squeeze(1,2)

        image_embeds = self.ln_vision(self.visual_encoder(image))

        if self.prompt_algo == 'PIN':
            prompt = self.MLP(self.pos_encoding.to(device=self.ln_vision.weight.device, dtype=self.ln_vision.weight.dtype))   
            image_embeds = image_embeds + prompt.repeat(image.shape[0], 1, 1)

    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = self.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )

    inputs_opt = self.opt_proj(query_output.last_hidden_state)

    atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

    inputs_embeds = self.opt_model.get_input_embeddings()(lang)
    inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

    attention_mask = torch.cat([atts_opt, attn], dim=1)

    with self.maybe_autocast():
        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

    outputs.logits = outputs[0][:, inputs_opt.shape[1]:, :]  # only get the output for the tokens
    return outputs


def generate_adapted_blip2(
    self,
    vision_x,
    lang_x,
    attention_mask,
    max_new_tokens=30,
    num_beams=5,
    use_nucleus_sampling=False,
    min_length=1,
    top_p=0.9,
    repetition_penalty=1.0,
    length_penalty=1.0,
    num_captions=1,
    temperature=1,
):
    """
    Args:
        samples (dict): A dictionary containing the following keys:
            - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
        max_length (int): The maximum length of the sequence to be generated.
        num_beams (int): Number of beams for beam search. 1 means no beam search.
        use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
        min_length (int): The minimum length of the sequence to be generated.
        top_p (float): The cumulative probability for nucleus sampling.
        repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
        num_captions (int): Number of captions to be generated for each image.
    Returns:
        captions (list): A list of strings of length batch_size * num_captions.
    """
    with self.maybe_autocast():
        image = vision_x.squeeze()
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if self.prompt_algo == 'PIN':
            prompt = self.MLP(self.pos_encoding.to(device=self.ln_vision.weight.device, dtype=self.ln_vision.weight.dtype))
            image_embeds = image_embeds + prompt.repeat(image_embeds.shape[0], 1, 1)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
            image.device
        )
        
        attention_mask = torch.cat([atts_opt, attention_mask], dim=1)
        
        # new version for transformers>=4.27
        inputs_embeds = self.opt_model.get_input_embeddings()(lang_x)
        inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
        
        outputs = self.opt_model.generate(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_new_tokens,
            min_length=min_length,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        return outputs
    

def VLM_adaption(args, model):
    """
    Adapts a vision-language model (VLM) based on provided configuration arguments.
    This function adjusts various model components and settings including prompt number tokens,
    embedding dimensions, and specific forward methods based on the selected adaptation
    algorithm ('CoOp', 'PIN', 'ViT_VPT', 'ViT_LoRA'). The function ensures that only specified
    components (e.g., prompts) require gradients, aligning the model's trainable parameters
    with the desired adaptation strategy. 

    Args:
        args: A configuration object containing adaptation settings such as prompt algorithm,
              image dimensions, patch sizes, and specific model settings.
        model: The VLM model to be adapted.

    Returns:
        The adapted model with updated settings and components based on the specified algorithm.
    """
    
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0
    image_emb_dim = 1024 if args.vlm == 'openflamingo' else 1408
    model.prompt_algo = args.prompt_algo

    if args.prompt_algo == 'PIN':

        patches_per_dim = args.image_size // args.vit_patch_size
        patches = patches_per_dim ** 2
        patches += 1 if args.vlm == 'blip2' else 0
        model.patches = patches
                            
        # create encoding from which to start the PIN
        if args.embed_channel == 'sinus':
            model.pos_encoding = get_sinusoid_encoding_table(patches, args.embed_channel_dim)
        elif args.embed_channel == 'learned':
            model.pos_encoding = torch.nn.Parameter(torch.randn(patches, args.embed_channel_dim))
            var = math.sqrt(2/(patches + args.embed_channel_dim))
            nn.init.uniform_(model.pos_encoding.data, -var, var)
            model.pos_encoding.requires_grad_(True)
        
        current_dim = args.embed_channel_dim
        end_dim = image_emb_dim
            
        layer_list = []
        for dim in args.MLP_hidden_dim:
            layer_list.append(nn.Linear(current_dim, dim))
            layer_list.append(nn.SiLU())
            layer_list.append(nn.LayerNorm(dim))
            current_dim = dim
        layer_list.append(nn.Linear(current_dim, end_dim))
        
        model.MLP = nn.Sequential(
        *layer_list
        )

        if args.vlm == 'openflamingo':
            model._encode_vision_x = MethodType(OF_adapted_encode_vision, model)
            print('adapt forward')
        elif args.vlm == 'blip2':
            model.forward = MethodType(BLIP_adapted_forward, model)
            model.generate = MethodType(generate_adapted_blip2, model)
        model.MLP.requires_grad_(True)
    
    elif args.prompt_algo == 'ViT_VPT':
        val = math.sqrt(6. / float(3 * reduce(mul, (args.vit_patch_size, args.vit_patch_size), 1) + image_emb_dim))  # 1024?
        # Determine the attribute path based on the flag
        enc_type = 'vision_encoder' if args.vlm=='openflamingo' else 'visual_encoder'

        # Set the dropout for the appropriate encoder
        encoder = getattr(model, enc_type)
        encoder.prompt_dropout = torch.nn.Dropout(args.prompt_dropout)
        encoder.prompt_num_tokens = args.prompt_num_tokens       
        encoder.prompt = torch.nn.Parameter(torch.zeros(1, args.prompt_num_tokens, image_emb_dim))  # Set the parameter directly
        torch.nn.init.uniform_(encoder.prompt, -val, val)        
        
        if args.vlm == 'openflamingo':
            encoder.forward = MethodType(new_ViT_VPT_forward_OF, encoder)
            model.vision_encoder.prompt.requires_grad_(True)
            model._encode_vision_x = MethodType(OF_adapted_encode_vision, model) # need to turn of torch.no_grad for encoder
        elif args.vlm == 'blip2':
            encoder.forward = MethodType(new_ViT_VPT_forward_BLIP, encoder)
            model.visual_encoder.prompt.requires_grad_(True)
            model.forward = MethodType(BLIP_adapted_forward, model)
            model.generate = MethodType(generate_adapted_blip2, model)
    
    elif args.prompt_algo == 'ViT_LoRA':
        assert args.vlm == 'openflamingo', 'LoRA only supported for OpenFlamingo'
        for module in model.vision_encoder.transformer.resblocks:
            # change structure of multi head attention such that it works with lora
            new_module = PlainMultiHeadAttention()
            new_module.set_parameters(module.attn)
            module.attn = new_module
        
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_fc", "qkv", "c_proj", "proj"],
            lora_dropout=0.1,
            bias="none",
        )
        model.vision_encoder = get_peft_model(model.vision_encoder, config)
        model._encode_vision_x = MethodType(OF_adapted_encode_vision, model) # enable grad for vision encoder which is turned off by default

    print(
        f"{args.vlm} model adapted with {args.prompt_algo} using {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model