
import torch
import torch.utils.checkpoint as checkpoint

# Adapt forward pass for ViT model used in OpenFlamingo
def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

def new_ViT_VPT_forward_OF(self, x: torch.Tensor):
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

    # class embeddings and positional embeddings
    x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
    
    # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    
    x = self.patch_dropout(x)
    x = self.ln_pre(x)

    # ~~~~~~~~~~~ Start Our Code ~~~~~~~~~~~ #
    expanded_prompt = self.prompt_dropout(self.prompt).expand(x.shape[0], -1, -1).to(x.dtype)
    x = torch.cat([x[:, :1, :], expanded_prompt, x[:, 1:, :]], dim=1)

    # ~~~~~~~~~~~ End Our Code ~~~~~~~~~~~ #

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    # ~~~~~~~~~~~ Start Our Code ~~~~~~~~~~~ #
    # x = torch.cat([x[:,:1,:], x[:, self.prompt_num_tokens+1:, :]], dim=1)
    # ~~~~~~~~~~~ End Our Code ~~~~~~~~~~~ #

    if self.attn_pool is not None:
        if self.attn_pool_contrastive is not None:
            # This is untested, WIP pooling that should match paper
            x = self.ln_post(x)  # TBD LN first or separate one after each pool?
            tokens = self.attn_pool(x)
            if self.attn_pool_type == 'parallel':
                pooled = self.attn_pool_contrastive(x)
            else:
                assert self.attn_pool_type == 'cascade'
                pooled = self.attn_pool_contrastive(tokens)
        else:
            # this is the original OpenCLIP CoCa setup, does not match paper
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
    elif self.final_ln_after_pool:
        pooled, tokens = self._global_pool(x)
        pooled = self.ln_post(pooled)
    else:
        x = self.ln_post(x)
        pooled, tokens = self._global_pool(x)

    if self.proj is not None:
        pooled = pooled @ self.proj
    
    if self.output_tokens:
        return pooled, tokens
    
    return pooled


# Adapt forward pass for ViT model used in BLIP
def new_ViT_VPT_forward_BLIP(self, x):
    x = self.patch_embed(x)
    batch_size, seq_len, _ = x.size()

    cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    x = torch.cat((cls_tokens, x), dim=1)
    if self.pos_embed is not None:
        x = x + self.pos_embed
    x = self.pos_drop(x)

    rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

    expanded_prompt = self.prompt_dropout(self.prompt).expand(x.shape[0], -1, -1).to(x.dtype)
    x = torch.cat([x[:,:1,:], 
                expanded_prompt, 
                x[:,1:,:]], dim=1)

    for i, blk in enumerate(self.blocks):
        if self.use_checkpoint:
            x = checkpoint.checkpoint(blk, x, rel_pos_bias)
        else:
            x = blk(x, rel_pos_bias)
    return x

