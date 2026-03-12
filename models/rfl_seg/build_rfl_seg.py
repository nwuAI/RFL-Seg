

import torch
from functools import partial
from .modeling.image_encoder import ImageEncoderViT
# from .modeling.origin_encoder import ImageEncoderViT
from .modeling.rfl_seg import Samus
from .modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from torch.nn import functional as F


def build_rfl_seg_vit_h(args, checkpoint=None):
    return _build_rfl_seg(
        args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_rfl_seg = build_rfl_seg_vit_h


def build_rfl_seg_vit_l(args, checkpoint=None):
    return _build_rfl_seg(
        args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_rfl_seg_vit_b(args, checkpoint=None):
    return _build_rfl_seg(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


rfl_seg_model_registry = {
    "default": build_rfl_seg_vit_h,
    "vit_h": build_rfl_seg_vit_h,
    "vit_l": build_rfl_seg_vit_l,
    "vit_b": build_rfl_seg_vit_b,
}


def _build_rfl_seg(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = args.encoder_input_size
    patch_size = image_size//32
    image_embedding_size = image_size // patch_size
    rfl_seg = Samus(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size= patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
    )
    rfl_seg.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            rfl_seg.load_state_dict(state_dict)
        except:
            new_state_dict = load_from2(rfl_seg, state_dict, image_size, patch_size)
            rfl_seg.load_state_dict(new_state_dict)
    return rfl_seg

def load_from(rfl_seg, sam_dict, image_size, patch_size):
    rfl_seg_dict = rfl_seg.state_dict()
    dict_trained = {k: v for k, v in sam_dict.items() if k in rfl_seg_dict}
    rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
    global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
    token_size = int(image_size//patch_size)
    for k in global_rel_pos_keys:
        rel_pos_params = dict_trained[k]
        h, w = rel_pos_params.shape
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
        dict_trained[k] = rel_pos_params[0, 0, ...]
    rfl_seg_dict.update(dict_trained)
    return rfl_seg_dict


def load_from2(rfl_seg, sam_dict, image_size, patch_size): # load the positional embedding
    rfl_seg_dict = rfl_seg.state_dict()
    dict_trained = {k: v for k, v in sam_dict.items() if k in rfl_seg_dict}
    token_size = int(image_size//patch_size)
    # pos_embed = dict_trained['image_encoder.pos_embed']
    # pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
    # pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
    # pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    # dict_trained['image_encoder.pos_embed'] = pos_embed
    rel_pos_keys = [k for k in dict_trained.keys() if 'rel_pos' in k]
    global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
    for k in global_rel_pos_keys:
        rel_pos_params = dict_trained[k]
        h, w = rel_pos_params.shape
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
        dict_trained[k] = rel_pos_params[0, 0, ...]
    rfl_seg_dict.update(dict_trained)
    return rfl_seg_dict
