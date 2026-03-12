
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple
from .image_encoder import ImageEncoderViT
# from .origin_encoder import ImageEncoderViT
from models.rfl_seg.modeling.mask_decoder import MaskDecoder
from models.rfl_seg.modeling.prompt_encoder import PromptEncoder
from .PE import ImageAwarePromptEnhancer, FeatureModulator



class Samus(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoder,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
            enable_point_prompt: bool = True,
            enable_box_prompt: bool = False,
            enable_mask_prompt: bool = True,
            use_PE: bool = True,
            embed_dim: int = 256
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        for n, value in self.image_encoder.named_parameters():
            if "cnn_embed" not in n and "post_pos_embed" not in n and "Adapter" not in n and "2.attn.rel_pos" not in n and "5.attn.rel_pos" not in n and "8.attn.rel_pos" not in n and "11.attn.rel_pos" not in n and "upneck" not in n:
                value.requires_grad = False

        self.config = {
            'enable_point_prompt': enable_point_prompt,
            'enable_box_prompt': enable_box_prompt,
            'enable_mask_prompt': enable_mask_prompt,
            'use_PE': use_PE,
        }

        if use_PE:
            self.sparse_enhancer = ImageAwarePromptEnhancer(embed_dim)
            self.dense_enhancer = ImageAwarePromptEnhancer(embed_dim)
            self.sparse_modulator = FeatureModulator(embed_dim)
            self.dense_modulator = FeatureModulator(embed_dim)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward_sam(
            self,
            batched_input: List[Dict[str, Any]],
            multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        input_general_images = torch.stack([self.preprocess(x["generated_image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images, input_general_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )

            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def forward(
            self,
            imgs: torch.Tensor, #[4,1,256,256]
            gen_imgs: torch.Tensor, # [4,1,256,256]
            pt: Tuple[torch.Tensor, torch.Tensor],  # [b n 2, b n]
            bbox: torch.Tensor,
            coarse_masks: torch.Tensor,#[4,1,128,128]
    ) -> torch.Tensor:
        imge, vit_feature_list, anomaly_image, cnn_feature_list = self.image_encoder(imgs, gen_imgs) # imge:[4,256,32,32]
        if len(pt[0].shape) == 3:
            if not self.config['enable_point_prompt']:
                pt = None
            if not self.config['enable_box_prompt']:
                bbox = None
            if not self.config['enable_mask_prompt']:
                coarse_masks = None

            se, de = self.prompt_encoder(  # se b 2 256, de b 256 32 32
                points=pt,
                boxes=bbox,
                masks=coarse_masks,
            )

            if self.config['use_PE']:
                se = self.sparse_enhancer(se, imge)
                se = self.sparse_modulator(se, imge)
                de = self.dense_modulator(de, imge)
                de = self.dense_enhancer(de, imge)


            low_res_masks, _ = self.mask_decoder(  # low_res_mask
                image_embeddings=imge,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False,
            )
            masks = F.interpolate(low_res_masks, (256, 256), mode="bilinear", align_corners=False)
            outputs = {"low_res_logits": low_res_masks, "masks": masks}
            return outputs

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

