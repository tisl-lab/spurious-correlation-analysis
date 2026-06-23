"""
LRP-style attribution wrapper for CLIP models.

Implements the Chefer et al. (2021) transformer attribution method adapted for CLIP's
image-text cosine similarity as the attribution target.

Methods
-------
transformer_attribution
    Per-layer (grad × attn), rolled out. Text-conditioned. This is the main method and
    mirrors what ViT_LRP.py / ViT_explanation_generator.py do for ViT classification.
attention_rollout
    Raw-attention rollout from the CLS token. Text-agnostic baseline.
gradient
    Gradient × Input at patch level. Text-conditioned via cosine similarity gradient.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import CLIPModel


class CLIPLRPWrapper:
    """LRP-based attribution wrapper for CLIP vision encoders."""

    def __init__(self, clip_model: CLIPModel, clip_processor=None):
        self.model = clip_model
        self.processor = clip_processor
        self.model.eval()

        # The transformer_attribution and attention_rollout methods call
        # vision_model with output_attentions=True, which requires the eager
        # attention implementation.  SDPA / flash-attention do not expose
        # intermediate attention weights, so they cannot be used here.
        # Load the model with CLIPModel.from_pretrained(..., attn_implementation="eager").
        attn_impl = getattr(clip_model.config, "_attn_implementation", None)
        if attn_impl and attn_impl != "eager":
            raise ValueError(
                f"CLIPLRPWrapper requires attn_implementation='eager', "
                f"but the model was loaded with '{attn_impl}'. "
                f"Reload with CLIPModel.from_pretrained(..., attn_implementation='eager')."
            )

        cfg = clip_model.config.vision_config
        self.patch_size: int = cfg.patch_size
        self.image_size: int = cfg.image_size
        self.grid_size: int = self.image_size // self.patch_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _text_embeds(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Normalized text embeddings; always detached (no grad needed)."""
        with torch.no_grad():
            text_out = self.model.text_model(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                return_dict=True,
            )
            embeds = self.model.text_projection(text_out.pooler_output)
            return embeds / embeds.norm(dim=-1, keepdim=True)

    def _vision_forward(
        self,
        pixel_values: torch.Tensor,
        retain_attn_grad: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward through the vision encoder.

        Returns normalized image embeddings and a list of per-layer attention
        weight tensors [batch, heads, seq, seq].  When retain_attn_grad=True each
        tensor has retain_grad() called on it so .grad is populated after backward.
        """
        vision_out = self.model.vision_model(
            pixel_values=pixel_values,
            output_attentions=True,
            return_dict=True,
        )
        attentions = list(vision_out.attentions)
        if retain_attn_grad:
            for attn in attentions:
                attn.retain_grad()

        image_embeds = self.model.visual_projection(vision_out.pooler_output)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds, attentions

    def _rollout(
        self,
        cams: list[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute attention rollout.

        Each element of cams is [1, seq, seq] — a head-averaged, per-layer
        attention or relevance matrix.  Returns the accumulated rollout
        [1, seq, seq].
        """
        num_tokens = cams[0].shape[-1]
        eye = torch.eye(num_tokens, device=device).unsqueeze(0)
        result = eye.clone()
        for cam in cams:
            cam = cam + eye
            cam = cam / cam.sum(dim=-1, keepdim=True)
            result = torch.bmm(cam, result)
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_lrp_image_text_attribution(
        self,
        pixel_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        method: str = "transformer_attribution",
    ) -> tuple[torch.Tensor, float]:
        """
        Generate attribution map for image-text similarity.

        Parameters
        ----------
        pixel_values : [1, 3, H, W]
        text_input_ids : [1, seq_len]
        text_attention_mask : [1, seq_len]
        method : "transformer_attribution" | "attention_rollout" | "gradient"

        Returns
        -------
        attribution : [grid_size, grid_size] float tensor, normalized to [0, 1]
        similarity : cosine similarity score (float)
        """
        dispatch = {
            "transformer_attribution": self._transformer_attribution,
            "attention_rollout": self._attention_rollout_attribution,
            "gradient": self._gradient_attribution,
        }
        if method not in dispatch:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {list(dispatch)}"
            )
        return dispatch[method](pixel_values, text_input_ids, text_attention_mask)

    # ------------------------------------------------------------------
    # Attribution methods
    # ------------------------------------------------------------------

    def _transformer_attribution(
        self,
        pixel_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Text-conditioned transformer attribution.

        Mirrors the 'transformer_attribution' method in ViT_LRP.py / generate_LRP():
          1. Forward pass, retaining grad on every layer's attention weights.
          2. Backward from the text-image cosine similarity (conditions attribution
             on the specific text prompt — different prompts give different maps).
          3. Per layer: cam = (attn * grad).clamp(min=0), averaged over heads.
          4. Rollout the per-layer cams from the CLS token row.
        """
        device = pixel_values.device
        text_embeds = self._text_embeds(text_input_ids, text_attention_mask)

        self.model.zero_grad()
        image_embeds, attentions = self._vision_forward(
            pixel_values, retain_attn_grad=True
        )

        similarity = (image_embeds * text_embeds).sum(dim=-1)
        similarity_val = similarity.item()
        similarity.backward()

        cams = []
        for attn in attentions:
            if attn.grad is None:
                raise RuntimeError(
                    "Attention gradients are None after backward. "
                    "Ensure the forward pass is not wrapped in torch.no_grad() "
                    "and that retain_grad() was called on each attention tensor."
                )
            # Element-wise: relevance = attention weight × its gradient toward the target
            cam = (attn * attn.grad).clamp(min=0)  # [1, heads, seq, seq]
            cam = cam[0].mean(dim=0, keepdim=True)  # [1, seq, seq]
            cams.append(cam)

        rollout = self._rollout(cams, device)
        patch_attr = rollout[0, 0, 1:].reshape(self.grid_size, self.grid_size)

        if patch_attr.max() > 0:
            patch_attr = patch_attr / patch_attr.max()

        return patch_attr.detach(), similarity_val

    def _attention_rollout_attribution(
        self,
        pixel_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Pure attention rollout — text-agnostic baseline.

        Averages attention heads per layer and rolls out from the CLS token.
        The similarity score is still reported for the given text prompt, but
        the spatial attribution itself does not vary with the prompt.
        """
        device = pixel_values.device

        with torch.no_grad():
            image_embeds, attentions = self._vision_forward(
                pixel_values, retain_attn_grad=False
            )
            text_embeds = self._text_embeds(text_input_ids, text_attention_mask)
            similarity_val = (image_embeds * text_embeds).sum(dim=-1).item()

        cams = [attn[0].mean(dim=0, keepdim=True) for attn in attentions]

        rollout = self._rollout(cams, device)
        patch_attr = rollout[0, 0, 1:].reshape(self.grid_size, self.grid_size)

        if patch_attr.max() > 0:
            patch_attr = patch_attr / patch_attr.max()

        return patch_attr, similarity_val

    def _gradient_attribution(
        self,
        pixel_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Gradient × Input attribution at patch level.

        Backpropagates the text-conditioned cosine similarity to pixel space and
        aggregates (grad × input) over channels and spatial positions within each
        patch, giving a first-order approximation of each patch's contribution.
        """
        device = pixel_values.device
        text_embeds = self._text_embeds(text_input_ids, text_attention_mask)

        pixel_values = pixel_values.clone().detach().requires_grad_(True)
        self.model.zero_grad()

        vision_out = self.model.vision_model(pixel_values=pixel_values, return_dict=True)
        image_embeds = self.model.visual_projection(vision_out.pooler_output)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        similarity = (image_embeds * text_embeds).sum(dim=-1)
        similarity_val = similarity.item()
        similarity.backward()

        if pixel_values.grad is None:
            raise RuntimeError("Failed to compute gradients for pixel values.")

        # Gradient × Input: [3, H, W] — take absolute value for unsigned attribution
        grad_x_input = (pixel_values.grad * pixel_values.detach())[0].abs()

        # Reshape to [C, grid_H, patch_H, grid_W, patch_W] then sum within patches
        gs, ps = self.grid_size, self.patch_size
        grad_x_input = grad_x_input.reshape(3, gs, ps, gs, ps)
        patch_attr = grad_x_input.sum(dim=(0, 2, 4)).cpu().numpy()  # [gs, gs]

        if patch_attr.max() > 0:
            patch_attr = patch_attr / patch_attr.max()

        return torch.from_numpy(patch_attr).float().to(device), similarity_val

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def generate_lrp_visualization(
        self,
        pixel_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        original_image_np: np.ndarray,
        method: str = "transformer_attribution",
    ) -> tuple[np.ndarray, float]:
        """
        Run attribution and produce a heatmap overlaid on the image.

        Parameters
        ----------
        pixel_values : [1, 3, H, W]
        text_input_ids : [1, seq_len]
        text_attention_mask : [1, seq_len]
        original_image_np : H×W×3 float32 RGB array in [0, 1]
        method : attribution method name

        Returns
        -------
        vis : BGR uint8 image (OpenCV-compatible)
        similarity : cosine similarity score
        """
        import cv2

        attribution, similarity = self.generate_lrp_image_text_attribution(
            pixel_values, text_input_ids, text_attention_mask, method=method
        )

        attr_np = attribution.cpu().numpy() if isinstance(attribution, torch.Tensor) else attribution

        # Upsample [grid_size, grid_size] → [image_size, image_size]
        attr_tensor = torch.from_numpy(attr_np[np.newaxis, np.newaxis].astype(np.float32))
        attr_upsampled = torch.nn.functional.interpolate(
            attr_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

        # applyColorMap returns BGR — convert to RGB before blending with the RGB image
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * attr_upsampled), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Blend in RGB space, then convert to BGR for the caller
        vis = heatmap_rgb + np.float32(original_image_np)
        vis = vis / np.max(vis)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        return vis, similarity
