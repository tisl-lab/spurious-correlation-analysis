"""
Portable RouteSAE — self-contained; imports nothing from the RouteSAE repo.

Drop this file plus a trained .pt checkpoint into any project.

Two entry points matter:

  1. CLIP behaviour through the SAE
         embeds = clip_embeds_with_sae(clip, sae, pixel_values)
     Runs CLIP with each patch's activation replaced by its SAE reconstruction
     at the layer the router chose, then returns the joint-space image embedding.

  2. Concepts and concept removal
         latents = encode_concepts(sae, clip, pixel_values)      # (B, T, latent)
         embeds  = clip_embeds_with_sae(clip, sae, pixel_values,
                                        ablate=[12, 5031])       # zero those concepts
     `ablate` zeroes the named latents inside the forward pass, so the effect
     propagates through CLIP's remaining layers - not a post-hoc edit of the
     output embedding.

The RouteSAE and TopK classes below are copied verbatim from the source repo so
existing checkpoints load unchanged.
"""

import argparse
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def pre_process(
    hidden_stats: torch.Tensor,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize hidden states to zero mean and unit variance.

    The SAE was trained on activations normalized this way; feeding raw
    activations produces silently wrong reconstructions.
    """
    mean = hidden_stats.mean(dim=-1, keepdim=True)
    std = hidden_stats.std(dim=-1, keepdim=True)
    x = (hidden_stats - mean) / (std + eps)
    return x, mean, std


# ---------------------------------------------------------------------------
# Copied verbatim from model.py (keeps state_dict keys identical)
# ---------------------------------------------------------------------------

class TopK(nn.Module):
    """
    TopK Sparse Autoencoder with fixed sparsity level.
    
    Architecture:
        pre_acts = encoder(x - pre_bias) + latent_bias
        latents = TopK(pre_acts, k)  # Only keep top-k activations
        reconstruction = decoder(latents) + pre_bias
    
    TopK SAE enforces exact sparsity by:
    - Computing all pre-activations
    - Selecting only the top-k largest values
    - Setting all others to zero
    
    This provides:
    - Predictable sparsity level (exactly k non-zero features)
    - No need for L1 regularization hyperparameter tuning
    - Direct control over computational cost
    
    The TopK operation is differentiable via straight-through estimator.
    
    Attributes:
        k: Number of features to activate (sparsity level)
        pre_bias: Learnable bias subtracted from input
        latent_bias: Learnable bias added to latent activations
        encoder: Linear layer mapping input to latent space
        decoder: Linear layer reconstructing input from latents
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        latent_size: int, 
        k: int
    ) -> None:
        """
        Initialize TopK SAE.
        
        Args:
            hidden_size: Dimensionality of the input residual stream activation
            latent_size: Number of latent features
            k: Number of features to activate (must be <= latent_size)
            
        Raises:
            ValueError: If hidden_size, latent_size, or k is invalid
        """
        if hidden_size <= 0 or latent_size <= 0:
            raise ValueError(f"hidden_size and latent_size must be positive, got {hidden_size} and {latent_size}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if k > latent_size:
            raise ValueError(f"k ({k}) cannot be larger than latent_size ({latent_size})")
            
        super(TopK, self).__init__()
        
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.k = k
        
        # Learnable parameters
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))
        
        # Encoder and decoder layers
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)

        # Initialize with tied weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize decoder weights as transpose of encoder."""
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activation values before TopK selection.
        
        Args:
            x: Input tensor (shape: [batch_size, seq_len, hidden_size])
            
        Returns:
            Pre-activation values (shape: [batch_size, seq_len, latent_size])
        """
        centered_x = x - self.pre_bias
        return self.encoder(centered_x) + self.latent_bias
    
    def get_latents(
        self, 
        pre_acts: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply TopK or threshold-based sparsity to pre-activations.
        
        Args:
            pre_acts: Pre-activation values (shape: [batch_size, seq_len, latent_size])
            infer_k: Optional override for k during inference (useful for ablations)
            theta: Optional threshold to use instead of TopK
            
        Returns:
            Sparse latent representation (shape: same as pre_acts)
            
        Raises:
            ValueError: If both infer_k and theta are provided
        """
        if infer_k is not None and theta is not None:
            raise ValueError('Cannot specify both infer_k and theta simultaneously. Choose one.')
        
        if theta is not None:
            # Threshold-based sparsity (for analysis)
            latents = torch.where(pre_acts > theta, pre_acts, torch.zeros_like(pre_acts))
        else:
            # TopK sparsity (default behavior)
            k = infer_k if infer_k is not None else self.k
            
            # Validate k
            if k > pre_acts.size(-1):
                warnings.warn(f"k ({k}) is larger than latent_size ({pre_acts.size(-1)}), using all latents")
                k = pre_acts.size(-1)
            
            # Select top-k activations
            topk_values, topk_indices = torch.topk(pre_acts, k, dim=-1)
            latents = torch.zeros_like(pre_acts)
            latents.scatter_(-1, topk_indices, topk_values)
        
        return latents

    def encode(
        self, 
        x: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Encode input to sparse latent representation.
        
        Args:
            x: Input tensor (shape: [batch_size, seq_len, hidden_size])
            infer_k: Optional override for k during inference
            theta: Optional threshold for sparsity
            
        Returns:
            Sparse latent representation (shape: [batch_size, seq_len, latent_size])
        """
        pre_acts = self.pre_acts(x)
        latents = self.get_latents(pre_acts, infer_k=infer_k, theta=theta)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to input space.
        
        Args:
            latents: Latent representation (shape: [batch_size, seq_len, latent_size])
            
        Returns:
            Reconstructed input (shape: [batch_size, seq_len, hidden_size])
        """
        return self.decoder(latents) + self.pre_bias
    
    def forward(
        self, 
        x: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the TopK autoencoder.
        
        Args:
            x: Input residual stream activation (shape: [batch_size, seq_len, hidden_size])
            infer_k: Optional override for number of active features during inference
            theta: Optional threshold for activation during inference
            
        Returns:
            Tuple of:
                - latents: Sparse latent representation with exactly k non-zero values per position
                          (shape: [batch_size, seq_len, latent_size])
                - x_hat: Reconstructed input (shape: [batch_size, seq_len, hidden_size])
                
        Example:
            >>> sae = TopK(hidden_size=768, latent_size=4096, k=64)
            >>> x = torch.randn(32, 128, 768)
            >>> latents, x_hat = sae(x)
            >>> # Verify exactly k non-zero per position
            >>> non_zero = (latents != 0).sum(dim=-1)
            >>> assert (non_zero == 64).all()
        """
        latents = self.encode(x, infer_k=infer_k, theta=theta)
        x_hat = self.decode(latents)
        return latents, x_hat




class RouteSAE(nn.Module):
    """
    Route Sparse Autoencoder with layer-wise routing mechanism.
    
    RouteSAE extends traditional SAEs to handle multi-layer representations
    by learning to route different tokens to different layers based on their
    semantic needs.
    
    Architecture:
        1. Router: Learns which layer(s) to process for each token
        2. Routing: Selects layer activations (hard or soft)
        3. SAE: Processes routed activations with TopK sparsity
    
    Key innovations:
    - Dynamic layer selection per token
    - Hard routing: Select single best layer (discrete, efficient)
    - Soft routing: Weighted combination of layers (differentiable, flexible)
    - Focuses on middle layers (layers n/4 to 3n/4)
    
    This allows the model to:
    - Capture both low-level and high-level features
    - Adapt processing depth to token semantics
    - Learn which layers contain most informative features
    
    Attributes:
        start_layer: First layer to consider for routing
        end_layer: Last layer to consider for routing (exclusive)
        router: Linear layer predicting layer weights
        sae: TopK SAE for processing routed activations
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        n_layers: int, 
        latent_size: int, 
        k: int
    ) -> None:
        """
        Initialize RouteSAE.
        
        Args:
            hidden_size: Dimensionality of layer activations
            n_layers: Total number of layers in the language model
            latent_size: Number of latent features in the SAE
            k: Number of active features (TopK sparsity)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if hidden_size <= 0 or latent_size <= 0 or n_layers <= 0 or k <= 0:
            raise ValueError("All dimensions must be positive")
        if k > latent_size:
            raise ValueError(f"k ({k}) cannot exceed latent_size ({latent_size})")
        if n_layers < 4:
            raise ValueError(f"n_layers ({n_layers}) should be at least 4 for meaningful routing")
            
        super(RouteSAE, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.latent_size = latent_size
        self.k = k
        
        # Focus on middle layers (typically most informative)
        self.start_layer = n_layers // 4
        self.end_layer = n_layers * 3 // 4 + 1
        self.n_routed_layers = self.end_layer - self.start_layer
        
        # Router: learns layer selection weights
        self.router = nn.Linear(hidden_size, self.n_routed_layers, bias=False)
        
        # SAE: processes routed activations
        self.sae = TopK(hidden_size, latent_size, k)

    def get_router_weights(
        self, 
        x: torch.Tensor, 
        aggre: str
    ) -> torch.Tensor:
        """
        Compute router weights for layer selection.
        
        Args:
            x: Multi-layer activations (shape: [batch, seq_len, n_layers, hidden_size])
            aggre: Aggregation method for router input:
                   - 'sum': Sum across layer dimension
                   - 'mean': Average across layer dimension
            
        Returns:
            Normalized router weights (shape: [batch, seq_len, n_layers])
            
        Raises:
            ValueError: If aggre method is not supported
        """
        if aggre == 'sum':
            router_input = x.sum(dim=2)  # Sum across layers
        elif aggre == 'mean':
            router_input = x.mean(dim=2)  # Average across layers
        else:
            raise ValueError(
                f'Unsupported aggregation method: {aggre}. '
                f'Expected one of ["sum", "mean"].'
            )
        
        # Compute router logits and normalize with softmax
        router_output = self.router(router_input)  # (batch, seq_len, n_layers)
        router_weights = F.softmax(router_output, dim=-1)  # Normalize to probabilities
        
        return router_weights
    
    def get_sae_input(
        self, 
        x: torch.Tensor, 
        router_weights: torch.Tensor, 
        routing: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply routing to select layer activations.
        
        Args:
            x: Multi-layer activations (shape: [batch, seq_len, n_layers, hidden_size])
            router_weights: Layer selection weights (shape: [batch, seq_len, n_layers])
            routing: Routing strategy:
                     - 'hard': Select single layer with highest weight (discrete)
                     - 'soft': Weighted combination of all layers (continuous)
            
        Returns:
            Tuple of:
                - batch_layer_weights: One-hot or soft layer weights 
                                      (shape: [batch, seq_len, n_layers])
                - routed_x: Selected/combined activations 
                           (shape: [batch, seq_len, hidden_size])
                           
        Raises:
            ValueError: If routing method is not supported
        """
        if routing == 'hard':
            # Hard routing: Select single best layer per token
            max_weights, target_layer = router_weights.max(dim=-1)
            # (batch, seq_len) -> indices of best layers
            
            # Create one-hot encoding for selected layers
            batch_layer_weights = torch.zeros_like(router_weights)
            batch_layer_weights.scatter_(2, target_layer.unsqueeze(-1), 1.0)
            
            # Gather activations from selected layers
            # Expand indices to match x's dimensions
            indices = target_layer.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, x.size(-1)
            )
            routed_x = torch.gather(x, 2, indices).squeeze(2)
            
            # Weight by router confidence
            routed_x = routed_x * max_weights.unsqueeze(-1)
        
        elif routing == 'soft':
            # Soft routing: Weighted combination of all layers
            # Expand router weights for broadcasting
            weights_expanded = router_weights.unsqueeze(-1)  # (batch, seq_len, n_layers, 1)
            
            # Weight each layer's activations
            weighted_hidden_states = x * weights_expanded
            
            # Sum across layers
            routed_x = weighted_hidden_states.sum(dim=2)  # (batch, seq_len, hidden_size)
            
            # Layer weights are the router weights themselves
            batch_layer_weights = router_weights
        
        else:
            raise ValueError(
                f'Unsupported routing method: {routing}. '
                f'Expected one of ["hard", "soft"].'
            )

        return batch_layer_weights, routed_x

    def forward(
        self, 
        x: torch.Tensor, 
        aggre: str, 
        routing: str,
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through RouteSAE.
        
        Args:
            x: Multi-layer activations (shape: [batch, seq_len, n_layers, hidden_size])
            aggre: Aggregation method for router: 'sum' or 'mean'
            routing: Routing strategy: 'hard' or 'soft'
            infer_k: Optional override for k during inference
            theta: Optional threshold for activation during inference
            
        Returns:
            Tuple of:
                - layer_weights: Selected/weighted layers (batch, seq_len, n_layers)
                - routed_x: Routed activations (batch, seq_len, hidden_size)
                - latents: SAE latent representation (batch, seq_len, latent_size)
                - x_hat: Reconstructed activations (batch, seq_len, hidden_size)
                - router_weights: Raw router weights before routing (batch, seq_len, n_layers)
                
        Note:
            For hard routing during inference, you can analyze which layers were selected
            most frequently using layer_weights. For soft routing, layer_weights shows
            the contribution of each layer.
            
        Example:
            >>> sae = RouteSAE(hidden_size=768, n_layers=16, latent_size=4096, k=64)
            >>> x = torch.randn(32, 128, 16, 768)  # multi-layer activations
            >>> layer_w, routed, latents, recon, router_w = sae(x, 'sum', 'hard')
            >>> # Analyze layer usage
            >>> layer_usage = layer_w.sum(dim=(0, 1))  # Count selections per layer
            >>> print(f"Most used layer: {layer_usage.argmax().item()}")
        """
        # Step 1: Compute router weights (which layers to use)
        router_weights = self.get_router_weights(x, aggre)
        
        # Step 2: Apply routing to select/combine layer activations
        batch_layer_weights, routed_x = self.get_sae_input(x, router_weights, routing)
        
        # Step 3: Process routed activations through SAE
        latents, x_hat = self.sae(routed_x, infer_k=infer_k, theta=theta)
        
        return batch_layer_weights, routed_x, latents, x_hat, router_weights
    



# ---------------------------------------------------------------------------
# Copied verbatim from utils.py
# ---------------------------------------------------------------------------

class RouteHook:
    """Forward hook for layer-specific RouteSAE interventions."""
    
    def __init__(
        self,
        cfg: argparse.Namespace,
        layer_idx: int,
        model: RouteSAE,
        batch_layer_weights: torch.Tensor,
        set_high: Optional[List[Tuple[int, float, int]]] = None,
        set_low: Optional[List[Tuple[int, float, int]]] = None,
        is_zero: bool = False  
    ) -> None:
        """
        Args:
            layer_idx: Current layer index
            batch_layer_weights: Shape (batch, seq_len, n_layers), indicates which layers to intervene
            set_high/set_low: Same as hook_SAE
            is_zero: If True, zero out activations instead of SAE intervention
        """
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.model = model
        self.batch_layer_weights = batch_layer_weights
        self.set_high = set_high or []
        self.set_low = set_low or []
        self.is_zero = is_zero  

    def __call__(
        self, 
        module: nn.Module, 
        inputs: tuple, 
        outputs: Union[torch.Tensor, tuple]
    ) -> Union[torch.Tensor, tuple]:
        """Apply SAE intervention to specified layer positions."""
        # Extract layer mask
        layer_mask = self.batch_layer_weights[
            :, :, self.layer_idx - self.model.start_layer + 1
        ].bool()

        if not layer_mask.any():
            return outputs

        # Unpack outputs
        if isinstance(outputs, tuple):
            outputs = list(outputs)
            output_tensor = outputs[0]
        else:
            output_tensor = outputs
        
        if output_tensor.shape[1] != layer_mask.shape[1]:
            return outputs

        if self.is_zero:
            # Zero out masked positions
            replace_mask = layer_mask.unsqueeze(-1).expand_as(output_tensor)
            output_tensor = output_tensor.clone()
            output_tensor[replace_mask] = 0
        else:
            # SAE intervention
            # Same dtype split as hook_SAE: float32 SAE, bfloat16 language model.
            lm_dtype = output_tensor.dtype
            x, mu, std = pre_process(output_tensor.float())
            latents = self.model.sae.encode(x, self.cfg.infer_k, self.cfg.theta)

            for (idx, val, mode) in self.set_high:
                if mode == 0:
                    latents[..., idx] += val
                elif mode == 1:
                    latents[..., idx] *= val

            for (idx, val, mode) in self.set_low:
                if mode == 0:
                    latents[..., idx] -= val
                elif mode == 1 and val != 0:
                    latents[..., idx] /= val

            x_hat = self.model.sae.decode(latents)
            reconstruct = (x_hat * std + mu).to(lm_dtype)

            # Replace masked positions with reconstructions
            replace_mask = layer_mask.unsqueeze(-1).expand_as(reconstruct)
            output_tensor = output_tensor.clone()
            output_tensor[replace_mask] = reconstruct[replace_mask]

        return tuple(outputs) if isinstance(outputs, list) else output_tensor


class RouteHookProjection:
    """Forward hook for layer-specific RouteSAE projection ablation.

    Same routed-position masking as RouteHook, but instead of round-tripping
    through encode -> edit latent -> decode, subtracts the projection onto
    span(P) (a precomputed projection matrix covering the concept directions
    to remove) directly. No need to re-derive sparse codes -- the concept
    directions are already known (they came from the decoder), so encoding
    is pure overhead here.

    P must be a (hidden_size, hidden_size) symmetric idempotent projection
    matrix built by msae_ftclip.build_projection_matrix (either the QR
    method, Q @ Q.T, or the pinv method, W.T @ pinv(W @ W.T) @ W -- both
    describe the same kind of object, this hook doesn't care which), and
    expressed in the SAE's own normalized input space (same zero-mean/
    unit-variance space pre_process() produces, which is what the decoder's
    directions live in). This hook normalizes the residual stream the same
    way before projecting, then un-normalizes the result, mirroring
    RouteHook's pre_process/un-normalize round trip exactly.
    """

    def __init__(
        self,
        layer_idx: int,
        model: 'RouteSAE',
        batch_layer_weights: torch.Tensor,
        P: torch.Tensor,
        lambda_coef: float = 1.0,
    ) -> None:
        self.layer_idx = layer_idx
        self.model = model
        self.batch_layer_weights = batch_layer_weights
        self.P = P                      # (hidden_size, hidden_size), symmetric idempotent
        self.lambda_coef = lambda_coef

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple,
        outputs: Union[torch.Tensor, tuple]
    ) -> Union[torch.Tensor, tuple]:
        layer_mask = self.batch_layer_weights[
            :, :, self.layer_idx - self.model.start_layer + 1
        ].bool()
        if not layer_mask.any():
            return outputs

        if isinstance(outputs, tuple):
            outputs = list(outputs)
            output_tensor = outputs[0]
        else:
            output_tensor = outputs

        if output_tensor.shape[1] != layer_mask.shape[1]:
            return outputs

        lm_dtype = output_tensor.dtype
        x, mu, std = pre_process(output_tensor.float())
        P = self.P.to(device=x.device, dtype=x.dtype)
        clean_x = x - self.lambda_coef * (x @ P)
        reconstruct = (clean_x * std + mu).to(lm_dtype)

        replace_mask = layer_mask.unsqueeze(-1).expand_as(reconstruct)
        output_tensor = output_tensor.clone()
        output_tensor[replace_mask] = reconstruct[replace_mask]

        return tuple(outputs) if isinstance(outputs, list) else output_tensor


def hook_routesae_projection(
    sae: RouteSAE,
    clip_model,
    batch_layer_weights: torch.Tensor,
    P: torch.Tensor,
    lambda_coef: float = 1.0,
) -> List[object]:
    """Register projection-ablation interventions on CLIP's vision encoder
    layers -- same layer/position targeting as hook_routesae, but via
    RouteHookProjection instead of RouteHook (see its docstring for why no
    encode/decode round trip is needed).

    hidden_states[k] is the output of encoder.layers[k-1], so a token routed
    to layer k is intervened on at module k-1 (same convention as
    hook_routesae).
    """
    handles = []
    num_layers = batch_layer_weights.size(-1)
    blocks = _vision_blocks(clip_model)
    needs_transpose = clip_backend(clip_model) == 'openai'

    for layer_idx in range(sae.start_layer - 1, num_layers + sae.start_layer - 1):
        if not batch_layer_weights[:, :, layer_idx - sae.start_layer + 1].any():
            continue

        hook = RouteHookProjection(
            layer_idx=layer_idx,
            model=sae,
            batch_layer_weights=batch_layer_weights,
            P=P,
            lambda_coef=lambda_coef,
        )
        if needs_transpose:
            hook = _LNDAdapter(hook)

        handles.append(blocks[layer_idx].register_forward_hook(hook))
    return handles



# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_routesae(
    checkpoint: str,
    hidden_size: int = 768,
    n_layers: int = 12,
    latent_size: int = 16384,
    k: int = 32,
    device: Union[str, torch.device] = 'cpu'
) -> RouteSAE:
    """Load a trained RouteSAE. Defaults match CLIP ViT-B/32 checkpoints.

    For Llama-3.2-1B checkpoints use hidden_size=2048, n_layers=16.
    """
    sae = RouteSAE(hidden_size, n_layers, latent_size, k)
    state = torch.load(checkpoint, weights_only=True, map_location=device)
    sae.load_state_dict(state)
    sae.to(device).eval()
    return sae


def _cfg(sae: RouteSAE, aggre: str, routing: str) -> argparse.Namespace:
    """RouteHook reads its options off a config object."""
    return argparse.Namespace(
        model='RouteSAE', aggre=aggre, routing=routing, infer_k=None, theta=None
    )


# ---------------------------------------------------------------------------
# CLIP activations - supports both HuggingFace and OpenAI CLIP
# ---------------------------------------------------------------------------
#
# The two libraries expose the same network differently:
#
#   HuggingFace   clip.vision_model.encoder.layers[i]   batch-first  (B, T, H)
#   OpenAI CLIP   clip.visual.transformer.resblocks[i]  seq-first    (T, B, H)
#
# Activations are numerically identical for the same weights (verified to
# 0.0e+00), so a checkpoint trained through one backend works through the other.


def clip_backend(clip_model) -> str:
    """Return 'hf' or 'openai' for a CLIP model object."""
    if hasattr(clip_model, 'vision_model'):
        return 'hf'
    if hasattr(clip_model, 'visual'):
        return 'openai'
    raise TypeError(
        'Unrecognized CLIP model: expected HuggingFace CLIPModel (.vision_model) '
        'or OpenAI CLIP (.visual)'
    )


def clip_num_layers(clip_model) -> int:
    """Number of vision transformer layers."""
    if clip_backend(clip_model) == 'hf':
        return len(clip_model.vision_model.encoder.layers)
    return len(clip_model.visual.transformer.resblocks)


def _vision_blocks(clip_model) -> nn.ModuleList:
    """The list of transformer blocks, whichever backend."""
    if clip_backend(clip_model) == 'hf':
        return clip_model.vision_model.encoder.layers
    return clip_model.visual.transformer.resblocks


class _LNDAdapter:
    """Wraps a batch-first hook so it can run on OpenAI CLIP's (T, B, H) tensors."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, module, inputs, outputs):
        tensor = outputs[0] if isinstance(outputs, tuple) else outputs
        result = self.inner(module, inputs, tensor.permute(1, 0, 2))
        tensor_out = result[0] if isinstance(result, tuple) else result
        return tensor_out.permute(1, 0, 2)


def clip_layer_stack(clip_model, pixel_values: torch.Tensor, n_layers: int = 12,
                     enable_grad: bool = False) -> torch.Tensor:
    """Per-layer patch activations shaped for RouteSAE: (B, T, routed_layers, H).

    Takes the middle band of layers (n//4 .. 3n//4), matching training.

    enable_grad=False (the default) keeps this under torch.no_grad(), which is
    what every extraction/ablation caller wants -- they only read activations
    and would otherwise retain a full autograd graph per batch. Pass True only
    when something needs to backprop THROUGH the stack to the input pixels
    (MACO's concept visualization does); a decorator can't be overridden by an
    outer torch.enable_grad(), which is why this is a flag rather than
    @torch.no_grad().
    """
    with (torch.enable_grad() if enable_grad else torch.no_grad()):
        start, end = n_layers // 4, n_layers * 3 // 4 + 1

        if clip_backend(clip_model) == 'hf':
            outputs = clip_model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            layers = outputs.hidden_states[start:end]
        else:
            # OpenAI CLIP has no output_hidden_states, so capture the blocks directly.
            blocks = clip_model.visual.transformer.resblocks
            captured = {}
            handles = [
                blocks[i].register_forward_hook(
                    lambda mod, inp, out, idx=i: captured.__setitem__(idx, out)
                )
                for i in range(start - 1, end - 1)
            ]
            try:
                clip_model.visual(pixel_values.type(clip_model.visual.conv1.weight.dtype))
            finally:
                for h in handles:
                    h.remove()
            # hidden_states[k] is the output of block k-1; transpose LND -> NLD
            layers = [captured[k - 1].permute(1, 0, 2) for k in range(start, end)]

        return torch.stack(layers, dim=0).permute(1, 2, 0, 3).float()


@torch.no_grad()
def clip_image_embeds(clip_model, last_hidden: torch.Tensor) -> torch.Tensor:
    """CLIP's joint-space image embedding from a final hidden state (B, T, H)."""
    if clip_backend(clip_model) == 'hf':
        pooled = clip_model.vision_model.post_layernorm(last_hidden[:, 0, :])
        return F.normalize(clip_model.visual_projection(pooled), dim=-1)

    visual = clip_model.visual
    pooled = visual.ln_post(last_hidden[:, 0, :])
    if visual.proj is not None:
        pooled = pooled @ visual.proj
    return F.normalize(pooled, dim=-1)


@torch.no_grad()
def clip_forward_last_hidden(clip_model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Run the vision tower and return its final hidden state as (B, T, H).

    Any hooks registered on the blocks take effect during this call.
    """
    if clip_backend(clip_model) == 'hf':
        return clip_model.vision_model(pixel_values=pixel_values).last_hidden_state.float()

    blocks = clip_model.visual.transformer.resblocks
    captured = {}
    handle = blocks[-1].register_forward_hook(
        lambda mod, inp, out: captured.__setitem__('last', out)
    )
    try:
        clip_model.visual(pixel_values.type(clip_model.visual.conv1.weight.dtype))
    finally:
        handle.remove()
    return captured['last'].permute(1, 0, 2).float()


# ---------------------------------------------------------------------------
# 1. Concepts
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_concepts(
    sae: RouteSAE,
    clip_model,
    pixel_values: torch.Tensor,
    aggre: str = 'sum',
    routing: str = 'hard'
) -> torch.Tensor:
    """Sparse concept activations per patch: (B, T, latent_size).

    Exactly k entries are non-zero per patch. Column j is concept j, comparable
    across images and across models trained with the same dictionary.
    """
    stack = clip_layer_stack(clip_model, pixel_values, sae.n_layers)
    x, _, _ = pre_process(stack)
    _, _, latents, _, _ = sae(x, aggre, routing)
    return latents


@torch.no_grad()
def routed_layers(
    sae: RouteSAE,
    clip_model,
    pixel_values: torch.Tensor,
    aggre: str = 'sum',
    routing: str = 'hard'
) -> torch.Tensor:
    """Which CLIP layer each patch was routed to: (B, T), in true layer numbers."""
    stack = clip_layer_stack(clip_model, pixel_values, sae.n_layers)
    x, _, _ = pre_process(stack)
    _, _, _, _, router_weights = sae(x, aggre, routing)
    return router_weights.argmax(dim=-1) + sae.start_layer


# ---------------------------------------------------------------------------
# 2. CLIP through the SAE, with optional concept removal
# ---------------------------------------------------------------------------

def hook_routesae(
    sae: RouteSAE,
    clip_model,
    batch_layer_weights: torch.Tensor,
    set_high: Optional[List[Tuple[int, float, int]]] = None,
    set_low: Optional[List[Tuple[int, float, int]]] = None,
    is_zero: bool = False,
    aggre: str = 'sum',
    routing: str = 'hard'
) -> List[object]:
    """Register SAE interventions on CLIP's vision encoder layers.

    hidden_states[k] is the output of encoder.layers[k-1], so a token routed to
    layer k is intervened on at module k-1.
    """
    handles = []
    num_layers = batch_layer_weights.size(-1)
    cfg = _cfg(sae, aggre, routing)
    blocks = _vision_blocks(clip_model)
    needs_transpose = clip_backend(clip_model) == 'openai'

    for layer_idx in range(sae.start_layer - 1, num_layers + sae.start_layer - 1):
        if not batch_layer_weights[:, :, layer_idx - sae.start_layer + 1].any():
            continue

        hook = RouteHook(
            cfg=cfg,
            layer_idx=layer_idx,
            model=sae,
            batch_layer_weights=batch_layer_weights,
            set_high=set_high,
            set_low=set_low,
            is_zero=is_zero,
        )
        if needs_transpose:
            hook = _LNDAdapter(hook)

        handles.append(blocks[layer_idx].register_forward_hook(hook))
    return handles


class RouteHookPerSample:
    """Like RouteHook, but zeros a DIFFERENT single concept per sample in the
    batch, instead of one shared set_high/set_low edit list applied to every
    row. Lets many (concept, image) pairs be tested in one forward pass
    instead of one pass per concept -- see msae_ftclip.py's
    Generate_Concept_Pool / hook_routesae_batched below.
    """

    def __init__(
        self,
        cfg: argparse.Namespace,
        layer_idx: int,
        model: RouteSAE,
        batch_layer_weights: torch.Tensor,
        sample_concept_idx: torch.Tensor,
    ) -> None:
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.model = model
        self.batch_layer_weights = batch_layer_weights
        self.sample_concept_idx = sample_concept_idx  # (B,) long, one concept id per row

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple,
        outputs: Union[torch.Tensor, tuple]
    ) -> Union[torch.Tensor, tuple]:
        layer_mask = self.batch_layer_weights[
            :, :, self.layer_idx - self.model.start_layer + 1
        ].bool()
        if not layer_mask.any():
            return outputs

        if isinstance(outputs, tuple):
            outputs = list(outputs)
            output_tensor = outputs[0]
        else:
            output_tensor = outputs

        if output_tensor.shape[1] != layer_mask.shape[1]:
            return outputs

        lm_dtype = output_tensor.dtype
        x, mu, std = pre_process(output_tensor.float())
        latents = self.model.sae.encode(x, self.cfg.infer_k, self.cfg.theta)

        B = latents.shape[0]
        latents[torch.arange(B, device=latents.device), :, self.sample_concept_idx] = 0.0

        x_hat = self.model.sae.decode(latents)
        reconstruct = (x_hat * std + mu).to(lm_dtype)

        replace_mask = layer_mask.unsqueeze(-1).expand_as(reconstruct)
        output_tensor = output_tensor.clone()
        output_tensor[replace_mask] = reconstruct[replace_mask]

        return tuple(outputs) if isinstance(outputs, list) else output_tensor


def hook_routesae_batched(
    sae: RouteSAE,
    clip_model,
    batch_layer_weights: torch.Tensor,
    sample_concept_idx: torch.Tensor,
    aggre: str = 'sum',
    routing: str = 'hard',
) -> List[object]:
    """Register per-sample concept-zeroing interventions across CLIP's routed
    layers -- row b of the batch gets sample_concept_idx[b] zeroed, letting
    many DIFFERENT (concept, image) pairs share one forward pass instead of
    needing one pass per concept (hook_routesae's set_high/set_low apply the
    same edit to the whole batch, which can't express that).
    """
    handles = []
    num_layers = batch_layer_weights.size(-1)
    cfg = _cfg(sae, aggre, routing)
    blocks = _vision_blocks(clip_model)
    needs_transpose = clip_backend(clip_model) == 'openai'

    for layer_idx in range(sae.start_layer - 1, num_layers + sae.start_layer - 1):
        if not batch_layer_weights[:, :, layer_idx - sae.start_layer + 1].any():
            continue

        hook = RouteHookPerSample(
            cfg=cfg,
            layer_idx=layer_idx,
            model=sae,
            batch_layer_weights=batch_layer_weights,
            sample_concept_idx=sample_concept_idx,
        )
        if needs_transpose:
            hook = _LNDAdapter(hook)

        handles.append(blocks[layer_idx].register_forward_hook(hook))
    return handles


@torch.no_grad()
def clip_embeds_with_sae(
    clip_model,
    sae: RouteSAE,
    pixel_values: torch.Tensor,
    ablate: Optional[Sequence[int]] = None,
    amplify: Optional[Sequence[Tuple[int, float]]] = None,
    aggre: str = 'sum',
    routing: str = 'hard'
) -> torch.Tensor:
    """CLIP image embeddings computed through the SAE reconstruction.

    Args:
        ablate:  concept indices to zero out (concept removal)
        amplify: (concept index, multiplier) pairs to scale up

    Interventions are applied to the latents inside the forward pass, so CLIP's
    remaining layers see the edited activations - this is a causal edit, not a
    post-hoc adjustment of the output embedding.
    """
    stack = clip_layer_stack(clip_model, pixel_values, sae.n_layers)
    x, _, _ = pre_process(stack)
    batch_layer_weights, _, _, _, _ = sae(x, aggre, routing)

    # RouteHook mode 1 is multiplicative, so removal is a multiply by zero.
    edits = [(int(i), 0.0, 1) for i in (ablate or [])]
    edits += [(int(i), float(v), 1) for i, v in (amplify or [])]

    handles = hook_routesae(
        sae, clip_model, batch_layer_weights,
        set_high=edits or None,
        aggre=aggre, routing=routing,
    )
    try:
        last_hidden = clip_forward_last_hidden(clip_model, pixel_values)
        return clip_image_embeds(clip_model, last_hidden)
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def clip_embeds_original(clip_model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Baseline embeddings with no SAE in the loop."""
    return clip_image_embeds(clip_model, clip_forward_last_hidden(clip_model, pixel_values))
