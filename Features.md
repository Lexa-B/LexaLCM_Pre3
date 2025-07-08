* Two-tower Latent Diffusion LCM
  * Contextualizer tower (Transformer-style encoder)
    * Causal self-attention with rotary positional encoding (RoPE)
      * RoPE is fused into the attention mechanism (no separate RoPE layer)
    * Fixed embeddings
      * Source: Pretrained SONAR encoder (Sentence-level Multimodal and Language-Agnostic Representations)
      * Dimensions: 1024-dimensional, 32-bit floating point (fp32) vectors
      * Note: Embeddings are fixed during LCM training and not updated.
    * SwiGLU activation
    * RMSNorm
  * Denoiser tower
    * Cross-attention 
      * Dropout: Applied at a rate of 0.15 during training
      * Function: Integrates contextual information from the Contextualizer tower to guide the denoising process
      * Implementation: Utilizes standard Transformer cross-attention mechanisms
    * Classifier-free guidance
      * Guidance scale (gscale) = 3.0
      * Guidance rescaling (grescale) = 0.7
      * Initial noise scale (σinit) = 0.6
      * Epsilon-scaling (λeps) = 1.00045
    * NO positional encoding
      * (no RoPE)
      * (relies solely on timestep embeddings)
    * AdaLN
      * AdaLN is modulated using both the timestep embedding and cross-attention context.
    * SwiGLU activation
    * RMSNorm
    * Timestep embedding
      * Type: 256-dimensional sinusoidal embeddings
      * Purpose: Encodes the diffusion timestep information, crucial for the denoising process
      * Two-layer MLP with SiLU activation
    * Gaussian noise generator (for denoising process)
    * Default cosine noise schedule
      * Type: Cosine schedule with T=100 steps (as per Nichol & Dhariwal, 2021)
      * Dropout: A dropout rate of 0.1 is applied during training for regularization; this is not applied to the noise schedule itself.
    * Denoising steps: 
      * 100 (training)
      * 40 (inference)
  * Sharred hidden dimension
    * Both towers operate within the same hidden dimensional space, facilitating seamless information flow between them.
  * Transformer layers, each incorporating attention mechanisms and feed-forward networks.
* PreNet and PostNet
  * PreNet: Normalizes and maps input SONAR embeddings to the model's hidden dimension.
  * PostNet: Denormalizes the output and maps it back to the SONAR embedding dimension.
* Mixed-precision training
  * 32-bit float (fp32) for the input layer
  * 32-bit float (fp32) for the RoPE
  * 16-bit bfloat (bf16) for the rest of the model
