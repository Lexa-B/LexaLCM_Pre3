# src/LexaLCM/Config/LCM_Config.py

from transformers import PretrainedConfig, CONFIG_MAPPING

class LexaLCMConfig(PretrainedConfig):
    model_type = "lexa_lcm_pre2"
    def __init__(
        self,
        input_dim=1024,
        d_model=1536, # 2048 is the default for Meta FAIR's LCM, but you can use 1536 for a smaller model
        d_latent=1024,
        num_context_layers=3, # 5 is the default for Meta FAIR's LCM, but you can use less for a significantly smaller model
        num_denoiser_layers=6, # 13 is the default for Meta FAIR's LCM, but you can use less for a significantly smaller model
        n_heads=16, # 16 is the default for Meta FAIR's LCM, but you can use 8 for a slightly lighter model
        d_ff=8192, # 8192 (* 4) is the default for SwiGLU, but you can use 6144 ( * 3) for a smaller model
        dropout_context=0.1,
        dropout_latent=0.1,
        dropout_denoiser=0.15, # 0.15 is the default for Meta FAIR's LCM, reduced this to fight exploding gradients
        denoiser_iterations_pretrain = 80,
        denoiser_iterations_inference = 40,
        AdaLN_Timestep_Embed_Dim = 256,
        cfg_scale = 0.0, # Classifier-Free Guidance Scale
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.d_model = d_model
        self.d_latent = d_latent
        self.num_context_layers = num_context_layers
        self.num_denoiser_layers = num_denoiser_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_context = dropout_context
        self.dropout_latent = dropout_latent
        self.dropout_denoiser = dropout_denoiser
        self.denoiser_iterations_pretrain = denoiser_iterations_pretrain
        self.denoiser_iterations_inference = denoiser_iterations_inference
        self.AdaLN_Timestep_Embed_Dim = AdaLN_Timestep_Embed_Dim
        self.cfg_scale = cfg_scale

CONFIG_MAPPING.register("lexa_lcm_pre2", LexaLCMConfig)
