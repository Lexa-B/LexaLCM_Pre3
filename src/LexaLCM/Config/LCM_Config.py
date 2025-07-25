# src/LexaLCM/Config/LCM_Config.py

from transformers import PretrainedConfig, CONFIG_MAPPING

class LexaLCMConfig(PretrainedConfig):
    model_type = "lexa_lcm_pre3"
    def __init__(
        self,
        input_dim=1024,
        d_model=2048, # 2048 is the default for Meta FAIR's 1.6B LCM, 4096 for their 7B LCM, but you can use 1536 for a smaller model
        d_latent=1024,
        num_context_layers=5, # 5 is the default for Meta FAIR's 1.6B and 7B LCMs, but you can use less for a significantly smaller model
        num_denoiser_layers=15, # 13 is the default for Meta FAIR's 1.6B LCM, 14 for their 7B LCM, but you can use less for a significantly smaller model
        n_heads=32, # 16 is the default for Meta FAIR's 1.6B LCM, 32 for their 7B LCM, but you can use 8 for a slightly lighter model
        d_ff=8192, # 8192 (d_model: 2048 * 4, 4096 * 2, 1536 * 6) is the default for SwiGLU, but you can use 6144 (d_model: 2048 * 3, 1536 * 4, BUT DOESN'T WORK WITH d_model=4096!) for a smaller model
        dropout_context=0.1,
        dropout_latent=0.1,
        dropout_denoiser=0.15, # 0.15 is the default for Meta FAIR's LCM, reduced this to fight exploding gradients
        denoiser_iterations_pretrain = 100, # 100 is the default for Meta FAIR's LCM
        denoiser_iterations_inference = 40, # 40 is the default for Meta FAIR's LCM
        AdaLN_Timestep_Embed_Dim = 256,
        cfg_scale = 0.0, # Classifier-Free Guidance Scale
        gpus = None, # GPU configuration for split-GPU setup
        **kwargs
    ):
    # def __init__(
    #     self,
    #     input_dim=1024,
    #     d_model=1024, # 2048 is the default for Meta FAIR's 1.6B LCM, but you can use 1536 for a smaller model
    #     d_latent=1024,
    #     num_context_layers=2, # 5 is the default for Meta FAIR's 1.6B LCM, but you can use less for a significantly smaller model
    #     num_denoiser_layers=2, # 13 is the default for Meta FAIR's 1.6B LCM, but you can use less for a significantly smaller model
    #     n_heads=8, # 16 is the default for Meta FAIR's 1.6B LCM, but you can use 8 for a slightly lighter model
    #     d_ff=2048, # 8192 (* 4) is the default for SwiGLU, but you can use 6144 ( * 3) for a smaller model
    #     dropout_context=0.1,
    #     dropout_latent=0.1,
    #     dropout_denoiser=0.15, # 0.15 is the default for Meta FAIR's LCM, reduced this to fight exploding gradients
    #     denoiser_iterations_pretrain = 4, # 100 is the default for Meta FAIR's LCM
    #     denoiser_iterations_inference = 4, # 40 is the default for Meta FAIR's LCM
    #     AdaLN_Timestep_Embed_Dim = 256,
    #     cfg_scale = 0.0, # Classifier-Free Guidance Scale
    #     **kwargs
    # ):
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
        self.gpus = gpus or {'denoiser': 0, 'contextualizer': 0, 'other': 0}

CONFIG_MAPPING.register("lexa_lcm_pre3", LexaLCMConfig)
