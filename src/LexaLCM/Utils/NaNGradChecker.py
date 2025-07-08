from transformers import TrainerCallback
import torch

class NaNGradChecker(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        for name, p in model.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    print(f"🚨 NaN in gradient of {name}")
                elif torch.isinf(p.grad).any():
                    print(f"🚨 Inf in gradient of {name}")
                    