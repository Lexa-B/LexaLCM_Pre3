from transformers import Trainer, Adafactor, get_scheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import wandb

# ToDo: make these global variables that can be set to True/False from the command line, maybe
Verbose_LCMTrainer = False

class LCMTrainer(Trainer):
    def __init__(self, *args, config_dict=None, inspection_decoder=None, periodic_inspection_steps=500, **kwargs):
        self.config_dict = config_dict or {}
        self.inspection_decoder = inspection_decoder
        self.periodic_inspection_steps = periodic_inspection_steps
        kwargs.pop("config_dict", None)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_name = self.args.optim.lower()
        params = self.model.parameters()

        if opt_name == "adafactor":
            clip_val = self.config_dict["training"].get("clip_threshold", 1.0)
            rel_step = self.config_dict["training"].get("adafactor_rel_step", True)
            lr = self.config_dict["training"].get("learning_rate", 0.00005)
            warmup_init = self.config_dict["training"].get("adafactor_warmup_init", True)
            self.optimizer = Adafactor(
                params,
                scale_parameter=True,
                relative_step=rel_step,
                clip_threshold=clip_val,
                warmup_init=warmup_init,
                lr=lr if rel_step is False else None, # Can't set lr if relative step or warmup_init is True
            )

            # 🩹 Manually patch param groups so dummy schedulers don't explode
            for group in self.optimizer.param_groups:
                if warmup_init and rel_step:
                    group["lr"] = 1e-3  # Safe dummy value
                    print("[LCMTrainer] ✅ Adafactor initialized with dummy lr for scheduler safety")
                else:
                    group["lr"] = lr
                    print(f"[LCMTrainer] ✅ Adafactor initialized with learning rate: {lr}")

            

        elif opt_name == "adamw":
            self.optimizer = AdamW(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        if optimizer is None:
            optimizer = self.optimizer

        if isinstance(optimizer, Adafactor):
            # ✅ Safe dummy scheduler just to satisfy Hugging Face's `.get_last_lr()`
            self.lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        else:
            self.lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )

        return self.lr_scheduler

    def log_gradient_norms(self):
        total_norm = 0.0
        grad_report = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                wandb.log({f"grad_norms/{name}": grad_norm}, step=self.state.global_step)
                total_norm += grad_norm ** 2

                # 🖨️ Collect a few gradients to print
                if len(grad_report) < 5:
                    grad_report.append(f"{name}: {grad_norm:.4f}")

        total_norm = total_norm ** 0.5
        wandb.log({"global_grad_norm": total_norm}, step=self.state.global_step)

        # 🖨️ Print the first few gradients once in a while
        if self.state.global_step < 5 or self.state.global_step % 50 == 0:
            print(f"[Gradients @ step {self.state.global_step}] global norm = {total_norm:.4f}")
            for line in grad_report:
                print(f"  • {line}")

    def training_step(self, model, inputs, batch_size=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Step-based inspection logic
        step = self.state.global_step
        if Verbose_LCMTrainer:
            print(f"[DEBUG] global_step = {step}, periodic_inspection_steps = {self.periodic_inspection_steps}. Encoder: {self.inspection_decoder}")
        # Periodic Inspection - If the argument is set, trigger inspection of the embeddings before model forward and after PostNet_D_Down
        if self.inspection_decoder is not None and self.periodic_inspection_steps > 0:
            # Only turn on inspection every N steps (e.g., 100)
            if step % self.periodic_inspection_steps == 0 and step != 0:
                if Verbose_LCMTrainer:
                    print("[DEBUG - Inspection] Enabling inspection_decoder for this step!")
                model.inspection_decoder = self.inspection_decoder
                model.periodic_inspection = True
            else:
                model.inspection_decoder = None
                model.periodic_inspection = False

            if step % self.periodic_inspection_steps == 0 and step != 0:
                model.inspection_decoder = self.inspection_decoder
                model.periodic_inspection = True
            else:
                model.inspection_decoder = None
                model.periodic_inspection = False

        # Inspection at every step - If the argument is set, trigger inspection of the embeddings before model forward and after PostNet_D_Down
        elif self.inspection_decoder is not None:
            model.inspection_decoder = self.inspection_decoder
            model.periodic_inspection = False

        # No inspection - If the argument is not set, do not trigger inspection of the embeddings before model forward and after PostNet_D_Down
        else:
            model.inspection_decoder = None
            model.periodic_inspection = False

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        loss.backward()

        print(f"[Step {self.state.global_step}] loss: {loss.item():.4f}")

        # ✅ Log gradient norms
        self.log_gradient_norms()

        return loss.detach()
