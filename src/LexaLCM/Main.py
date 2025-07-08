# src/LexaLCM/Main.py

import yaml
import argparse
import wandb
import torch
import numpy as np
import evaluate
import os
import json

from src.LexaLCM.Config.LCM_Config import LexaLCMConfig
from LexaLCM.LCM_Model import LexaLCM
from LexaLCM.Data.DataHandler import LCMDataset, LCMDataset_DryRun, LCMCollator
from LexaLCM.Utils.NaNGradChecker import NaNGradChecker
from LexaLCM.Utils.LCMTrainer import LCMTrainer
from transformers import TrainingArguments
from Submodules.Pipeline_SONAR.src.pipelines import EmbeddingToTextPipeline


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    # Convert to numpy if they're tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Calculate L2 loss
    l2_loss = np.mean(np.sqrt(np.sum((predictions - labels) ** 2, axis=-1)))
    
    return {
        "eval_loss": float(l2_loss),
        "eval_l2_loss": float(l2_loss)
    }

def inspect_embedding_batch(batch_embeddings, decoder, n_seq=2):
    # batch_embeddings: [Batch, Seq, Emb]
    for i in range(min(n_seq, batch_embeddings.shape[0])):
        print(f"[INSPECT] Batch {i}:")
        for j in range(batch_embeddings.shape[1]):
            decoded = decoder(batch_embeddings[i, j, :].unsqueeze(0))
            print(f"  Seq {j}: {decoded}")

def RunTraining(config_training, model, train_dataset, val_dataset=None, dry_run=False, resume_from_checkpoint=None, resume_path=None, inspection_decoder=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if not dry_run:
        model.train()
    else:
        model.eval()

    print(f"Trainable Parameters: {count_trainable_params(model):,}")

    # Ensure save steps is a multiple of eval steps
    eval_steps = config_training['training']['eval_every']
    save_steps = config_training['training']['save_every']

    if eval_steps == 0:
        print("⚠️ Evaluation is disabled.")
        save_steps = save_steps  # no adjustment necessary
    else:
        if save_steps % eval_steps != 0:
            save_steps = ((save_steps // eval_steps) + 1) * eval_steps
            print(f"⚠️ Adjusted save_steps to {save_steps} to be a multiple of eval_steps ({eval_steps})")

    evaluation_strategy = "no" if eval_steps == 0 else "steps"
    load_best_model = evaluation_strategy != "no"

    training_args = TrainingArguments(
        output_dir=config_training['training']['output_dir'],
        per_device_train_batch_size=config_training['training']['batch_size'],
        bf16=config_training['training']['bf16'],
        max_steps=config_training['training']['max_steps'],
        logging_steps=config_training['wandb']['log_every'],
        logging_first_step=True,
        logging_dir="./logs",
        eval_strategy=evaluation_strategy,
        eval_steps=None if evaluation_strategy == "no" else eval_steps,
        save_steps=save_steps,
        learning_rate=None if config_training['training']['optimizer'].lower() == "adafactor" else config_training['training']['learning_rate'],
        weight_decay=config_training['training']['weight_decay'],
        warmup_steps=config_training['training']['warmup_steps'],
        run_name=config_training['wandb']['run_name'],
        remove_unused_columns=False,
        report_to="wandb",
        load_best_model_at_end=load_best_model,
        metric_for_best_model="eval_loss" if load_best_model else None,
        greater_is_better=False if load_best_model else None,
        max_grad_norm=config_training['training']['max_grad_norm'] if config_training['training']['optimizer'].lower() == "adamw" else None,
        optim=config_training['training']['optimizer'].lower(),  # Pass "adamw" or "adafactor"
        dataloader_num_workers=config_training['training']['num_workers'],
    )

    trainer = LCMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=LCMCollator(),
        compute_metrics=compute_metrics,
        config_dict=config_training,
        inspection_decoder=inspection_decoder,
    )

    print("\n🚀 Starting training...")
    if resume_path is not None and resume_path != "None" and os.path.exists(resume_path):
        print(f"📦 Resuming training from checkpoint: {resume_path}")  
        trainer.train(resume_from_checkpoint=resume_path)
    elif resume_from_checkpoint is not None and resume_from_checkpoint != "None" and not os.path.exists(resume_from_checkpoint):
        print(f"❌ Checkpoint file not found: {resume_from_checkpoint}")
        print("🐣 Starting training from scratch...")
        trainer.train()
    else:
        print("🐣 Starting training from scratch...")
        trainer.train()

    # trainer.get_train_dataloader = lambda: train_dataloader
    # trainer.get_eval_dataloader = lambda eval_dataset=None: eval_dataloader
    trainer.add_callback(NaNGradChecker())

    print("✅ Training complete!")
 

def LoadConfig(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def Main():
    print("🚀 Starting LexaLCM Pre2 Training")

    # CLI Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", action="store_true", help="Run a single batch through the model for sanity check.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print additional debug output.")
    parser.add_argument("-i", "--inspect-embeddings", action="store_true", help="Intercept and decode batch embeddings before model forward, then decode the output embeddings.")
    parser.add_argument("-p", "--periodic-inspection", action="store_true", help="A 250th-step inspection of embeddings before model forward and after PostNet_D_Down.")
    args = parser.parse_args()

    # Load Config
    print("🔍 Loading config...")
    config_path = 'src/LexaLCM/Config/Pretrain/Config_Pretrain_Pre2.yaml'
    config_training = LoadConfig(config_path)

    if args.verbose:
        print(f"Config loaded: keys = {list(config_training.keys())}")
        print("🪵 Verbose mode ON")

    # Init WandB
    if "wandb" in config_training:
        wandb.init(
            project=config_training["wandb"]["project"],
            name=config_training["wandb"].get("run_name", None),
            config=config_training
        )

    # Init Model
    resume_path = config_training["training"].get("resume_from", None)

    model_config = LexaLCMConfig()
    model = LexaLCM(model_config)

    # Inspection Decoder
    sonar_decoder_pipeline = EmbeddingToTextPipeline(language="eng_Latn", verbose=True, dtype=torch.float32)
    inspection_steps = 10

    # Inspect Embeddings - If the argument is set, pass an embedding decoder to the model to trigger inspection of the embeddings before they are passed to the PreNet_C
    if args.inspect_embeddings:
        print("🔍 Inspecting SONAR embeddings at every step...")
        model.inspection_decoder = sonar_decoder_pipeline 
    else:
        model.inspection_decoder = None

    # Periodic Inspection - If the argument is set, trigger inspection of the embeddings before model forward and after PostNet_D_Down every n steps (e.g., 250)
    if args.periodic_inspection:
        print(f"🔍 Inspecting SONAR embeddings at every {inspection_steps} steps...")
        model.periodic_inspection = inspection_steps
        model.inspection_decoder = sonar_decoder_pipeline
        training_inspection_decoder = sonar_decoder_pipeline
    else:
        model.periodic_inspection = None
        model.inspection_decoder = None
        training_inspection_decoder = None

    
    # Dry Run
    if args.dry_run:
        print("🧪 Running dry run with test embeddings...")
        dataset = LCMDataset_DryRun()

        RunTraining(config_training, model, train_dataset=dataset, dry_run=True, inspection_decoder=training_inspection_decoder)
        return

    # Full Training
    print("📚 Loading full training datasets...")
    data_conf = config_training["data"]
    max_len = config_training['training'].get('max_seq_len', None)
    if args.verbose:
        print(f"Max sequence length: {max_len}")

    train_dataset = LCMDataset(
        data_dir=data_conf["data_dir"],
        split=data_conf["train_split"],
        text_column=data_conf["text_column"],
        max_seq_len=max_len
    )
    val_dataset = LCMDataset(
        data_dir=data_conf["data_dir"],
        split=data_conf["val_split"],
        text_column=data_conf["text_column"],
        max_seq_len=max_len,
        sample_size=500
    )

    RunTraining(config_training, model, train_dataset, val_dataset, dry_run=False, resume_path=resume_path, inspection_decoder=training_inspection_decoder)

if __name__ == "__main__":
    Main()
