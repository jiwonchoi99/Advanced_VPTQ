"""
HJ
- change from using qwen model to using Llama model 
- vptq_qwen_vl => vptq (library change)
- qwen_vl 모델에서 돌아가는걸 Llama 에서 돌아가게 LoRATRainer.py 에 있는 코드를 참고해서 필요한 library 와 함수를 가져왔다
- dataset 도 custom dataset 이 아니라 LoRATrainer 에서 사용하는 Wikitext dataset 을 가져오도록 바꾸었다
"""

# Basic
import os
import argparse
import math
import wandb
import warnings
import torch
import torch.nn as nn
import pdb
import numpy as np
import os.path as osp
import sys

# Multi-GPU
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Model & Dataset
from transformers import (
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig
import torch.nn.functional as F

from lib import get_peft_model, load_amem_dataset

# Import VPTQ Llama
_vptq_path = osp.abspath(osp.join(osp.dirname(__file__), '/home/sslunder52/project/vptq_transposed/03_codes/'))
if _vptq_path not in sys.path:
    sys.path.insert(0, _vptq_path)
import vptq  # type: ignore[import] #Llama uses vptq instead of vptq_qwen_vl

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -------------------------------
# LoRA Attachment Validation
# -------------------------------
def log_lora_attachment_summary(model, max_examples: int = 15):
    """
    Print a concise summary of whether LoRA adapters were attached, and specifically
    how many were attached to VPTQ's VQuantLinear layers.
    """
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer  # type: ignore[import]
    except Exception:
        BaseTunerLayer = None

    # Try to import VQuantLinear (optional; only used for counting)
    VQuantLinear = None
    try:
        from vptq.layers.vqlinear import VQuantLinear as _VQL  # type: ignore[import]
        VQuantLinear = _VQL
    except Exception:
        pass

    tuner_layers = []
    vql_tuner_layers = []
    examples = []

    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    for name, module in model.named_modules():
        if BaseTunerLayer is not None and isinstance(module, BaseTunerLayer):
            tuner_layers.append((name, module))
            # For LoRA layers, base layer is usually accessible via get_base_layer()
            base = None
            if hasattr(module, "get_base_layer"):
                try:
                    base = module.get_base_layer()
                except Exception:
                    base = None
            if VQuantLinear is not None and base is not None and isinstance(base, VQuantLinear):
                vql_tuner_layers.append((name, module))
                if len(examples) < max_examples:
                    examples.append(name)

    print("\n[LoRA attachment summary]")
    print(f"- total params: {total_params:,}")
    print(f"- trainable params: {trainable_params:,} ({(trainable_params/total_params*100 if total_params else 0):.4f}%)")
    print(f"- tuner layers found: {len(tuner_layers)}")
    if VQuantLinear is None:
        print("- VQuantLinear import: failed (cannot count VQuantLinear-attached LoRA precisely)")
    else:
        print(f"- LoRA-on-VQuantLinear layers: {len(vql_tuner_layers)}")
        if examples:
            print(f"- examples (first {len(examples)}): {examples}")
    if len(tuner_layers) == 0:
        print("WARNING: No PEFT tuner layers found. LoRA may not have been attached.")
    print("")

# -------------------------------
# Configuration
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="LoRA VPTQ Fine-tuning with Llama")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                       help="Original model name from Hugging Face (for tokenizer)")
    parser.add_argument("--quantized_model_path", type=str, required=True,
                       help="Path to VPTQ quantized model")
    parser.add_argument("--custom_data_path", type=str, default=None,
                       help="Path to custom dataset directory (Mobile Agent v3 trajectory data)")
    parser.add_argument("--custom_dataset", action="store_true", 
                       help="Use custom dataset")
    parser.add_argument("--dataset_name", type=str, default="wikitext", 
                       help="Dataset name (if not using custom dataset)")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", 
                       help="Dataset configuration")
    parser.add_argument("--output_dir", type=str, default="./llama-3.1-vptq-lora-finetuned", 
                       help="Output directory for model")
    parser.add_argument("--run_name", type=str, default="run_name", 
                       help="W&B run name")

    parser.add_argument("--max_length", type=int, default=2048, 
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, 
                       help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=500, 
                       help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=500, 
                       help="Save steps")
    parser.add_argument("--save_total_limit", type=int, default=3, 
                       help="Save total limit")
    parser.add_argument("--lora_r", type=int, default=16, 
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, 
                       help="LoRA dropout")

    # Memory/perf knobs (important for long-context VL training)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce activation/weight-save memory during backward.",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (not recommended for long max_length).",
    )
    
    # Multi-GPU arguments
    parser.add_argument("--use_multi_gpu", action="store_true", 
                       help="Enable multi-GPU training")
    parser.add_argument("--local_rank", type=int, default=-1, 
                       help="Local rank for distributed training")
    parser.add_argument("--world_size", type=int, default=1, 
                       help="Total number of processes")
    parser.add_argument("--master_addr", type=str, default="localhost", 
                       help="Master address for distributed training")
    parser.add_argument("--master_port", type=str, default="12355", 
                       help="Master port for distributed training")
    
    return parser.parse_args()

# -------------------------------
# Multi-GPU Setup
# -------------------------------
def setup_distributed(args):
    """Setup distributed training"""
    if args.use_multi_gpu:
        # Initialize distributed training
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        
        # Set the device
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        return device, local_rank
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

# -------------------------------
# LoRA Configuration
# -------------------------------
def create_lora_config(args):
    """Create LoRA configuration"""
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,  # Use Rank-Stabilized LoRA
    )

# -------------------------------
# Dataset Preprocessing
# -------------------------------
def preprocess_dataset(tokenizer, dataset, max_length):
    """Preprocess dataset for causal language modeling"""
    def tokenize_function(examples):
        enc = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        labels = []
        for ids, mask in zip(input_ids, attn):
            labels.append([tok if m == 1 else -100 for tok, m in zip(ids, mask)])

        enc["labels"] = labels
        return enc

    # Filter out empty texts
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset

"""
Class Qwen25VLCollator
Class MobileAgentV3VLTorchDataset 
def loag_mobile_agent_v3_vl_dataset_for_training
제거
"""


# -------------------------------
# Perplexity Callback
# -------------------------------

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            loss_val = metrics["eval_loss"]
            if loss_val is None or (isinstance(loss_val, float) and (math.isnan(loss_val) or math.isinf(loss_val))):
                ppl = float("nan")
            else:
                try:
                    ppl = math.exp(loss_val)
                except OverflowError:
                    ppl = float("inf")
            metrics["eval_perplexity"] = ppl
            if os.getenv("WANDB_API_KEY"):
                try:
                    wandb.log({"eval/perplexity": ppl, "step": state.global_step})
                except Exception:
                    pass
            print(f"[Eval] Perplexity: {ppl:.4f}")

# -------------------------------
# Main Training Function
# -------------------------------
def main():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    args = parse_args()
    
    # Setup distributed training
    device, local_rank = setup_distributed(args)
    
    # Only print on main process
    if not args.use_multi_gpu or local_rank == 0:
        print(f"🚀 Starting LoRA Fine-tuning with VPTQ Quantized Model")
        print(f"📁 Quantized model path: {args.quantized_model_path}")
        if args.use_multi_gpu:
            print(f"🌐 Multi-GPU training enabled with {torch.cuda.device_count()} GPUs")
    
    # -------------------------------
    # Load Model and Tokenizer
    # -------------------------------
    if not args.use_multi_gpu or local_rank == 0:
        print("📥 Loading VPTQ quantized model and tokenizer...")
    
    # Load tokenizer + processor from original HF model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # causal LM recommended

    """
    Processor for Qwen2.5-VL
    제거
    """
    
    # Load VPTQ quantized model
    if not osp.exists(osp.join(args.quantized_model_path, "config.json")):
        raise ValueError(f"Quantized model not found at {args.quantized_model_path}")
    
    # Check if it's a Qwen-VL model
    import json
    with open(osp.join(args.quantized_model_path, "config.json"), 'r') as f:
        config = json.load(f)

    # Always load as AutomodelForCausalLM for Llama
    model = vptq.AutoModelForCausalLM.from_pretrained(
        args.quantized_model_path,
        device=str(device),
        torch_dtype=torch.bfloat16,

    )

    #pdb.set_trace() (to check if model is correctly loaded)

    if not args.use_multi_gpu or local_rank == 0:
        print(f"✅ Successfully loaded VPTQ quantized model from {args.quantized_model_path}")

    if getattr(model.config, "pad_token_id", None) != tokenizer.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Training stability + memory knobs
    # - use_cache must be False for gradient checkpointing
    # - enabling checkpointing is critical when max_length is large (e.g., 2048/4096)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Prefer enabling gradient checkpointing (can be disabled explicitly via --no_gradient_checkpointing)
    enable_gc = True
    if getattr(args, "no_gradient_checkpointing", False):
        enable_gc = False
    if getattr(args, "gradient_checkpointing", False):
        enable_gc = True
    if enable_gc and hasattr(model, "gradient_checkpointing_enable"):
        try:
            # Transformers recommends use_reentrant=False for many modern models.
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            # Backward compatible: older versions don't accept kwargs.
            model.gradient_checkpointing_enable()
    elif (not enable_gc) and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    
    # Create LoRA config
    lora_config = create_lora_config(args)
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Validate LoRA attachment (only on main process)
    if not args.use_multi_gpu or local_rank == 0:
        log_lora_attachment_summary(model)
    
    # Print trainable parameters (only on main process)
    if not args.use_multi_gpu or local_rank == 0:
        model.print_trainable_parameters()
        print(model)
    
    # -------------------------------
    # Load and Preprocess Dataset
    # -------------------------------
    if not args.use_multi_gpu or local_rank == 0:
        print("📊 Loading and preprocessing dataset...")
    
    """
    Dataset 가져오는 코드를 LoRATrainer.py 에서 가지고 왔다
    """

    # Load dataset
    if args.custom_dataset:
        if "amem" in args.dataset_name:
            dataset = load_amem_dataset(args.dataset_config, args.dataset_name.replace("amem_", ""))
        else:
            dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    
    # Preprocess dataset
    train_dataset = preprocess_dataset(tokenizer, dataset["train"], args.max_length)
    eval_dataset = preprocess_dataset(tokenizer, dataset["validation"], args.max_length)

    # -------------------------------
    # Training Configuration
    # -------------------------------
    if not args.use_multi_gpu or local_rank == 0:
        print("⚙️ Setting up training configuration...")
    
    # IMPORTANT: enforce per-device batch size=1 for Qwen2.5-VL multimodal inputs
    if "qwen" in args.model_name:
        per_device_bs = 1
    else:
        per_device_bs = args.batch_size

    grad_accum = max(1, int(args.batch_size))
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size, 
        per_device_eval_batch_size=args.batch_size, #
        gradient_accumulation_steps=grad_accum,
        warmup_steps=100,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,  # Use bfloat16 for better stability
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else None,
        run_name=args.run_name,
        remove_unused_columns=False, 
        gradient_checkpointing=enable_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if enable_gc else None,
        # Multi-GPU specific settings
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )
    
    "Qwen25VLCollator() 에서 Transformer.DataCollatorForLanguageModeling() 으로 바꿨다"

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # -------------------------------
    # Initialize Trainer
    # -------------------------------
    if not args.use_multi_gpu or local_rank == 0:
        print("🎯 Initializing trainer...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    # Add Perplexity Callback
    trainer.add_callback(PerplexityCallback())
    
    # -------------------------------
    # Training
    # -------------------------------
    if not args.use_multi_gpu or local_rank == 0:
        print("🚀 Starting training...")
        if args.use_multi_gpu:
            world = torch.cuda.device_count()
            effective_batch_size = per_device_bs * grad_accum * world
            print(f"📊 Effective batch size: {effective_batch_size} (per-device={per_device_bs}, grad_accum={grad_accum}, gpus={world})")
    
    try:
        print("🔍 Running baseline evaluation before training...")
        baseline_metrics = trainer.evaluate()
        if "eval_loss" in baseline_metrics:
            try:
                baseline_ppl = math.exp(baseline_metrics["eval_loss"])
            except OverflowError:
                baseline_ppl = float("inf")
            print(f"📊 Baseline Perplexity: {baseline_ppl:.4f}")
            if os.getenv("WANDB_API_KEY"):
                wandb.log({"eval/baseline_perplexity": baseline_ppl, "step": 0})

        # Free as much as possible before training (baseline eval can leave cached blocks)
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        # Train the model
        trainer.train()
        
        # Save the final model (only on main process)
        if not args.use_multi_gpu or local_rank == 0:
            trainer.save_model()
            tokenizer.save_pretrained(args.output_dir)
            print(f"✅ Training completed! Model saved to {args.output_dir}")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    finally:
        # Cleanup distributed training
        cleanup_distributed()


if __name__ == "__main__":
    main()

