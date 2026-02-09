set -euo pipefail

# Ensure correct conda env (torch/transformers/peft installed)
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate vptq_loraft || true
fi

cd ../03_codes/LoRA_Finetuning
export WANDB_PROJECT=LoRA_VPTQ_finetuning   
export CUDA_VISIBLE_DEVICES=1,7                                                 
NCCL_P2P_DISABLE=1                                                   
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=12355 LoRA_VPTQ_Trainer.py \
        --use_multi_gpu \
        --model_name 'meta-llama/Llama-3.1-8B'                                                                 \
        --quantized_model_path '/home/sslunder52/project/vptq_transposed/01_outputs/Llama-3.1-8B/test6_PackedModel/v6_c8192_b16_och/2026-02-08-21-31-20/packed_model'                             \
        --dataset_name 'wikitext'                                                                       \
        --dataset_config 'wikitext-2-raw-v1'                                                            \
        --output_dir '../01_outputs/llama3-1-8B-wikitext2-lora-finetuned'                                     \
        --run_name 'llama3-1-8B-wikitext2-lora-finetuned'                                                    \
        --max_length 4096                                                                               \
        --batch_size 4                                                                                  \
        --num_epochs 3                                                                                  \
        --learning_rate 2e-4                                                                            \
        --eval_steps 200                                                                                \
        --save_steps 200                                                                                \
        --save_total_limit 10                                                                           \
        --lora_r 16                                                                                     \
        --lora_alpha 32                                                                                 \
        --lora_dropout 0.1                                                                              \
        --gradient_checkpointing                                                                        \
        --world_size 2                                                                                  \
        --master_addr localhost                                                                     \
        --master_port 12355                                                                            \
        | tee ../01_outputs/llama3-2-1B-wikitext2-lora-finetuning-log.txt

                                                                
# --nnodes=1 --nproc_per_node=2 --master_port=12355 : 에서 nproc_per_node 는 gpu 사용하는 갯수 만큼 설정!!                                                               
# model_name                 : Original model name from Hugging Face (for tokenizer)
# quantized_model_path       : Path to VPTQ quantized model directory
# custom_dataset             : Use custom dataset flag
# custom_data_path           : Path to Mobile Agent v3 trajectory data directory
# output_dir                 : Directory to save the fine-tuned model
# run_name                   : W&B run name
# max_length                 : Maximum sequence length for training
# batch_size                 : Training batch size per GPU
# num_epochs                 : Number of training epochs
# learning_rate              : Learning rate for training
# eval_steps                 : Evaluation steps
# save_steps                 : Save steps
# save_total_limit           : Save total limit
# lora_r                     : LoRA rank parameter
# lora_alpha                 : LoRA alpha parameter  
# lora_dropout               : LoRA dropout rate
# use_multi_gpu              : Enable multi-GPU training (add --use_multi_gpu flag)
# world_size                 : Total number of GPUs to use 
# master_addr                : Master address for distributed training
# master_port                : Master port for distributed training

