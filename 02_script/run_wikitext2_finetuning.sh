cd ../01_codes
export WANDB_PROJECT=LoRA_finetuning   
export CUDA_VISIBLE_DEVICES=0                                                     
NCCL_P2P_DISABLE=1                                                   
python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=12355 LoRATrainer.py   \
        --model_name 'meta-llama/Llama-3.2-1B'                                                      \
        --dataset_name 'wikitext'                                                                   \
        --dataset_config 'wikitext-2-raw-v1'                                                        \
        --output_dir '../03_outputs/llama3-2-1B-wikitext2-lora-finetuned'                           \
        --run_name 'Llama-3.2-1B-wikitext2-lora'                                                    \
        --max_length 512                                                                            \
        --batch_size 16                                                                             \
        --num_epochs 3                                                                              \
        --learning_rate 2e-4                                                                        \
        --eval_steps 200                                                                            \
        --save_steps 200                                                                            \
        --save_total_limit 10                                                                       \
        --lora_r 64                                                                                 \
        --lora_alpha 32                                                                             \
        --lora_dropout 0.1                                                                          \
        --world_size 1                                                                              \
        --master_addr localhost                                                                     \
        --master_port 12355                                                                         \
        | tee ../03_outputs/llama3-2-1B-wikitext2-lora-finetuning-log.txt

                                                                
# model_name                 : 'meta-llama/Llama-2-7b' or other Hugging Face models
# dataset_name               : 'wikitext', 'amem_keywords_tags', 'amem_content'
# dataset_config             : Dataset configuration (e.g., 'wikitext-2-raw-v1')
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
# use_multi_gpu              : Enable multi-GPU training
# world_size                 : Total number of GPUs to use
# master_addr                : Master address for distributed training
# master_port                : Master port for distributed training                                     