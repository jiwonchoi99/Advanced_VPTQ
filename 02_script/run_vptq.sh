#!/bin/bash

# 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=$HOME/glibc-2.34/lib:$LD_LIBRARY_PATH
export HF_HUB_DOWNLOAD_TIMEOUT=600
export WANDB_SILENT=true

# 파일 디스크립터 한계 증가
ulimit -n 65536


echo -e "\n\nRunning\n\n"

v=6 #vector_lens
c=4096 #codebook 에 있는 number of centroids
gpu=1,3,6
num_gpu=3 #위에 gpu 갯수랑 일치해야한다.

echo "Running Command 1"

CUDA_VISIBLE_DEVICES=${gpu} python /home/sslunder52/project/Advaced_VPTQ/03_codes/VPTQ/run_vptq.py \
        --model_name meta-llama/Llama-3.1-8B \
        --seq_len 2048 \
        --quant_step 1 \
        --blocksize 128 \
        --output_dir /home/sslunder52/project/Advaced_VPTQ/01_outputs/Llama-3.1-8B/v${v}_c${c} \
        --seed 0 \
        --hessian_path /home/sslunder41/project/VPTQ/hess/Hessians-Llama-31-8B-Instruct-6144-8k \
        --inv_hessian_path /home/sslunder41/project/VPTQ/invhess/InvHessians-Llama-31-8B-Instruct-6144-8k \
        --num_gpus ${num_gpu} \
        --eval_nsamples 128 \
        --vector_lens 4 ${v} \
        --num_centroids 4096 ${c} \
        --outlier_size 0 \
        --npercent 1 \
        --num_res_centroids -1 -1 \
        --group_num 16 \
        --group_size -1 \
        --kiter 100 \
        --ktol 1e-5 \
        --kmeans_mode hessian \
        --enable_norm False \
        --norm_dim 1 \
        --save_model False \
        --save_packed_model True \
        --save_qlinear True \
        --new_eval True \
        --eval_mode False \
        --enable_perm True \
        --enable_residual False \
        --vector_quant_dim out \
        --enable_transpose True \
        --bitwidth 8 \
        --bsize 1024 \
                


# * : 주요 변수들
#model_name meta-llama/Llama-3.1-8B : original model name from Hugging Face () 
#seq_len 2048 : sequence length used during evaluation
#quant_step 1 : number of columns (in ich direction) in one tile during error propagation
#blocksize 128 : size of block during error propagation
#output_dir : quantization 결과 저장되는 경로
#seed 0 : (고정값)
#hessian_path : hessian 경로
#inv_hessian_path : inv_hessian 경로
#*num_gpus ${num_gpu} : number of gpus to be used
#eval_nsamples : number of samples to be evaluated
#*vector_lens : int int / 첫번째 int = outlier vector length / 두번째 int = vector length
#*num_centroids : int int / 첫번째 int = number of outlier centroids / 두번째 int = number of centroids
#*npercent : percent of outlier 
#*num_res_centroids : int int / 첫번째 int = number of outlier centroids for residual VQ (구현 이슈로 -1 로 고정)
        # 두번째 int = number of centroids for residual VQ 
#*group_num : number of groups in one weight 
#kiter : number of iterations when initializing centroids
#ktol : k-mean clustering 에서 tolerance 값
#kmeans_mode : normal, hessian 둘 중 하나 선택 
#enable_norm : enable normalization during quantization
#*norm_dim : dimension of normalization (0 = och direction, 1 = ich direction)
#save_model : save quantized model
#save_packed_model : save packed model 
#save_qlinear : save qlinear for every layer during quantization (이 기능을 켜두면 quantization 중간에 에러가 떠도 다시 실행이 가능하다)
#new_eval True : 
#eval_mode : quantization 이 다 끝났을때 saved 된 모델로 다시 evaluation 을 돌리고 싶다면 eval_mode = True 로 두고 실행
#*enable_perm : enable permutation based on hessian matrix
#enable_residual : enable residual VQ
#*vector_quant_dim : vector quantization dimension (in, out) / in = ich direction & out = och direction
#*enable_transpose : vector_quant_dim 이 in 일 경우에는 enable_transpose 를 True 로 두기
        # vector_quant_dim 이 out 일 경우에는 enable_transpose 를 False 로 두기
#*bitwidth : bit width when quantizing codebook (16 = do not quantize codebook)
#*bsize 1024 : codebook quantization 할 때 하나의 scale 을 share 하는 group