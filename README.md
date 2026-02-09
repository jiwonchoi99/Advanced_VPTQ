# 📦 각 Directory 설명

## 01_outputs
directory where the results of quantization are saved

### Save 형식
[model_name] > v[vector_length]_c[number_of_centroids] > implemented_time

예를 들어 /01_outputs/Llama-3.1-8B/v4_c4096/2026-01-28-13-30-02 이면, Llama-3.1-8B 모델을 vector length 4와 number of centroids 를 4096 으로 VPTQ 를 적용한 모델이고 2026-01-28 13시30분02초에 실행 시작했던 파일이다.
- logs : 각 gpu 의 로그 파일
- model : Vector Quantization 이 적용된 model (script 에서 save_model = True 로 두었을때만 생성된다)
- packed_model : index packing 이 적용된 vector quantize model (script 에서 save_packed_model = True 로 두었을때만 생성된다)
- ppl_results.json : perplexity result of quantized model

## 02_script
- run_vptq.sh : 특정 모델을 VPTQ 로 Vector Quantize 할 때 쓰이는 script
- run_vptq_lora_finetuning.sh : VPTQ 로 Vector Quantize 된 모델을 LoRA Finetuning 시킬 떄 쓰이는 script

## 03_codes
- LoRA_Finetuning : codes for LoRA_Finetuning
- VPTQ : codes for VPTQ

---

# 🚀 How to implement VPTQ quantization
02_scripts -> run_vptq.sh 스크립트 파일 열어서 변수들 설정하기!

## Multi-gpu 세팅
1. line 20 에 사용할 GPU number 랑 line21 에 몇 개 gpu 사용하는지 적기
```
gpu=1,3,6
num_gpu=3
```

## 주요 Quantization 변수들 설정
1. vector_lens (outlier_vector_length   vector_length) : 한번에 묶을 벡터의 길이를 설정
> outlier 기능을 사용하지 않을때는 outlier_vector_length = -1 로 두기
2. num_centroids (outlier_centroids   centroids) : 하나의 코드북이 저장할 centroids 개수 설정
> outlier 기능을 사용하지 않을때는 outlier_centroids = -1 로 두기
3. npercent (int) : 전체 weight 에서 int % 만큼을 outlier 로 설정한다. 
4. num_red_centroids (outlier_residual_centroids   residual_centroids) : residual quantization 에 사용 할 하나의 코드북이 저장 할 centroids 개수 설정
> residual quantization 기능을 사용하지 않을땐 -1 -1 로 두기
5. vector_quant_dim (in 또는 out) : VQ dimension 정하기
> in = ich 방향으로 벡터 묶기 / out = och 방향으로 벡터 묶기
6. enable_transpose (bool) : ich 방향으로 벡터를 묶으면 enable_transpose = False, och 방향으로 묶으면 True 로 두기
7. bitwidth (int) : codebook quantize 를 몇 비트로 할지 정하기 (16으로 두면 codebook quantization 이 실행되지 않는다)

---

## 실행 예시
- outlier 과 residual vector quantization 기능 끄고, bpv 를 3으로 두기 위해 vector length = 4, codebook size = 2^12, number of groups = 4 으로 설정한 뒤에, och 방향으로 벡터를 묶어서 VPTQ 를 실행하고, codeobok quantization 은 8bit 로 실행하려면 아래와 같이 변수를 설정한다. 
- 그리고 terminal 에 ./run_vptq.sh 입력

```
v=4
c=4096 
...
--vector_lens -1 ${v} \ 
--num_centroids -1 ${c} \
...
--npercent 0 \
--num_res_centroids -1 -1 \
--group_num 4 \
...
--vector_quant_dim out \
--enable_transpose True \
--bitwidth 8 \
...
```


## 다른 예시
논문에서 나온 Table 8 의 Llama3-8B 2.24bit 로 설정하기 위해서는 아래와 같이 설정하기 (2.24 bpv)
- outlier vector length = 4, vector length = 6
- number of outlier centroids = 4096, number of centroids = 4096
- npercent = 1 (outlier 는 전체의 1 percent)
- residual quantization 기능 끄기 (num_red_centrodis 를 -1 -1 로 세팅)
- number of groups = 16
- quantization dimension = och (vector_quant_dim = out & enable_transpose = True 로 두기)
- bitwidth = 16 (codebook quantization 실행 안함)

```
v=6
c=4096 
...
--vector_lens 4 ${v} \
--num_centroids 4096 ${c} \
...
--npercent 1 \
--num_res_centroids -1 -1 \
--group_num 16 \
...
--vector_quant_dim out \
--enable_transpose True \
--bitwidth 16 \
...
```

## Temrinal 에 나오는 결과 (성공적으로 VPTQ 가 작동할 때)
- {} <= 중괄호 안에 있는 내용은 HJ가 적어둔 comment 이다. 실제 실행시킬때는 terminal 에 나오지 않는다.
- 아래처럼 temrinal 에 뜨면 quantization 이 잘 진행될 것이다.

```
(vptq) sslunder52@pim-gpu06:/home/sslunder52/project/Advaced_VPTQ/02_script$ ./run_vptq.sh {script 파일 실행!}


Running


Running Command 1
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 124.44it/s]
model sequence length: 2048
exp time: 2026-02-09 14:18:30 
{script 에 적어둔 arguments 들이 잘못 들어간게 있는지 확인하기}
args: VPTQArguments(model_name='meta-llama/Llama-3.1-8B', seq_len=2048, quant_step=1, percdamp=0.01, blocksize=128, output_dir='/home/sslunder52/project/Advaced_VPTQ/01_outputs/Llama-3.1-8B/v6_c4096/2026-02-09-14-18-29', seed=0, eval=False, new_eval=True, save_model=False, save_packed_model=True, disable_actorder=False, hessian_path='/home/sslunder41/project/VPTQ/hess/Hessians-Llama-31-8B-Instruct-6144-8k', inv_hessian_path='/home/sslunder41/project/VPTQ/invhess/InvHessians-Llama-31-8B-Instruct-6144-8k', num_gpus=3, eval_nsamples=128, save_qlinear=True, absorb_perm=False, enable_residual=False, eval_mode=False, outlier_size=0)
quant_args: QuantizationArguments(vector_lens=[4, 6], num_centroids=[4096, 4096], num_res_centroids=[-1, -1], npercent=1.0, group_num=16, group_size=-1, kiter=100, ktol=1e-05, kseed=0, kmeans_mode='hessian', kmeans_alpha=0, enable_norm=False, norm_dim=1, enable_perm=True, enable_transpose=True, vector_quant_dim='out', bitwidth=8, bsize=1024)
Starting VPTQ...
model dtype: torch.bfloat16

----quantization start ...---- 2026-02-09 14:20:24
gpu 0 tasks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
gpu 1 tasks: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
gpu 2 tasks: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
gpu 1 {위에 gpu 0 tasks 가 1번 gpu 에 할당된다}
gpu 3 
gpu 6
INFO - ----Quantizing on cuda:0----
INFO - ----Quantizing layer 0 ...---- 2026-02-09 05:20:46 on cuda:0 dtype torch.bfloat16
[['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']]
INFO - dict_keys(['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'])
INFO - load Hessian from /home/sslunder41/project/VPTQ/hess/Hessians-Llama-31-8B-Instruct-6144-8k/0_qkv.pt
INFO - load inv Hessian from /home/sslunder41/project/VPTQ/invhess/InvHessians-Llama-31-8B-Instruct-6144-8k/0_qkv.pt
... 
```