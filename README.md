# 📦 각 Directory 설명

## 01_outputs
directory where the results of quantization are saved

### Save 형식
[model_name] > v[vector_length]_c[number_of_centroids] > implemented_time
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
- eg. 0번, 1번 GPU 사용하려면 gpu=0,1 & num_gpu=2

## 주요 Quantization 변수들 설정
1. vector_lens (outlier_vector_length vector_length) : 한번에 묶을 벡터의 길이를 설정
> outlier 기능을 사용하지 않을때는 outlier_vector_length = -1 로 두기
2. num_centroids (outlier_centroids centroids) : 하나의 코드북이 저장할 centroids 갯수 설정
> outlier 기능을 사용하지 않을때는 outlier_centroids = -1 로 두기
3. npercent (int) : 전체 weight 에서 int % 만큼을 outlier 로 설정한다. 
4. num_red_centroids (outlier_residual_centroids residual_centroids) : residual quantization 기능을 켜두었을때 residual quantization 에 사용할 하나의 코드북이 저장 할 centroids 갯수 설정
> residual quantization 기능을 사용하지 않을땐 -1 -1 로 두기
5. vector_quant_dim (in 또는 out) : VQ dimension 정하기
> in = ich 방향으로 벡터 묶기 / out = och 방향으로 벡터 묶기
6. enable_transpose : ich 방향으로 벡터를 묶으면 enable_transpose = False, och 방향으로 묶으면 True 로 두기
7. bitwidth : codebook quantize 를 몇비트로 할지 정하기 (16으로 두면 codebook quantization 이 실행되지 않는다)

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
- bitwidth = 16 (codebook quantization 실행 안 함)

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