model_name=Time-LLM
d_model=16
d_ff=32
llama_layers=6
master_port=200097
num_process=24
gpu_index=0
batch_size=64
comment=TimeLLM-SWAT


for LLM in 'LLM' 'Ramdom' 'Linear' 'Att' 'Trans' 'NoLLM'
do
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 run_main_multiGPU_anomaly.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path datasets/anomaly_detection/SWaT \
  --model_id SWAT_Time-LLM \
  --data SWAT \
  --features M \
  --seq_len 512 \
  --pred_len 0 \
  --n_heads 4 \
  --dropout 0.3 \
  --enc_in 51 \
  --c_out 51 \
  --itr 1 \
  --anomaly_ratio 0.5 \
  --patience 3 \
  --learning_rate 0.0001 \
  --train_epochs 5 \
  --lradj type1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --llm_layers $llama_layers \
  --batch_size $batch_size \
  --model $model_name \
  --model_comment $comment \
  --llm_model $LLM
done