model_name=Time-LLM
train_epochs=100
llama_layers=32
master_port=200097
num_process=24
batch_size=4
d_model=32
d_ff=32
model_name=Time-LLM
comment='TimeLLM-weather'


#### Att and Trans use Learning rate 1e-6 or 1e-5.
#### you can increase the batch size higher if the LLm is not LLaMa and Random
for pred in '96' '192' '336' '720'
do
for LLM in 'LLaMa' 'Random' 'Linear' 'Att' 'Trans' 'NoLLM'
do
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 --main_process_port=29600 run_main_multiGPU_forecasting.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/nathan/LLM4TS/datasets/forecasting/weather \
  --data_path weather.csv \
  --model_id weather_512_$pred \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len $pred \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $LLM \
  --llm_dim '4096'
done
done