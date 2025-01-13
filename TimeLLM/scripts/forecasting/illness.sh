model_name=Time-LLM
train_epochs=100
llama_layers=32
master_port=200097
num_process=64
batch_size=4
d_model=32
d_ff=128
seq_len=104


comment='TimeLLM-Illness'

#### Att and Trans use Learning rate 1e-6 or 1e-5.
#### you can increase the batch size higher if the LLm is not LLaMa and Random

for pred_len in 24 36 48 60
do
for percent in 100
do
for LLM in 'LLaMa' 'Random' 'Linear' 'Att' 'Trans' 'NoLLM'
do
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 --main_process_port=29600 run_main_multiGPU_forecasting.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/forecasting/illness/ \
    --data_path national_illness.csv \
    --model_id illness_$model'_'$pred_len \
    --model $model_name \
    --data illness \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --train_epochs 5 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1 \
    --patch_len 24 \
    --stride 8 \
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
done