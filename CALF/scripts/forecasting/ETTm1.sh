export CUDA_VISIBLE_DEVICES=0
gpu_fraction=0.15
batch_size=64
seq_len=96
model=CALF

for pred_len in 96 192 336 720
do

python /home/nathan/LLM4TS/Imputation_task/CALF/run.py \
    --root_path /home/nathan/LLM4TS/datasets/forecasting/ETT-small  \
    --data_path ETTm1.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTm1_$model'_'$seq_len'_'$pred_len \
    --data ETTm1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size $batch_size \
    --learning_rate 0.0005 \
    --lradj type1 \
    --train_epochs 5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 20 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5 \
    --LLM GPT2 \
    --gpu_fraction $gpu_fraction


python /home/nathan/LLM4TS/Imputation_task/CALF/run.py \
    --root_path /home/nathan/LLM4TS/datasets/forecasting/ETT-small  \
    --data_path ETTm1.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTm1_$model'_'$seq_len'_'$pred_len \
    --data ETTm1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size $batch_size \
    --learning_rate 0.0005 \
    --lradj type1 \
    --train_epochs 5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 20 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5 \
    --LLM Random \
    --gpu_fraction $gpu_fraction


# python /home/nathan/LLM4TS/Imputation_task/CALF/run.py \
#     --root_path /home/nathan/LLM4TS/datasets/forecasting/ETT-small  \
#     --data_path ETTm1.csv \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model_id ETTm1_$model'_'$seq_len'_'$pred_len \
#     --data ETTm1 \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --batch_size $batch_size \
#     --learning_rate 0.0005 \
#     --lradj type1 \
#     --train_epochs 5 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 0.3 \
#     --enc_in 7 \
#     --c_out 7 \
#     --gpt_layer 6 \
#     --itr 1 \
#     --model $model \
#     --cos 1 \
#     --tmax 20 \
#     --r 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --patience 5 \
#     --LLM Linear \
#     --gpu_fraction $gpu_fraction


# python /home/nathan/LLM4TS/Imputation_task/CALF/run.py \
#     --root_path /home/nathan/LLM4TS/datasets/forecasting/ETT-small  \
#     --data_path ETTm1.csv \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model_id ETTm1_$model'_'$seq_len'_'$pred_len \
#     --data ETTm1 \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --batch_size $batch_size \
#     --learning_rate 0.0005 \
#     --lradj type1 \
#     --train_epochs 5 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 0.3 \
#     --enc_in 7 \
#     --c_out 7 \
#     --gpt_layer 6 \
#     --itr 1 \
#     --model $model \
#     --cos 1 \
#     --tmax 20 \
#     --r 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --patience 5 \
#     --LLM Att \
#     --gpu_fraction $gpu_fraction

# python /home/nathan/LLM4TS/Imputation_task/CALF/run.py \
#     --root_path /home/nathan/LLM4TS/datasets/forecasting/ETT-small  \
#     --data_path ETTm1.csv \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model_id ETTm1_$model'_'$seq_len'_'$pred_len \
#     --data ETTm1 \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --batch_size $batch_size \
#     --learning_rate 0.0005 \
#     --lradj type1 \
#     --train_epochs 5 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 0.3 \
#     --enc_in 7 \
#     --c_out 7 \
#     --gpt_layer 6 \
#     --itr 1 \
#     --model $model \
#     --cos 1 \
#     --tmax 20 \
#     --r 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --patience 5 \
#     --LLM Trans \
#     --gpu_fraction $gpu_fraction

# python /home/nathan/LLM4TS/Imputation_task/CALF/run.py \
#     --root_path /home/nathan/LLM4TS/datasets/forecasting/ETT-small  \
#     --data_path ETTm1.csv \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model_id ETTm1_$model'_'$seq_len'_'$pred_len \
#     --data ETTm1 \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --batch_size $batch_size \
#     --learning_rate 0.0005 \
#     --lradj type1 \
#     --train_epochs 5 \
#     --d_model 768 \
#     --n_heads 4 \
#     --d_ff 768 \
#     --dropout 0.3 \
#     --enc_in 7 \
#     --c_out 7 \
#     --gpt_layer 6 \
#     --itr 1 \
#     --model $model \
#     --cos 1 \
#     --tmax 20 \
#     --r 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --patience 5 \
#     --LLM NoLLM \
#     --gpu_fraction $gpu_fraction
echo '====================================================================================================================='
done