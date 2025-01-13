seq_len=96
model=CALF


for pred_len in 720
do

CUDA_VISIBLE_DEVICES=2 python /home/nathan/LLM4TS/Imputation_task/CALF/run.py \
    --root_path /home/nathan/LLM4TS/datasets/forecasting/ETT-small  \
    --data_path ETTh1.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
    --data ETTh1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 1 \
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
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 10 \
    --LLM GPT2 \
    --gpu_fraction 1


# python /home/nathan/LLM4TS/Imputation_task/CALF/run.py \
#     --root_path /home/nathan/LLM4TS/datasets/forecasting/ETT-small  \
#     --data_path ETTh1.csv \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
#     --data ETTh1 \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --batch_size 64 \
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
#     --r 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --patience 10 \
#     --LLM Linear \
#     --gpu_fraction 0.2


python /home/nathan/LLM4TS/Imputation_task/CALF/run.py \
    --root_path /home/nathan/LLM4TS/datasets/forecasting/ETT-small  \
    --data_path ETTh1.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
    --data ETTh1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 64 \
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
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 10 \
    --LLM Random \
    --gpu_fraction 1


# python /home/nathan/LLM4TS/Imputation_task/CALF/run.py \
#     --root_path /home/nathan/LLM4TS/datasets/forecasting/ETT-small  \
#     --data_path ETTh1.csv \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
#     --data ETTh1 \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --batch_size 64 \
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
#     --r 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --patience 10 \
#     --LLM Trans \
#     --gpu_fraction 0.2



# python /home/nathan/LLM4TS/Imputation_task/CALF/run.py \
#     --root_path /home/nathan/LLM4TS/datasets/forecasting/ETT-small  \
#     --data_path ETTh1.csv \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
#     --data ETTh1 \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --batch_size 64 \
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
#     --r 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --patience 10 \
#     --LLM NoLLM \
#     --gpu_fraction 0.2


# python /home/nathan/LLM4TS/Imputation_task/CALF/run.py \
#     --root_path /home/nathan/LLM4TS/datasets/forecasting/ETT-small  \
#     --data_path ETTh1.csv \
#     --is_training 1 \
#     --task_name long_term_forecast \
#     --model_id ETTh1_$model'_'$seq_len'_'$pred_len \
#     --data ETTh1 \
#     --seq_len $seq_len \
#     --label_len 0 \
#     --pred_len $pred_len \
#     --batch_size 64 \
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
#     --r 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --patience 10 \
#     --LLM Att \
#     --gpu_fraction 0.2

echo '====================================================================================================================='
done