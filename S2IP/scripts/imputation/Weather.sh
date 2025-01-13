model_name=S2IPLLM


for LLM in 'Random' 'Linear' 'Att' 'Trans' 'NoLLM'; do
    python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/weather/ \
    --data_path weather.csv \
    --model_id weather_mask_0.125 \
    --mask_rate 0.125 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --gpt_layer 3 \
    --d_model 768 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 5 \
    --learning_rate 0.0001 \
    --LLM $LLM \
    --patch_size 16 \
    --stride 8 \
    --add_prompt 1 \
    --prompt_length 2 \
    --batch_size 1024 \
    --sim_coef -0.1 \
    --pool_size 1000 \
    --period 24 \
    --percent 100 \
    --trend_length 96 \
    --seasonal_length 48



    python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/weather/ \
    --data_path weather.csv \
    --model_id weather_mask_0.25 \
    --mask_rate 0.25 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --gpt_layer 3 \
    --d_model 768 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 5 \
    --learning_rate 0.0001 \
    --LLM $LLM \
    --patch_size 16 \
    --stride 8 \
    --add_prompt 1 \
    --prompt_length 2 \
    --batch_size 2048 \
    --sim_coef -0.1 \
    --pool_size 1000 \
    --period 24 \
    --percent 100 \
    --trend_length 96 \
    --seasonal_length 48


    python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/weather/ \
    --data_path weather.csv \
    --model_id weather_mask_0.375 \
    --mask_rate 0.375 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --gpt_layer 3 \
    --d_model 768 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 5 \
    --learning_rate 0.0001 \
    --LLM $LLM \
    --patch_size 16 \
    --stride 8 \
    --add_prompt 1 \
    --prompt_length 2 \
    --batch_size 2048 \
    --sim_coef -0.1 \
    --pool_size 1000 \
    --period 24 \
    --percent 100 \
    --trend_length 96 \
    --seasonal_length 48


    python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/weather/ \
    --data_path weather.csv \
    --model_id weather_mask_0.5 \
    --mask_rate 0.5 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --gpt_layer 3 \
    --d_model 768 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 5 \
    --learning_rate 0.0001 \
    --LLM $LLM \
    --patch_size 16 \
    --stride 8 \
    --add_prompt 1 \
    --prompt_length 2 \
    --batch_size 1024 \
    --sim_coef -0.1 \
    --pool_size 1000 \
    --period 24 \
    --percent 100 \
    --trend_length 96 \
    --seasonal_length 48
done