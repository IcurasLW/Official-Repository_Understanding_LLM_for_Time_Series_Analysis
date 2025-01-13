model_name=S2IPLLM


for LLM in 'GPT2' 'Random' 'Linear' 'Att' 'Trans' 'NoLLM'; do
    python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/electricity \
    --data_path electricity.csv \
    --model_id ECL_mask_0.125 \
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
    --train_epochs 3 \
    --learning_rate 0.001 \
    --LLM $LLM \
    --patch_size 16 \
    --stride 8 \
    --add_prompt 1 \
    --prompt_length 4 \
    --batch_size 1024 \
    --sim_coef -0.1 \
    --pool_size 1000 \
    --period 24 \
    --percent 100 \
    --trend_length 24 \
    --seasonal_length 4


    python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/electricity \
    --data_path electricity.csv \
    --model_id ECL_mask_0.25 \
    --mask_rate 0.25 \
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
    --train_epochs 3 \
    --learning_rate 0.001 \
    --LLM $LLM \
    --patch_size 16 \
    --stride 8 \
    --add_prompt 1 \
    --prompt_length 4 \
    --batch_size 1024 \
    --sim_coef -0.1 \
    --pool_size 1000 \
    --period 24 \
    --percent 100 \
    --trend_length 24 \
    --seasonal_length 4


    python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/electricity \
    --data_path electricity.csv \
    --model_id ECL_mask_0.375 \
    --mask_rate 0.375 \
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
    --train_epochs 3 \
    --learning_rate 0.001 \
    --LLM $LLM \
    --patch_size 16 \
    --stride 8 \
    --add_prompt 1 \
    --prompt_length 4 \
    --batch_size 1024 \
    --sim_coef -0.1 \
    --pool_size 1000 \
    --period 24 \
    --percent 100 \
    --trend_length 24 \
    --seasonal_length 4


    python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/electricity \
    --data_path electricity.csv \
    --model_id ECL_mask_0.5 \
    --mask_rate 0.5 \
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
    --train_epochs 3 \
    --learning_rate 0.001 \
    --LLM $LLM \
    --patch_size 16 \
    --stride 8 \
    --add_prompt 1 \
    --prompt_length 4 \
    --batch_size 1024 \
    --sim_coef -0.1 \
    --pool_size 1000 \
    --period 24 \
    --percent 100 \
    --trend_length 24 \
    --seasonal_length 4
done