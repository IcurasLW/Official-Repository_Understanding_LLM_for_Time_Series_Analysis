model_name=S2IPLLM
# Learning rate=0.00001 for Att and Trans otherwise 0.001



for LLM in 'NoLLM'; do
  python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_mask_0.125 \
    --mask_rate 0.125 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --gpt_layers 3 \
    --d_model 768 \
    --patch_size 1 \
    --stride 1 \
    --des 'Exp' \
    --train_epochs 10 \
    --itr 1 \
    --mlp 1 \
    --learning_rate 0.001 \
    --LLM $LLM \
    --patch_size 16 \
    --stride 8 \
    --add_prompt 1 \
    --prompt_length 4 \
    --batch_size 128 \
    --sim_coef -0.05 \
    --pool_size  1000 \
    --percent 100 \
    --trend_length 96 \
    --seasonal_length 96



  python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_mask_0.25 \
    --mask_rate 0.25 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --gpt_layer 3 \
    --d_model 768 \
    --patch_size 1 \
    --stride 1 \
    --des 'Exp' \
    --train_epochs 10 \
    --itr 1 \
    --mlp 1 \
    --learning_rate 0.001 \
    --LLM $LLM \
    --patch_size 16 \
    --stride 8 \
    --add_prompt 1 \
    --prompt_length 4 \
    --batch_size 128 \
    --sim_coef -0.05 \
    --pool_size  1000 \
    --percent 100 \
    --trend_length 96 \
    --seasonal_length 96
    


  python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_mask_0.375 \
    --mask_rate 0.375 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --gpt_layer 3 \
    --d_model 768 \
    --patch_size 16 \
    --stride 8 \
    --des 'Exp' \
    --train_epochs 10 \
    --itr 1 \
    --mlp 1 \
    --learning_rate 0.001 \
    --LLM $LLM \
    --patch_size 16 \
    --stride 8 \
    --add_prompt 1 \
    --prompt_length 4 \
    --batch_size 128 \
    --sim_coef -0.05 \
    --pool_size  1000 \
    --percent 100 \
    --trend_length 96 \
    --seasonal_length 96
    


  python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_mask_0.5 \
    --mask_rate 0.5 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --gpt_layer 3 \
    --d_model 768 \
    --patch_size 1 \
    --stride 1 \
    --des 'Exp' \
    --train_epochs 10 \
    --itr 1 \
    --mlp 1 \
    --learning_rate 0.001 \
    --LLM $LLM \
    --patch_size 16 \
    --stride 8 \
    --add_prompt 1 \
    --prompt_length 4 \
    --batch_size 128 \
    --sim_coef -0.05 \
    --pool_size  1000 \
    --percent 100 \
    --trend_length 96 \
    --seasonal_length 96 
done