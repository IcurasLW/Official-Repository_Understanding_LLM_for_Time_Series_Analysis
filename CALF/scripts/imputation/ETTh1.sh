model_name=CALF
LLM=GPT2


python -u /home/nathan/LLM4TS/Imputation_task/CALF/run.py\
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
  --batch_size 16 \
  --d_model 768 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --LLM $LLM


python -u /home/nathan/LLM4TS/Imputation_task/CALF/run.py\
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
  --gpt_layers 3 \
  --batch_size 16 \
  --d_model 768 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --LLM $LLM


python -u /home/nathan/LLM4TS/Imputation_task/CALF/run.py\
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
  --gpt_layers 3 \
  --batch_size 16 \
  --d_model 768 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --LLM $LLM


python -u /home/nathan/LLM4TS/Imputation_task/CALF/run.py\
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
  --gpt_layers 3 \
  --batch_size 16 \
  --d_model 768 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --LLM $LLM