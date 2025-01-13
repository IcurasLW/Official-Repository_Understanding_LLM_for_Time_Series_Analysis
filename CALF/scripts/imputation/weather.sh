model_name=CALF
LLM=GPT2



python -u /home/nathan/LLM4TS/Imputation_task/CALF/run.py\
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
  --gpt_layer 6 \
  --d_model 768 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 30 \
  --learning_rate 0.0005 \
  --LLM $LLM



python -u /home/nathan/LLM4TS/Imputation_task/CALF/run.py\
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
  --gpt_layer 6 \
  --d_model 768 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 30 \
  --learning_rate 0.0005 \
  --LLM $LLM



python -u /home/nathan/LLM4TS/Imputation_task/CALF/run.py\
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
  --gpt_layer 6 \
  --d_model 768 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 30 \
  --learning_rate 0.0005 \
  --LLM $LLM



python -u /home/nathan/LLM4TS/Imputation_task/CALF/run.py\
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
  --gpt_layer 6 \
  --d_model 768 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 30 \
  --learning_rate 0.0005 \
  --LLM $LLM