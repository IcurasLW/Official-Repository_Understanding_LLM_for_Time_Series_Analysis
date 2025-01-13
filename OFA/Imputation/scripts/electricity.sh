model_name=GPT4TS



# Make sure you replace --root_path correctly to your path
for lr in 0.001
do
for LLM in 'GPT2' 'Random' 'Linear' 'Att' 'Trans' 'NoLLM'
do
python -u /home/nathan/LLM4TS/Imputation_task/OneFitsAll/run.py \
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
  --patch_size 1 \
  --stride 1 \
  --gpt_layer 3 \
  --d_model 768 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 3 \
  --train_epochs 30 \
  --learning_rate $lr \
  --LLM $LLM



python -u /home/nathan/LLM4TS/Imputation_task/OneFitsAll/run.py \
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
  --patch_size 1 \
  --stride 1 \
  --gpt_layer 3 \
  --d_model 768 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 3 \
  --train_epochs 30 \
  --learning_rate $lr \
  --LLM $LLM


python -u /home/nathan/LLM4TS/Imputation_task/OneFitsAll/run.py \
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
  --patch_size 1 \
  --stride 1 \
  --gpt_layer 3 \
  --d_model 768 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 3 \
  --train_epochs 30 \
  --learning_rate $lr \
  --LLM $LLM



python -u /home/nathan/LLM4TS/Imputation_task/OneFitsAll/run.py \
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
  --patch_size 1 \
  --stride 1 \
  --gpt_layer 3 \
  --d_model 768 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 3 \
  --train_epochs 30 \
  --learning_rate $lr \
  --LLM $LLM

done
done