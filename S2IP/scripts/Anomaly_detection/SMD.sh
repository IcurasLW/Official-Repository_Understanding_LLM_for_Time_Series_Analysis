model=S2IPLLM
for LLM in 'GPT2' 'Random' 'Linear' 'Att' 'Trans' 'NoLLM'
do
CUDA_VISIBLE_DEVICES=2 python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path /home/nathan/LLM4TS/datasets/anomaly_detection/SMD \
  --model_id SMD \
  --model $model \
  --data SMD \
  --features M \
  --loss MSE \
  --des 'Exp' \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 768 \
  --d_ff 768 \
  --gpt_layer 6 \
  --enc_in 38 \
  --c_out 38 \
  --itr 1 \
  --number_variable 38 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --patch_size 16 \
  --stride 8 \
  --learning_rate 0.0001 \
  --train_epochs 5 \
  --LLM $LLM \
  --add_prompt 1 \
  --prompt_length 4 \
  --sim_coef -0.05 \
  --pool_size  1000 \
  --percent 100 \
  --trend_length 4 \
  --seasonal_length 5 \
  --period 100
done