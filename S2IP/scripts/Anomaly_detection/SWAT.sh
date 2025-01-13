model=S2IPLLM
for LLM in 'Att' 'Trans'
do
python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path /home/nathan/LLM4TS/datasets/anomaly_detection/SWaT \
  --model_id SWAT \
  --model $model \
  --data SWAT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --gpt_layer 6 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 16 \
  --stride 8 \
  --enc_in 51 \
  --c_out 51 \
  --itr 1 \
  --anomaly_ratio 1 \
  --number_variable 51 \
  --batch_size 512 \
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