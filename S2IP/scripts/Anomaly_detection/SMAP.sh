model=S2IPLLM
for LLM in 'GPT2' 'Random' 'Linear' 'Att' 'Trans' 'NoLLM'
do
  python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/anomaly_detection/SMAP \
    --model_id SMAP \
    --model $model \
    --data SMAP \
    --features M \
    --seq_len 100 \
    --pred_len 0 \
    --gpt_layer 6 \
    --d_model 768 \
    --d_ff 768 \
    --enc_in 25 \
    --c_out 25 \
    --anomaly_ratio 1 \
    --batch_size 32 \
    --patch_size 16 \
    --stride 8 \
    --learning_rate 0.0001 \
    --train_epochs 3 \
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