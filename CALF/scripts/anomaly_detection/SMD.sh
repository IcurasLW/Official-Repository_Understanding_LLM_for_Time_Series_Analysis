
model=CALF

for LLM in 'GPT2' 'Random' 'Linear' 'Att' 'Trans' 'NoLLM'
do
python -u /home/nathan/LLM4TS/Anomaly_Detection_task/CALF/run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path /home/nathan/LLM4TS/datasets/anomaly_detection/SMD \
  --model_id SMD \
  --model CALF \
  --data SMD \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 768 \
  --n_heads 4 \
  --d_ff 768 \
  --dropout 0.3 \
  --enc_in 38 \
  --c_out 38 \
  --gpt_layer 6 \
  --itr 1 \
  --r 8 \
  --lora_alpha 32 \
  --anomaly_ratio 0.5 \
  --lora_dropout 0.1 \
  --patience 3 \
  --learning_rate 0.0001 \
  --train_epochs 5 \
  --lradj type1 \
  --LLM $LLM
done