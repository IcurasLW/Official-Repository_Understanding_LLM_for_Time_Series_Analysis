

model=CALF


for LLM in 'GPT2' 'Random' 'Linear' 'Att' 'Trans' 'NoLLM'
do
python -u /home/nathan/LLM4TS/Anomaly_Detection_task/CALF/run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path /home/nathan/LLM4TS/datasets/anomaly_detection/PSM \
  --model_id PSM \
  --model CALF \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 768 \
  --n_heads 4 \
  --d_ff 768 \
  --dropout 0.3 \
  --enc_in 25 \
  --c_out 25 \
  --gpt_layer 6 \
  --itr 1 \
  --r 8 \
  --lora_alpha 32 \
  --anomaly_ratio 1 \
  --lora_dropout 0.1 \
  --patience 3 \
  --learning_rate 0.0001 \
  --train_epochs 5 \
  --lradj type1 \
  --LLM $LLM
done