

model=CALF
LLM=NoLLM

python /home/nathan/LLM4TS/Classification_task/CALF/run.py \
    --model_id PEMS-SF \
    --model CALF \
    --root_path /home/nathan/LLM4TS/datasets/classification/PEMS-SF \
    --data_path PEMS-SF \
    --data UEA \
    --d_model 768 \
    --pred_len 0 \
    --task_name classification \
    --gpt_layer 6 \
    --dropout 0.3 \
    --n_heads 4 \
    --d_ff 128 \
    --itr 1 \
    --r 8 \
    --task_loss ce \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 3 \
    --learning_rate 0.0005 \
    --train_epochs 30 \
    --lradj type1 \
    --batch_size 8 \
    --LLM $LLM