
model=S2IP


for LLM in 'Random'; do
    python /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
        --model_id EthanolConcentration \
        --is_training 1 \
        --model S2IPLLM \
        --root_path /home/nathan/LLM4TS/datasets/classification/EthanolConcentration \
        --data_path EthanolConcentration\
        --data UEA \
        --d_model 768 \
        --task_name classification \
        --gpt_layers 6 \
        --d_model 768 \
        --patch_size 8 \
        --stride 8 \
        --itr 1 \
        --add_prompt 1 \
        --sim_coef -0.1 \
        --pool_size 1000 \
        --percent 100 \
        --loss CE \
        --enc_in 3 \
        --batch_size 16 \
        --trend_length 96 \
        --period 96 \
        --seasonal_length 96 \
        --learning_rate 0.0001 \
        --train_epochs 30 \
        --LLM $LLM
done