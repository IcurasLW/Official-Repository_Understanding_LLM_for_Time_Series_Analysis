
model=S2IPLLM

for LLM in 'Random'; do
    python /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
        --model_id UWaveGestureLibrary \
        --model $model \
        --root_path /home/nathan/LLM4TS/datasets/classification/UWaveGestureLibrary \
        --data_path UWaveGestureLibrary\
        --data UEA \
        --is_training 1 \
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
        --trend_length 8 \
        --period 8 \
        --seasonal_length 8 \
        --learning_rate 0.0001 \
        --train_epochs 30 \
        --LLM $LLM
done