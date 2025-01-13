model=S2IPLLM




for LLM in 'Random'; do
    python /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
        --model_id PEMS-SF \
        --model $model \
        --root_path /home/nathan/LLM4TS/datasets/classification/PEMS-SF \
        --data_path PEMS-SF\
        --data UEA \
        --is_training 1 \
        --d_model 768 \
        --task_name classification \
        --gpt_layers 6 \
        --d_model 768 \
        --patch_size 16 \
        --stride 8 \
        --itr 1 \
        --add_prompt 1 \
        --sim_coef -0.1 \
        --pool_size 500 \
        --percent 100 \
        --loss CE \
        --enc_in 963 \
        --batch_size 16 \
        --trend_length 15 \
        --period 15 \
        --batch_size 16 \
        --seasonal_length 15 \
        --learning_rate 0.0001 \
        --train_epochs 10 \
        --LLM $LLM
done