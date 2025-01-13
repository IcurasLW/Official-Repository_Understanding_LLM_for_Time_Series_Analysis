seq_len=104
gpu_index=0
for LLM in 'GPT2' 'NoLLM'
do
# for pred_len in 24 36 48 60
for pred_len in 24 36 48
do
for percent in 100
do
CUDA_VISIBLE_DEVICES=$gpu_index python -u /home/nathan/LLM4TS/Anomaly_Detection_task/S2IP/run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/forecasting/illness/ \
    --data_path national_illness.csv \
    --model_id illness_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --model S2IPLLM \
    --data custom \
    --number_variable 1 \
    --features M \
    --seq_len 104 \
    --label_len 0 \
    --pred_len $pred_len \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1 \
    --d_model 768 \
    --learning_rate 0.0001 \
    --patch_size 24 \
    --stride 2 \
    --add_prompt 1 \
    --prompt_length 4 \
    --batch_size 128 \
    --sim_coef -0.05 \
    --pool_size  1000 \
    --percent 100 \
    --trend_length 24 \
    --seasonal_length 24 \
    --LLM $LLM \
    --gpu_fraction 1
done
done
done