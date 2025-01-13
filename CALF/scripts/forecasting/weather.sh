export CUDA_VISIBLE_DEVICES=0
gpu_fraction=0.15
seq_len=96
model=CALF


for pred_len in 96 192 336 720
do
for LLM in 'GPT2' 'Random' 'Linear' 'Att' 'Trans' 'NoLLM'
do
python /home/nathan/LLM4TS/Imputation_task/CALF/run.py \
    --root_path /home/nathan/LLM4TS/datasets/forecasting/weather/ \
    --data_path weather.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id weather_$model'_'$seq_len'_'$pred_len \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --train_epochs 5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --lradj type3 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5 \
    --task_loss smooth_l1 \ m
    --feature_loss smooth_l1 \
    --output_loss smooth_l1 \
    --LLM $LLM \
    --gpu_fraction $gpu_fraction


echo '====================================================================================================================='
done
done