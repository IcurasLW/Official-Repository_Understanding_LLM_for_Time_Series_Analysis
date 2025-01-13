seq_len=480
model=GPT4TS

for percent in 100
do
for pred_len in 192 336 720
do
for LLM in 'GPT2' 'Random'
do
python /home/nathan/LLM4TS/Forecasting_task/OneFitsAll/Long-term_Forecasting/main.py \
    --root_path /home/nathan/LLM4TS/datasets/forecasting/traffic \
    --data_path traffic.csv \
    --model_id traffic_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 400 \
    --learning_rate 0.001 \
    --train_epochs 4 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --batch_size 500 \
    --LLM $LLM
done
done
done