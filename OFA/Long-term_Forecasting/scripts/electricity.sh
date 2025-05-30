export CUDA_VISIBLE_DEVICES=0

seq_len=512
model=GPT4TS
batch_size=1
gpu_index=2
for pred_len in 96 192 336 720
do
for percent in 100
do

python /home/nathan/LLM4TS/Forecasting_task/OneFitsAll/Long-term_Forecasting/main.py \
    --root_path /home/nathan/LLM4TS/datasets/forecasting/electricity \
    --data_path electricity.csv \
    --model_id ECL_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size $batch_size \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --gpu_fraction 0.3 \
    --gpu_index $gpu_index \
    --LLM Random # change 

python /home/nathan/LLM4TS/Forecasting_task/OneFitsAll/Long-term_Forecasting/main.py \
    --root_path /home/nathan/LLM4TS/datasets/forecasting/electricity \
    --data_path electricity.csv \
    --model_id ECL_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size $batch_size \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --gpu_fraction 0.3 \
    --gpu_index $gpu_index \
    --LLM Linear
done
done
