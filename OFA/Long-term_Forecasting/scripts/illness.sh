export CUDA_VISIBLE_DEVICES=0

seq_len=104
model=GPT4TS


for pred_len in 24 36 48 60
do
for percent in 100
do
for LLM in 'GPT2'  'Random' 'Att' 'Trans' 'Linear' 'NoLLM'
do
python /media/nathan/DATA/1Adelaide/Irregular_Time_Series/NeurIPS2023-One-Fits-All/Long-term_Forecasting/main.py \
    --root_path /media/nathan/DATA/1Adelaide/Irregular_Time_Series/datasets/Forcasting_dataset/illness/ \
    --data_path national_illness.csv \
    --model_id illness_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 50 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 24 \
    --stride 2 \
    --percent $percent \
    --gpt_layer 3 \
    --itr 1 \
    --model $model \
    --is_gpt 1 \
    --LLM $LLM
done
done
done