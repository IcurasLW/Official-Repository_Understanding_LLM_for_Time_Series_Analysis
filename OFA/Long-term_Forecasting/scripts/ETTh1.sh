
export CUDA_VISIBLE_DEVICES=0

seq_len=336
model=GPT4TS

for percent in 100
do
for pred_len in 96 192 336 720
do
for lr in 0.0001
do
for LLM in 'GPT2'  'Random' 'Att' 'Trans' 'Linear' 'NoLLM'
do
python /media/nathan/DATA/1Adelaide/Irregular_Time_Series/NeurIPS2023-One-Fits-All/Long-term_Forecasting/main.py \
    --root_path /media/nathan/DATA/1Adelaide/Irregular_Time_Series/datasets/Forcasting_dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_h \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 128 \
    --lradj type4 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.5 \
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
    --itr 1 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --is_gpt 1 \
    --LLM $LLM
done
done
done
done
