export CUDA_VISIBLE_DEVICES=1

seq_len=104
model=CALF


for pred_len in 60
do
for percent in 100
do
for LLM in 'GPT2' 'Random'
do
python /home/nathan/LLM4TS/Imputation_task/CALF/run.py \
    --root_path /home/nathan/LLM4TS/datasets/forecasting/illness/ \
    --data_path national_illness.csv \
    --model_id illness_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --task_name long_term_forecast \
    --is_training 1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --percent $percent \
    --dropout 0.3 \
    --gpt_layer 6 \
    --itr 1 \
    --enc_in 7 \
    --c_out 7 \
    --model $model \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 10 \
    --r 8 \
    --LLM $LLM 
done
done
done