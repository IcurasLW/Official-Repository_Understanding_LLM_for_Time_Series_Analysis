
model_name=TimeLLM
train_epochs=10
learning_rate=0.001
llama_layers=6

master_port=00097
num_process=8
batch_size=4
d_model=32
d_ff=128


comment='TimeLLM-ETTh2'


for LLM in 'LLaMa' 'Random' 'Linear' 'Att' 'Trans' 'NoLLM'
do
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 --mixed_precision=fp16 run_main_multiGPU_imputation.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_mask_0.125 \
    --mask_rate 0.125 \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --patch_len 16 \
    --stride 8 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --llm_layers $llama_layers \
    --train_epochs $train_epochs \
    --model_comment $comment \
    --llm_model $LLM \
    --llm_dim '4096'

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 --mixed_precision=fp16 run_main_multiGPU_imputation.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_mask_0.25 \
    --mask_rate 0.25 \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --patch_len 16 \
    --stride 8 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --llm_layers $llama_layers \
    --train_epochs $train_epochs \
    --model_comment $comment \
    --llm_model $LLM \
    --llm_dim '4096'


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 --mixed_precision=fp16 run_main_multiGPU_imputation.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_mask_0.375 \
    --mask_rate 0.375 \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --patch_len 16 \
    --stride 8 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --llm_layers $llama_layers \
    --train_epochs $train_epochs \
    --model_comment $comment \
    --llm_model $LLM \
    --llm_dim '4096'

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 --mixed_precision=fp16 run_main_multiGPU_imputation.py \
    --task_name imputation \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/Imputation/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_mask_0.5 \
    --mask_rate 0.5 \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --patch_len 16 \
    --stride 8 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --llm_layers $llama_layers \
    --train_epochs $train_epochs \
    --model_comment $comment \
    --llm_model $LLM \
    --llm_dim '4096'
done

