model_name=Linear
train_epochs=100
learning_rate=0.01
llama_layers=32

master_port=00097
num_process=8
batch_size=256
d_model=32
d_ff=128


comment='TimeLLM-ETTh1'


python /home/nathan/LLM4TS/Time-LLM/run_main_single_gpu.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/nathan/LLM4TS/Time-LLM/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \



# python /home/nathan/LLM4TS/Time-LLM/run_main_single_gpu.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path /home/nathan/LLM4TS/Time-LLM/dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_512_96 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 512 \
#   --label_len 48 \
#   --pred_len 96 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment