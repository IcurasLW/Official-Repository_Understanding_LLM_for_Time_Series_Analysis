model_name=TimeLLM
train_epochs=20
learning_rate=0.01
llama_layers=6

master_port=00097
num_process=8
batch_size=8
d_model=16
d_ff=128

comment='TimeLLM-PEMS-SF'

# Define the list of LLM models you want to iterate over
for LLM in 'Att', 'Trans'; do
  # Run the Python script with the current LLM value
  python /home/nathan/LLM4TS/Classification_task/Time-LLM/run_main_single_gpu_classification.py \
    --task_name classification \
    --is_training 1 \
    --root_path /home/nathan/LLM4TS/datasets/classification \
    --data_path PEMS-SF \
    --model_id PEMS-SF \
    --model $model_name \
    --data PEMS-SF \
    --features M \
    --factor 3 \
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
    --llm_model $LLM
done
