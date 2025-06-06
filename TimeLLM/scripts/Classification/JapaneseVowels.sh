model_name=TimeLLM
train_epochs=30
learning_rate=0.01
llama_layers=32

master_port=00097
num_process=8
batch_size=8
d_model=16
d_ff=128



comment='TimeLLM-JapaneseVowels'

for LLM in 'LLaMa' 'Random' 'Linear' 'Att', 'Trans' 'NoLLM'; do
python run_main_multiGPU_classification.py \
  --task_name classification \
  --is_training 1 \
  --root_path /home/nathan/LLM4TS/datasets/classification \
  --data_path JapaneseVowels \
  --model_id JapaneseVowels \
  --model $model_name \
  --data JapaneseVowels \
  --features M \
  --factor 3 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 4 \
  --stride 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $LLM 
done

