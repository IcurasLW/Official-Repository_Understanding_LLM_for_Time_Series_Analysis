

# Make sure you replace --data_dir correctly to your path
for LLM in 'GPT2' 'Random' 'Linear' 'Att' 'Trans' 'NoLLM'
do
python main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name Heartbeat \
    --records_file Classification_records.xls \
    --data_dir datasets/UEA/Heartbeat \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --epochs 50 \
    --lr 0.0005 \
    --patch_size 32 \
    --stride 16 \
    --optimizer RAdam \
    --d_model 768 \
    --pos_encoding learnable \
    --task classification \
    --key_metric accuracy \
    --LLM $LLM
done