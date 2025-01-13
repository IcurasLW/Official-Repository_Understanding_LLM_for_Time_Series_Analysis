
# Make sure you replace --data_dir correctly to your path


for LLM in 'GPT2' 'Random' 'Linear' 'Att' 'Trans' 'NoLLM'
do
python main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name EthanolConcentration \
    --records_file Classification_records.xls \
    --data_dir /home/nathan/LLM4TS/datasets/classification/EthanolConcentration \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --epochs 50 \
    --lr 0.001 \
    --patch_size 8 \
    --stride 8 \
    --optimizer RAdam \
    --d_model 768 \
    --pos_encoding learnable \
    --task classification \
    --key_metric accuracy \
    --LLM NoLLM
done