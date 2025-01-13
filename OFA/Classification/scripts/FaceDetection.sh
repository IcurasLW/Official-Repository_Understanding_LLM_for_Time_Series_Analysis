


# Make sure you replace --data_dir correctly to your path

for LLM in 'GPT2' 'Random' 'Linear' 'Att' 'Trans' 'NoLLM'
do
python main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name FaceDetection \
    --records_file Classification_records.xls \
    --data_dir datasets/UEA/FaceDetection \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --epochs 50 \
    --lr 0.0001 \
    --patch_size 4 \
    --stride 2 \
    --optimizer RAdam \
    --d_model 768 \
    --pos_encoding learnable \
    --task classification \
    --key_metric accuracy \
    --LLM $LLM
done
