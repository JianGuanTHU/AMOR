export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
PORT=25641
MODEL_SIZE=7b
NUM_GPUS=8
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
NUM_TRAIN_EPOCHS=2

TRAIN_NAME=warmup_data
MODEL_PATH=./model/Llama-2-7b-chat-hf/
OUTPUT_PATH=./model/init_model
python3 ./init.py $MODEL_PATH $OUTPUT_PATH
cp $MODEL_PATH/tokenizer* $OUTPUT_PATH
MODEL_PATH=$OUTPUT_PATH
OUTPUT_DIR=./model/warmup_model
TRAIN_FILE=../data/warmup/${TRAIN_NAME}.json

echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ./ds_configs/stage3_no_offloading_accelerate.conf \
    --main_process_port ${PORT} \
    ./finetune.py \
    --model_name_or_path ${MODEL_PATH} \
    --tokenizer_name ${MODEL_PATH} \
    --use_slow_tokenizer \
    --train_file $TRAIN_FILE \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --output_dir  $OUTPUT_DIR \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_flash_attn