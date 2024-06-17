OUTPUT_DIR='output_dir'
DATA_PATH='synthetic_objects_path'
BACKGROUND="background_dataset_path"

export NCCL_P2P_DISABLE=1

python run.py \
    --run_name 'OF_PIN_TRAIN' \
    --vlm openflamingo \
    --train_batch 64 \
    --lr 0.001 \
    --wd 0 \
    --epoch 160 \
    --train_size 60000 \
    --val_size 1000 \
    --test_size 1000 \
    --workers 12 \
    --rare_cases_cutoff 20 \
    --MLP_hidden_dim 512 768 \
    --embed_channel "sinus" \
    --embed_channel_dim 64 \
    --save_path ${OUTPUT_DIR} \
    --dataset_root ${DATA_PATH} \
    --background_url ${BACKGROUND} \
    --grid_size 16 \
    --max_new_tokens 12 \
    --middle_prompt 'located at' \
    --range_max_value 224 \
    --overlap_ratio 0.5 \
    --outside_img_ratio 0.0 \
    --num_objects_train 3 \
    --num_objects 3 \
    --overlap_constraint_for_both \
    --max_count_sampling 10 \
    --freq_eval_objdet 10\
    --save_freq 5 \
    --prompt_algo 'PIN' \
    --reconstruct_obj_name \
    --num_objects_fixed \
    --epoch_change_reconstruct 40 \
