OUTPUT_DIR='output_dir'
DATA_PATH='synthetic_objects_path'
BACKGROUND="background_dataset_path"
DATA_PVOC='pvoc_path'
DATA_COCO='coco_path'
DATA_LVIS='lvis_path'

export NCCL_P2P_DISABLE=1

python run.py \
    --run_name 'test_OF_PIN' \
    --test_mode \
    --load_ckpt_path PINs/PIN_OF.pt \
    --vlm openflamingo \
    --train_batch 64 \
    --save_path ${OUTPUT_DIR} \
    --dataset_root ${DATA_PATH} \
    --background_url ${BACKGROUND} \
    --pvoc_dataset_root ${DATA_PVOC} \
    --coco_dataset_root ${DATA_COCO} \
    --lvis_root ${DATA_LVIS} \
    --MLP_hidden_dim 512 768 \
    --embed_channel "sinus" \
    --embed_channel_dim 64 \
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
    --freq_eval_objdet 1 \
    --prompt_algo 'PIN' \
