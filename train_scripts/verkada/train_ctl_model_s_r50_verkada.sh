python3 train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
DATASETS.NAMES 'verkada_data' \
GPU_IDS [0] \
DATASETS.ROOT_DIR '/home/georgez/datasets' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
DATALOADER.USE_RESAMPLING False \
USE_MIXED_PRECISION False