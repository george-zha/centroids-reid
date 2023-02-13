python3 train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'verkada_data' \
DATASETS.ROOT_DIR '/home/george/datasets' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/market1501/verkada_test' \
SOLVER.EVAL_PERIOD 40 \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH './experiments/public-epoch119.ckpt' \
OUTPUT_DIR './logs/verkada/'
