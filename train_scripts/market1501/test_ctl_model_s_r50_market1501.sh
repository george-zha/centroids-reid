python3 train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR '/home/georgez/datasets' \
SOLVER.IMS_PER_BATCH 16 \
SOLVER.BASE_LR 0.00035 \
SOLVER.EVAL_PERIOD 40 \
TEST.ONLY_TEST True \
TEST.IMS_PER_BATCH 128 \
TEST.FEAT_NORM True \
TEST.THRESHOLD 0.8 \
TEST.VISUALIZE "no" \
MODEL.PRETRAIN_PATH "logs/market1501/market1501.ckpt" \
OUTPUT_DIR './logs/market1501/' \