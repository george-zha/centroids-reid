python inference/create_embeddings.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.ROOT_DIR '/home/ubuntu/datasets/inference/gallery/' \
TEST.IMS_PER_BATCH 128 \
OUTPUT_DIR './inference/output-dir' \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "/home/ubuntu/market1501.ckpt"