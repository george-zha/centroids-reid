python inference/pytorch_infer_resnet50.py \
--config_file="configs/320_resnet50_ibn_a.yml" \
--gallery_data='/home/ubuntu/datasets/verkada_data/gallery' \
--query_data='/home/ubuntu/datasets/verkada_data/query' \
--normalize_features \
--topk=5 \
GPU_IDS [0] \
TEST.IMS_PER_BATCH 128 \
OUTPUT_DIR 'output-dir' \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "/home/ubuntu/centroids-reid/resnet50_neuron.pt" \
SOLVER.DISTANCE_FUNC 'cosine'