set -x

pip install -r requirements.txt

IMAGENET_TRAIN="IMAGENET_DIR/train/shards-{00000..01281}.tar"
IMAGENET_VAL="IMAGENET_DIR/val/shards-{00000..00049}.tar"

# Batch Size = 8 GPUs * 512 = 4096
torchrun --standalone --nnodes=1 --nproc_per_node 8 \
       linear_prob.py \
       -a resnet50 --lr 0.1 \
       --dist-url "tcp://localhost:1234" \
       --num-classes 1000 \
       --train-data $IMAGENET_TRAIN \
       --val-data $IMAGENET_VAL \
       --exp-folder "EXP_FOLDER" \
       --exp-name "mocov2_in1k_lr0.03_bs256_100ep" \
       --pretrain "CHECKPOINT_DIR"
