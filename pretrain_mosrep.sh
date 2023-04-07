set -x

pip install -r requirements.txt

IMAGENET_TRAIN="IMAGENET_DIR/train/shards-{00000..01281}.tar"
IMAGENET_VAL="IMAGENET_DIR/val/shards-{00000..00049}.tar"

# Batch Size = 8 GPUs * 64 = 256
torchrun --standalone --nnodes=1 --nproc_per_node 8 \
       pretrain_ddp.py \
       -a "resnet50" \
       -m "mosrep" \
       -b 64 \
       -j 8 \
       --lr 0.03 \
       --epochs 100 \
       --multi-crop \
       --global-scale 0.2 1.0 \
       --global-size 224 \
       --local-scale 0.1 0.6 \
       --local-size 112 \
       --shift-enable 1.0 \
       --shift-pix 48 \
       --shift-beta 0.5 \
       --moco-k 65536 \
       --seed 42 \
       --dist-url "tcp://localhost:1234" \
       --train-data $IMAGENET_TRAIN \
       --exp-folder "EXP_FOLDER" \
       --exp-name "mosrep_in1k_lr0.03_bs256_100ep"
