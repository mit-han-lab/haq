export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore finetune.py     \
 -a qmobilenetv2                 \
 -c checkpoints/imagenet_qmobilenetv2_lr010e30_ratio060      \
 --data_name imagenet            \
 --data data/imagenet/           \
 --epochs 30                     \
 --lr 0.01                       \
 --train_batch 256               \
 --wd 4e-5                       \
 --workers 32                    \
 --pretrained                    \
 --linear_quantization           \
# --eval                         \


