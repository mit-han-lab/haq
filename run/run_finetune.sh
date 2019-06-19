python -W ignore finetune.py     \
 -a resnet50                     \
 -c checkpoints/resnet50         \
 --epochs 100                    \
 --lr 0.01                       \
 --train_batch 256               \
 --workers 32                    \
 --pretrained                    \
 --gpu_id 0,1,2,3                \

