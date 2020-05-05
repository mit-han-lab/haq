export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore finetune.py     \
 -a resnet50                     \
 --resume checkpoints/resnet50/resnet50_0.1_75.48.pth.tar        \
 --workers 32                    \
 --test_batch 512                \
 --free_high_bit False           \
 --eval                          \
