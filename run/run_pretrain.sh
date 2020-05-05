export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore pretrain.py                    \
 -a mobilenetv2                                 \
 -c checkpoints/mobilenetv2_lr005b256e150_fp16  \
 --epochs 150                                   \
 --lr 0.05                                      \
 --wd 0.00004                                   \
 --train_batch 256                              \
 --workers 32                                   \
 --half                                         \
