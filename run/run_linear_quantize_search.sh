export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore rl_quantize.py     \
 --arch qmobilenetv2                \
 --dataset imagenet100              \
 --dataset_root data/imagenet100    \
 --suffix ratio0556bit28            \
 --preserve_ratio 0.556             \
 --float_bit 8                      \
 --max_bit 8                        \
 --min_bit 2                        \
 --n_worker 32                      \
 --data_bsize 128                   \
 --train_size 20000                 \
 --val_size 10000                   \
 --linear_quantization              \
