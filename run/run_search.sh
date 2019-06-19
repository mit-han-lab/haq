python -W ignore rl_quantize.py     \
 --arch mobilenet_v3                \
 --dataset imagenet                 \
 --suffix ratio010                  \
 --preserve_ratio 0.1               \
 --n_worker 32                      \
 --data_bsize 256                   \
 --train_size 20000                 \
 --val_size 10000                   \
 --gpu_id 0,1,2,3                   \

