python3 new_evaluate1.py /data/imagenet --model mobilenetv2_tiny_zero -b 1024 --sched cosine  --epochs 20 --lr-base 2.5e-5 --input-size 3 144 144 --weight-decay 4e-5 --momentum 0.9 --smoothing 0.1 -j 8 --seed 42 --opt sgd --output ./output/train/norm_pc_11000_1 --resume="./mobilenetv2_w35_r144.pth.tar" --experiment mobilenetv2_w35_ --warmup-epochs 0 --pin-mem --grad-accum-steps 1 --num-patches=3 --per-patch-stage=5

#torchrun --standalone --nnodes 1 --nproc_per_node 2 --master_port 12345 --node_rank 4 train.py /data/imagenet --model mobilenetv2_non_replicate -b 128 --sched step --decay-rate 0.98 --decay-epochs 1.0 --epochs 300 --lr-base 0.045 --input-size 3 224 224 --weight-decay 4e-5 --momentum 0.9 --smoothing 0.0 -j 8 --seed 42 --opt sgd --output ./output/train/mobilenetv2_non_reflect --experiment patchpadding --warmup-epochs 0 --local_rank 4 --log-wandb