export CUDA_VISIBLE_DEVICES=7
python main.py --dist False \
    --val_dir ../dataset/COCO/val \
    --train_dir ../dataset/COCO/train \
    --output_dir "output/test" \
    --eval_freq 5 \
    --num_bits 48 --img_size 256 --batch_size 16 --epochs 400 \
    --scheduler CosineLRScheduler,lr_min=5e-7,t_initial=400,warmup_lr_init=1e-6,warmup_t=5 \
    --optimizer Lamb,lr=2e-2 \
    --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
    --scaling_w 0.3 --scale_channels False --attenuation none \
    --loss_w_type bce --loss_margin 1 \
    --tile both --tile_size 64 --tile_type random_grid