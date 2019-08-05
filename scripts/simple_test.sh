# Place the pretrained model in ./checkpoints
python generate.py --dataroot ./dataset/test_star --name eyeup --gpu_ids -1 --eval --checkpoints_dir ./checkpoints
python generate.py --dataroot ./dataset/test_star --name anger --gpu_ids -1 --eval --checkpoints_dir ./checkpoints

# Make GIFS
python img2gif.py --exp_names happy,eyeup --dataroot ./dataset/test_star