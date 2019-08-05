# Train on CPU
python train.py --dataroot ./dataset/happy --name happy --checkpoints_dir ./checkpoints --gpu_ids -1

# Train on GPU
# python train.py --dataroot ./dataset/happy --name happy --checkpoints_dir ./checkpoints --gpu_ids 0

# Train on Cheeks&Eyes with no patch
# python train.py --dataroot ./dataset/eyeup --name eyeup --checkpoints_dir ./checkpoints --gpu_ids -1 --no_patch