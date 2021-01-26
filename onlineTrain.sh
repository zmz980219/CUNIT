#CUDA_VISIBLE_DEVICES=0 python train.py --dataroot /home/yt/projects/CUNIT_2.0/datasets/chictopia_atr_test/ --name train_0124 --model_save_freq 1 --img_save_freq 1
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot /home/yt/projects/CUNIT_2.0/datasets/chictopia_atr_test/ --name train_0124 --resume '/home/yt/projects/CUNIT/results/train_0124/00000.pth'  --model_save_freq 1 --img_save_freq 1
