export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.8 --num_epochs 3   --batch_size 128  --dataset SKAB  --input_c 8  --win_size 50 --win_size_1 10 --count 21 --p 0.2 --select 0

