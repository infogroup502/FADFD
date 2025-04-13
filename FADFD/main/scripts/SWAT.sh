export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.8 --num_epochs 3   --batch_size 256  --dataset SWAT  --input_c 51  --win_size 5 --win_size_1 20 --count 6 --p 1

