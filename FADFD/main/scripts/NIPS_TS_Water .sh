export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.5 --num_epochs 3   --batch_size 256  --dataset NIPS_TS_Water  --input_c 9  --win_size 92 --win_size_1 14 --count 8 --p 1

