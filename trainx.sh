CUDA_VISIBLE_DEVICES=0 python train.py --fold=0 --frac=0.15 > LOGS/f15_fold0.txt
CUDA_VISIBLE_DEVICES=0 python train.py --fold=0 --frac=0.25 > LOGS/f25_fold0.txt
CUDA_VISIBLE_DEVICES=0 python train.py --fold=0 --frac=0.35 > LOGS/f35_fold0.txt
CUDA_VISIBLE_DEVICES=0 python train.py --fold=0 --frac=0.45 > LOGS/f45_fold0.txt

