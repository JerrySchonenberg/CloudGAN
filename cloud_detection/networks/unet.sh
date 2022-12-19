# Running 3 replications of U-Net on the 38-Cloud dataset
python3 main.py --summary --dataset_train "../../datasets/38-cloud/train/" --dataset_test "../../datasets/38-cloud/test/" --epochs 100 --batch_size 64 --workers 4 --augmentation --mp --model UNET --seed 8
python3 main.py --summary --dataset_train "../../datasets/38-cloud/train/" --dataset_test "../../datasets/38-cloud/test/" --epochs 100 --batch_size 64 --workers 4 --augmentation --mp --model UNET --seed 42
python3 main.py --summary --dataset_train "../../datasets/38-cloud/train/" --dataset_test "../../datasets/38-cloud/test/" --epochs 100 --batch_size 64 --workers 4 --augmentation --mp --model UNET --seed 1024
