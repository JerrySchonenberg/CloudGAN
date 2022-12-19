# Running 3 replications on the 38-Cloud dataset
python3 main.py --summary --dataset_train "../../datasets/38-cloud/train/" --dataset_test "../../datasets/38-cloud/test/" --epochs 64 --batch_size 64 --workers 4 --augmentation --mp --model AE --seed 8
python3 main.py --summary --dataset_train "../../datasets/38-cloud/train/" --dataset_test "../../datasets/38-cloud/test/" --epochs 64 --batch_size 64 --workers 4 --augmentation --mp --model AE --seed 42
python3 main.py --summary --dataset_train "../../datasets/38-cloud/train/" --dataset_test "../../datasets/38-cloud/test/" --epochs 64 --batch_size 64 --workers 4 --augmentation --mp --model AE --seed 1024
