CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd yolov7

torchrun --nproc_per_node=8 train_aux.py --workers 8 --batch-size 8 --data data.yaml --img-size 512 --epochs 10 --cfg cfg/training/yolov7-w6.yaml --weights yolov7-w6.pt --name yolov7x-w6-v1. --hyp data/my_hyp.yaml --project /workspace/kaggle_benetech/logs

# torchrun --nproc_per_node=8 train_aux.py --workers 8 --batch-size 32 --data data_2.yaml --img-size 1024 --epochs 10 --cfg cfg/training/yolov7-w6.yaml --weights yolov7-w6.pt --name yolov7x-w6-v2. --hyp data/my_hyp.yaml --project /workspace/kaggle_benetech/logs


# torchrun --nproc_per_node=8 train_aux.py --workers 8 --batch-size 32 --data data_2.yaml --img-size 640 --epochs 10 --cfg cfg/training/yolov7-e6.yaml --weights yolov7-e6.pt --name yolov7x-e6-v2. --hyp data/my_hyp.yaml --project /workspace/kaggle_benetech/logs
