CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd yolov7

torchrun --nproc_per_node=8 train_aux.py --workers 8 --batch-size 32 --data data.yaml --img-size 640 --epochs 20 --cfg cfg/training/yolov7-w6.yaml --weights yolov7-w6.pt --name yolov7x-w6- --hyp data/my_hyp.yaml --project /workspace/kaggle_benetech/logs
# torchrun --nproc_per_node=8 train.py --workers 8 --batch-size 32 --data data.yaml --img-size 640 --epochs 50 --cfg cfg/training/yolov7.yaml --weights yolov7.pt --name yolov7- --hyp data/hyp.scratch.custom.yaml --project /workspace/kaggle_benetech/logs