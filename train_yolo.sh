CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd yolov7

# torchrun --nproc_per_node=8 train_aux.py --workers 8 --batch-size 32 --data data_8.yaml --img-size 1024 1024 --epochs 5 --cfg cfg/training/yolov7-w6.yaml --weights yolov7-w6.pt --name yolov7x-w6-v8.2-repro. --hyp data/my_hyp.yaml --project /workspace/kaggle_benetech/logs

# torchrun --nproc_per_node=8 train_aux.py --workers 8 --batch-size 32 --data data_11_sim.yaml --img-size 1024 1024 --epochs 5 --cfg cfg/training/yolov7-w6.yaml --weights yolov7-w6.pt --name yolov7x-w6-v11_sim. --hyp data/my_hyp.yaml --project /workspace/kaggle_benetech/logs


torchrun --nproc_per_node=8 train_aux.py --workers 8 --batch-size 32 --data data_12_sim.yaml --img-size 2048 2048 --epochs 10 --cfg cfg/training/yolov7-w6.yaml --weights yolov7-w6.pt --name yolov7x-w6-v12_sim. --hyp data/my_hyp.yaml --project /workspace/kaggle_benetech/logs