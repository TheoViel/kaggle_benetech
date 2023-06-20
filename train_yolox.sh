CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd yolox

python tools/train.py -f exps/benetech_1_l_1.py -d 8 -b 32 --fp16 -o -c yolox_l.pth  --cache

echo

python tools/train.py -f exps/benetech_1_m_1.py -d 8 -b 32 --fp16 -o -c yolox_l.pth  --cache
