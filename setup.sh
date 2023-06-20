echo "Install requirements"
echo

pip install -r requirements.txt


echo
echo "Setup YoloX"
echo

cd yolox
pip3 install -v -e .

pip install -U protobuf==3.20.0
pip install globox
cd ..

echo
echo "Setup CACHED"
echo

pip install -U openmim
mim install mmengine
mim install mmcv==1.7.1 mmdet==2.28.1
mim install input/mmdet-2.28.1-py3-none-any.whl --force-reinstall

