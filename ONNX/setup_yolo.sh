#!/bin/bash

# 1. 환경 준비
echo "1. 시스템 업데이트 및 Miniconda 설치"
sudo apt-get update
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh -b -p $HOME/miniconda3
rm Miniconda3-latest-Linux-aarch64.sh

# conda 초기화
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# 2. 필수 라이브러리 설치 (버전 지정)
echo "2. conda 환경 생성 및 라이브러리 설치"
conda create -n yolo-env python=3.13.5 -y
conda activate yolo-env
conda install numpy=2.2.6 onnxruntime=1.20.1 requests=2.32.5 -y
pip install ultralytics==8.3.199
pip install torch==2.8.0+cpu torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu

# 3. PyTorch 모델을 ONNX로 변환
echo "3. YOLOv8n 모델을 ONNX로 변환"
# conver_onnx.py 스크립트 파일 생성
cat << EOF > conver_onnx.py
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx')
print("YOLOv8 모델이 onnx로 변환되었습니다.")
EOF

# 스크립트 실행
python conver_onnx.py

echo "모든 과정이 완료되었습니다. 'yolo-env' 환경이 생성되었고 'yolov8n.onnx' 파일이 준비되었습니다."