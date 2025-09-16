# ONNX 모델 분할 추론 가이드라인 (Raspberry Pi 기준)

이 가이드라인은 Raspberry Pi (aarch64, ARM64) 환경에서 YOLOv8n ONNX 모델을 분할하고, 분할된 모델을 사용하여 효율적으로 추론하는 방법을 소개

---

1. 환경 세팅 및 준비
`setup_yolo.sh` 스크립트는 Miniconda부터 필요한 모든 라이브러리를 특정 버전으로 설치 및 YOLOv8모델을 ONNX까지 변환하는 스크립트
```bash
chmod +x setup_yolo.sh
./setup_yolo.sh
```

2. ONNX 모델 구조 확인
모델 분할을 위해 중간 레이어의 출력 이름을 확인하는 단계
```bash
python print_onnx_structure.py
```

3. ONNX 모델 분할
확인한 출력 이름을 사용하여 모델을 `backbone`과 `head`로 분할
```bash
python split_onnx_model.py
```

4. 분할된 모텔 추론 실행
분할된 모델을 사용하여 순차적으로 추론을 실행하는 단계
```bash
python run_split_inference.py
```
