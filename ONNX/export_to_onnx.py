from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='onnx')
print("YOLOv8 모델이 onnx로 변환되었습니다.")
