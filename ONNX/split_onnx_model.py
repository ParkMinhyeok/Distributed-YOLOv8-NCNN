from onnx import load
from onnx.utils import extract_model

input_model = "yolov8n.onnx"

# backbone 추출
extract_model(
    input_model,
    "yolov8n_backbone.onnx",
    input_names=["images"],                           # 모델 입력
    output_names=["/model.4/cv2/conv/Conv_output_0"], # backbone 출력 (Netron에서 확인한 출력 텐서 이름)
)

# head 추출
extract_model(
    input_model,
    "yolov8n_head.onnx",
    input_names=["/model.4/cv2/conv/Conv_output_0"],  # backbone 출력이 head 입력
    output_names=["output0"],                         # 최종 detection layer
)
