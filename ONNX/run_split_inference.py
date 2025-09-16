import onnxruntime as ort
import numpy as np

img = np.random.randn(1,3,640,640).astype(np.float32)

backbone_sess = ort.InferenceSession("yolov8n_backbone.onnx")
head_sess = ort.InferenceSession("yolov8n_head.onnx")

# 1) backbone 실행
feat = backbone_sess.run(
    ["/model.4/cv2/conv/Conv_output_0"], {"images": img}
)[0]

# 중간 결과 출력
print("Backbone output shape:", feat.shape)
print("Backbone output sample values:")
print(feat[0, :4, :5, :5])  # 배치=0, 채널 앞 4개, 공간 앞 5x5만 출력

# 2) head 실행
outputs = head_sess.run(None, {"/model.4/cv2/conv/Conv_output_0": feat})

print("Final output shape:", outputs[0].shape)
