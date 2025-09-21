import onnx
from onnx.utils import extract_model

input_model = "yolov8n.onnx"
model = onnx.load(input_model)

print("=== ONNX Graph Structure ===")
layer_outputs = []
for i, node in enumerate(model.graph.node):
    print(f"{i}: {node.op_type}, inputs={node.input}, outputs={node.output}")
    if len(node.output) > 0:
        layer_outputs.append((i, node.output[0]))

print("\n총", len(layer_outputs), "개의 레이어 output 발견\n")

for idx, output_name in layer_outputs:
    try:
        backbone_file = f"yolov8n_backbone_{idx}.onnx"
        extract_model(
            input_model,
            backbone_file,
            input_names=["images"],
            output_names=[output_name],
        )

        head_file = f"yolov8n_head_{idx}.onnx"
        extract_model(
            input_model,
            head_file,
            input_names=[output_name],
            output_names=["output0"],
        )

        print(f"[SUCCESS] Layer {idx} ({output_name}) → backbone:{backbone_file}, head:{head_file}")

    except Exception as e:
        print(f"[FAILED ] Layer {idx} ({output_name}) → {e}")
