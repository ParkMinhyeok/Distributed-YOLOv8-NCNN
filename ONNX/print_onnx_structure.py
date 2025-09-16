import onnx

model = onnx.load("yolov8n.onnx")
for i, node in enumerate(model.graph.node):
    print(i, node.op_type, node.input, "->", node.output)
