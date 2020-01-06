import onnx

onnx_model_name = "onnx/alexnet.onnx"

print(f"Checking to see if ONNX Model is Valid...\n")
onnx_model = onnx.load(onnx_model_name)
print("here")
onnx.checker.check_model(onnx_model)
print(f"Model successfully loaded!  Benchmark Complete\n")