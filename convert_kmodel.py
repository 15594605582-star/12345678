# -*- coding: utf-8 -*-
# convert_kmodel.py — GitHub Actions 批量转换(参数化)
# 用法: python convert_kmodel.py <input.onnx> <output.kmodel>
import sys
import nncase

ONNX = sys.argv[1] if len(sys.argv) > 1 else "best.onnx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "color_yolov8n.kmodel"

co = nncase.CompileOptions()
co.target = "k230"
co.preprocess = False
co.swapRB = False
co.input_layout = "NCHW"
co.output_layout = "NCHW"
co.input_shape = [1, 3, 320, 320]
co.input_type = "uint8"
co.input_range = [0, 255]
co.input_mean = [0, 0, 0]
co.quant_type = "uint8"
co.calibrate_method = "NoClip"
co.dump_ir = False
co.dump_asm = False
co.dump_dir = "tmp_dump"

compiler = nncase.Compiler(co)
compiler.import_onnx(ONNX, nncase.ImportOptions())
print(ONNX, "imported")
compiler.compile()
print(ONNX, "compiled")
data = compiler.gencode_tobytes()
with open(OUT, "wb") as f:
    f.write(data)
print(OUT, "saved", len(data), "bytes")
