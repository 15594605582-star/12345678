# -*- coding: utf-8 -*-
# convert_kmodel.py — GitHub Actions / Linux 用转换脚本
# ONNX → K230 kmodel (nncase 2.9.0 + nncase-kpu 2.9.0)
import nncase

ONNX = "best.onnx"
OUT = "color_yolov8n.kmodel"

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
print("onnx imported")
compiler.compile()
print("compiled")
data = compiler.gencode_tobytes()
with open(OUT, "wb") as f:
    f.write(data)
print("kmodel saved:", OUT, len(data), "bytes")
