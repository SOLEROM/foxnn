#!/usr/bin/env python3
"""
convert_onnx_to_rknn.py – Convert an ONNX model to RKNN for RKNN-Toolkit2

Usage:
    python convert_onnx_to_rknn.py <model.onnx> <platform> [dtype] [output.rknn] [dataset.txt]

Arguments:
    model.onnx     Path to ONNX model file
    platform       Target chip (e.g. rk3588, rv1106, rk3568, ...)
    dtype          Quantization type: fp | i8 | u8   (default: fp)
    output.rknn    Output RKNN filename             (default: model.rknn)
    dataset.txt    Optional dataset file for quant  (required if dtype is i8/u8)
"""

import sys
import onnx
from rknn.api import RKNN

DEFAULT_OUT = 'model.rknn'
DEFAULT_DTYPE = 'fp'
VALID_PLATFORMS = {
    'rk3562', 'rk3566', 'rk3568', 'rk3576', 'rk3588',
    'rv1103', 'rv1106', 'rk1808', 'rv1109', 'rv1126'
}

def parse_cli():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    onnx_path = sys.argv[1]
    platform = sys.argv[2].lower()
    if platform not in VALID_PLATFORMS:
        sys.exit(f'ERROR: Unknown platform "{platform}"')

    dtype = sys.argv[3].lower() if len(sys.argv) > 3 else DEFAULT_DTYPE
    if dtype not in ('fp', 'i8', 'u8'):
        sys.exit(f'ERROR: dtype must be fp / i8 / u8, got "{dtype}"')

    out_rknn = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_OUT
    dataset_txt = sys.argv[5] if len(sys.argv) > 5 else None

    do_quant = dtype in ('i8', 'u8')
    if do_quant and not dataset_txt:
        sys.exit('ERROR: Quantization requires dataset.txt file')

    return onnx_path, platform, dtype, do_quant, out_rknn, dataset_txt

def detect_input_channels(onnx_path):
    model = onnx.load(onnx_path)
    input_tensor = model.graph.input[0]
    shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]

    # Infer layout and channels
    if len(shape) == 4:
        n, c, h, w = shape
        # NHWC if last dim is 1 or 3 or 4 and second is small
        if c in (1, 3, 4) and h > 10:
            return shape[3]  # NHWC → C = last
        else:
            return shape[1]  # NCHW → C = second
    elif len(shape) == 2:
        return shape[1]
    else:
        sys.exit(f'ERROR: Unable to infer channel count from shape {shape}')

def main():
    onnx_path, platform, dtype, do_quant, out_rknn, dataset_txt = parse_cli()

    num_channels = detect_input_channels(onnx_path)
    mean_values = [[0] * num_channels]
    std_values  = [[1] * num_channels]
    
    print(f"found num_channels to be {num_channels} ")

    rknn = RKNN(verbose=True)

    print(f'[1/4] config() → C={num_channels}')
    rknn.config(mean_values=mean_values,
                std_values=std_values,
                target_platform=platform)

    print('[2/4] load_onnx() →', onnx_path)
    if rknn.load_onnx(onnx_path) != 0:
        sys.exit('load_onnx failed')

    print('[3/4] build()', 'with quant' if do_quant else 'FP16')
    if rknn.build(do_quantization=do_quant,
                  dataset=dataset_txt if do_quant else None) != 0:
        sys.exit('build failed')

    print('[4/4] export_rknn() →', out_rknn)
    if rknn.export_rknn(out_rknn) != 0:
        sys.exit('export failed')

    print(f'✔ RKNN model saved to: {out_rknn}')
    rknn.release()

if __name__ == '__main__':
    main()
