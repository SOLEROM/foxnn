#!/usr/bin/env python3
"""
convert_onnx_to_rknn.py
A minimal helper to turn an ONNX file into an RKNN file with RKNN-Toolkit2.

Usage
-----
conda activate RKNN-Toolkit2          # your existing env
python convert_onnx_to_rknn.py \
       simple_convnet.onnx            \  # input ONNX
       rk3588                         \  # target NPU platform
       fp                             \  # [i8|u8|fp]  (optional, default=fp)
       simple_convnet.rknn               # (optional) output path
"""
import sys
from rknn.api import RKNN            # part of RKNN-Toolkit2 :contentReference[oaicite:0]{index=0}

# ---- basic defaults you can tweak quickly ----
DATASET_TXT = 'dataset.txt'          # only needed when quantising
DEFAULT_OUT = 'model.rknn'
DEFAULT_DTYPE = 'fp'                 # fp = no quant; i8/u8 = quant
VALID_PLATFORMS = {
    'rk3562','rk3566','rk3568','rk3576','rk3588',
    'rv1103','rv1106','rk1808','rv1109','rv1126'
}

def parse_cli():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    onnx_path  = sys.argv[1]
    platform   = sys.argv[2].lower()
    if platform not in VALID_PLATFORMS:
        sys.exit(f'ERROR: unknown platform {platform}')

    dtype      = sys.argv[3].lower() if len(sys.argv) > 3 else DEFAULT_DTYPE
    if dtype not in ('i8','u8','fp'):
        sys.exit(f'ERROR: dtype must be i8/u8/fp, got {dtype}')

    out_path   = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_OUT
    do_quant   = dtype in ('i8','u8')          # only quantise for int
    return onnx_path, platform, do_quant, out_path

def main():
    onnx_path, platform, do_quant, out_path = parse_cli()

    rknn = RKNN(verbose=False)

    print('[1/4] config()')
    rknn.config(mean_values=[[0]],
                std_values=[[255]],
                target_platform=platform)



    print('[2/4] load_onnx()', onnx_path)
    if rknn.load_onnx(onnx_path) != 0:
        sys.exit('load_onnx failed')

    print('[3/4] build()', 'with quant' if do_quant else 'FP32')
    if rknn.build(do_quantization=do_quant,
                  dataset=DATASET_TXT if do_quant else None) != 0:
        sys.exit('build failed')

    print('[4/4] export_rknn()', out_path)
    if rknn.export_rknn(out_path) != 0:
        sys.exit('export failed')

    print('✔ Done – RKNN saved at:', out_path)
    rknn.release()

if __name__ == '__main__':
    main()
