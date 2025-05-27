# model conversions

* ref : https://github.com/LuckfoxTECH/luckfox_pico_rknn_example/blob/kernel-5.10.160/scripts/luckfox_onnx_to_rknn/README.md

* model conversion based on [toolkit2](./picoUltra/rknn2.md) in dev docker conda env and a convert.py script from luckfox_pico_rknn_example repo


## generic ? convert

```
~/shared/faceIT/luckfox_pico_rknn_example/scripts/luckfox_onnx_to_rknn/convert.py 
=======================================================================
import sys

from rknn.api import RKNN

def parse_arg():
    if len(sys.argv) < 5:
        print("Usage: python3 {} [onnx_model_path] [dataset_path] [output_rknn_path] [model_type]".format(sys.argv[0]));
        exit(1)

    model_path = sys.argv[1]
    dataset_path= sys.argv[2]
    output_path = sys.argv[3]
    model_type = sys.argv[4]

    return model_path, dataset_path, output_path,model_type

if __name__ == '__main__':
    model_path, dataset_path, output_path, model_type= parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    if model_type == 'Retinaface':
        rknn.config(mean_values=[[104, 117, 123]], std_values=[[1, 1, 1]], target_platform='rv1103',
                                quantized_algorithm="normal", quant_img_RGB2BGR=True,optimization_level=0)
        print("Use retinaface mode")
    else:
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rv1103')
    
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=dataset_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
```


```

python convert.py <onnx_model_path> <dataset_path> <export_model_path> <model_type(Retinaface etc.)>


    onnx_model_path: Path to the exported ONNX model, provided in the luckfox_onnx_to_rknn/model directory in the example
    dataset_path: Provide a small number of images as references for model conversion by specifying their file paths in a .txt file as a parameter to the conversion script.
    export_model_path: Name and location of the exported RKNN model, ensure it has a .rknn extension
    model_type: Different RKNN preprocessing settings are provided based on the type of model being converted. In the example, input "Retinaface", "Facenet", or "Yolov5".

```

## Configuration

*  adjust preprocessing settings based on the preprocessing code in the model training source code.


```
For example, in Retinaface, the image preprocessing code subtracts the mean values of the three channels. Therefore, when calling the RKNN.Config interface, the mean_values parameter needs to be configured.

# Pre-process config
    print('--> Config model')
    if model_type == 'Retinaface':
        rknn.config(mean_values=[[104, 117, 123]], std_values=[[1, 1, 1]], target_platform='rv1103',
                                quantized_algorithm="normal", quant_img_RGB2BGR=True,optimization_level=0)
        print("Use retinaface mode")


```

* Input format (quant_img_RGB2BGR) ;  Set to True only if your model was trained with BGR images (like OpenCV default). Otherwise, leave it False.

* Input shape and layout / example ONNX model must have a static input shape (e.g., [1, 3, 224, 224])

* adjust quantized_algorithm ( normal" is fine for starters)
* adjust optimization_level (0,1,2)