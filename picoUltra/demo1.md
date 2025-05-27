# demo1

```
git clone https://github.com/airockchip/yolov5.git
cd yolov5

<!-- python3 -m venv venv -->
source venv/bin/activate


pip install -r requirements.txt

/// use older version due to compatibility issues
pip install torch==2.5.1 torchvision==0.20.1
```


```
python export.py --rknpu --weight yolov5s.pt  --opset 19   


produce:
    28M  yolov5s.onnx
    15M  yolov5s.pt

```


* yolov5s.pt weight file in the project directory will be automatically pulled and converted. 
* The yolov5s.onnx and RK_anchors.txt files will be generated in the project folder
* The post_process function in the application source code will use the parameters in RK_anchors.txt

* the generated yolov5s.onnx model removes the incompatible result extraction part of RKNPU and is implemented using CPU in the application source code.

```
!!!!!!!!!  deactivate => return to dev docker environment !!!!!!!!!
```


```
git clone https://github.com/airockchip/rknn_model_zoo.git
cd rknn_model_zoo
cp ../yolov5/yolov5s.onnx examples/yolov5/model

cd examples/yolov5/python/
python3 convert.py  ../model/yolov5s.onnx rv1106

```

```

python3 convert.py <onnx_model> <TARGET_PLATFORM> <dtype(optional)> <output_rknn_path(optional)>
    
    
    <onnx_model>: ONNX model path.
    <TARGET_PLATFORM>: Specify the NPU platform name. For example, "rv1106".
    <quant_dtype>: Optional. It can be specified as i8 or fp. i8 indicates quantization, fp indicates no quantization, default is i8.
    <output_rknn_path>: Optional. Used to specify the save path of the RKNN model. By default, it is saved in the same directory as the ONNX model, named 'yolov5.rknn'.


```


```
> python3 convert.py  ../model/yolov5s.onnx rv1106
W __init__: rknn-toolkit2 version: 1.6.0+81f21f4d
--> Config model
done
--> Loading model
Loading : 100%|████████████████████████████████████████████████| 135/135 [00:00<00:00, 29636.29it/s]
done
--> Building model

W build: found outlier value, this may affect quantization accuracy
const name          abs_mean    abs_std     outlier value
onnx::Conv_375      0.83        1.39        14.282      


GraphPreparing : 100%|█████████████████████████████████████████| 149/149 [00:00<00:00, 14677.11it/s]
Quantizating : 100%|██████████████████████████████████████████████| 149/149 [00:04<00:00, 36.13it/s]
W build: The default input dtype of 'images' is changed from 'float32' to 'int8' in rknn model for performance!
                       Please take care of this change when deploy rknn model with Runtime API!
W build: The default output dtype of 'output0' is changed from 'float32' to 'int8' in rknn model for performance!
                      Please take care of this change when deploy rknn model with Runtime API!
W build: The default output dtype of '371' is changed from 'float32' to 'int8' in rknn model for performance!
                      Please take care of this change when deploy rknn model with Runtime API!
W build: The default output dtype of '373' is changed from 'float32' to 'int8' in rknn model for performance!
                      Please take care of this change when deploy rknn model with Runtime API!
done
--> Export rknn model
done


```

### results:

```
15M  yolov5s.pt
28M  yolov5s.onnx
7.9M  yolov5.rknn
```

## cross

* compile the dev env sdk to use it cross compiler;

```
export GCC_COMPILER=/home/user/shared/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf

chmod +x ./build-linux.sh

./build-linux.sh -t rv1106 -a armv7l -d yolov5



install/
`-- rv1106_linux_armv7l
    `-- rknn_yolov5_demo
        |-- lib
        |   |-- librga.so
        |   `-- librknnmrt.so
        |-- model
        |   |-- bus.jpg
        |   |-- coco_80_labels_list.txt
        |   `-- yolov5.rknn
        `-- rknn_yolov5_demo



```


## test

```
scp -r rknn_yolov5_demo/ root@172.32.0.93:/root
sshpass -p luckfox ssh -x root@172.32.0.93
cd rknn_yolov5_demo
./rknn_yolov5_demo model/yolov5.rknn model/bus.jpg 
exit

scp root@172.32.0.93:/root/rknn_yolov5_demo/out.png .

```