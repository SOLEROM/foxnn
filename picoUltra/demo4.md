# facenet full app PREBUILD

* (1) face detection model : retinaface-pytorch build retinaface.onnx
(conda retinaface)
* (2) Face Feature Extraction model : facenet
    * extract 128-dimensional features based on the input face image
    * calculate the Euclidean distance with the features 

## ref
* prebuild repo https://github.com/LuckfoxTECH/luckfox_pico_rknn_example/blob/kernel-5.10.160/scripts/luckfox_onnx_to_rknn/README.md
* and [use the rknn2 toolkit](./rknn2.md)


## prebuilds

```
git clone https://github.com/LuckfoxTECH/luckfox_pico_rknn_example


luckfox_pico_rknn_example/scripts/luckfox_onnx_to_rknn/model/
==============================================================
|-- facenet.onnx
|-- facenet.rknn
|-- retinaface.onnx
|-- retinaface.rknn
|-- yolov5.onnx
`-- yolov5.rknn


```




## run embedded app

```

export LUCKFOX_SDK_PATH=~/shared/luckfox-pico/
export GLIBC_COMPILER=/home/user/shared/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-


cd /home/user/shared/faceIT/luckfox_pico_rknn_example

./build.sh

1) luckfox_pico_retinaface_facenet
2) luckfox_pico_retinaface_facenet_spidev
3) luckfox_pico_yolov5
Enter your choice [1-3]: 1

results with:

install/glibc/luckfox_pico_retinaface_facenet_demo/
|-- lib
|   `-- librga.so
|-- luckfox_pico_retinaface_facenet
`-- model
    |-- RetinaFace.rknn
    |-- mobilefacenet.rknn
    `-- test.jpg


```

move to embedded side (chmod +x) and run

```
./luckfox_pico_retinaface_facenet ./model/RetinaFace.rknn ./model/mobilefacenet.rknn ./model/test.jpg
```
