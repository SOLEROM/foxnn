# demo0 -from  the zoo

* test with ready models in zoo: https://github.com/airockchip/rknn_model_zoo

### yolov5 example

* get the data

```
git clone https://github.com/airockchip/rknn_model_zoo.git


cd <rknn_model_zoo Path>/rknn_model_zoo/examples/yolov5/model
chmod a+x download_model.sh
./download_model.sh


model
=======
|-- anchors_yolov5.txt
|-- bus.jpg
|-- coco_80_labels_list.txt
|-- download_model.sh
|-- yolov5.rknn
`-- yolov5s.onnx

```

* convert onnx to rknn model

```

conda activate RKNN-Toolkit2
cd python/

python3 convert.py  ../model/yolov5s.onnx rv1106 i8 ./yolov5s_i8.rknn
//  i8 indicates quantization
done
--> Export rknn model
done
         7.9M May 27 11:19 yolov5s_i8.rknn

```

* cross-compilation on the rknn_model_zoo/examples/yolov5


```
export GCC_COMPILER=/home/user/shared/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-
cd ../../../
chmod +x ./build-linux.sh
./build-linux.sh -t rv1106 -a armv7l -d yolov5

```

* results 

```
install/rv1106_linux_armv7l/rknn_yolov5_demo/
==============================================
|-- lib
|   |-- librga.so
|   `-- librknnmrt.so
|-- model
|   |-- bus.jpg
|   |-- cars.jpg
|   |-- coco_80_labels_list.txt
|   `-- yolov5.rknn
`-- rknn_yolov5_demo

```