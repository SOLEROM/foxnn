# smoke test for uclibc

## build

* pay attention as may build for glibc or uclibc ; this will build uclibc

on the sdk we will find

```
RKNN_SDK=/home/user/shared/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api

|-- aarch64
|   `-- librknnrt.so
|-- armhf
|   `-- librknnrt.so
|-- armhf-uclibc
|   |-- librknnmrt.a        
|   `-- librknnmrt.so       ===> use this
`-- include
    |-- rknn_api.h          ===> use this
    |-- rknn_custom_op.h
    `-- rknn_matmul_api.h
```

###  for uclibc

```
CROSS_COMPILE=> /home/user/shared/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-

rknn_api.h      =>  $(RKNN_SDK)/include

rknnmrt         => $(RKNN_SDK)/armhf-uclibc
```


## librknnmrt

* on the RKNN_SDK
strings /home/user/shared/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/armhf-uclibc/librknnmrt.so  | grep -i version
librknnmrt version: 1.6.0 (9a7b5d24c@2023-12-13T17:33:10)


* from the zoo:
strings /home/user/shared/rknn_model_zoo/install/rv1106_linux_armv7l/rknn_yolov5_demo/lib/librknnmrt.so |grep -i version
librknnmrt version: 2.3.0 (c949ad889d@2024-11-07T11:37:43)


## test run

*  model build with 1.6
*  runtime librknnmrt 1.6

```
strings yolov5.rknn | grep version
target_platform":["rv1106"],"version":"1.6.0+81f21f4d"

strings lib/librknnmrt.so  | grep version
librknnmrt version: 1.6.0 (9a7b5d24c@2023-12-13T17:33:10)


[root@luckfox out]# ./npu_probe yolov5.rknn 
╔═══════════════════════════════════════════════
║ NPU probe test
╠═══════════════════════════════════════════════
║ Loading model: yolov5.rknn
╚→ rknn_init returned 0
```