
# smokeTest for RV1106 (uclibc vs glibc)

* system build was done on rknn buildroot docker
* model and cross-compile was done on rknn docker

## system versions

system burn:

* buildroot (build with build.sh script)

```

```

* ubu: (build with build.sh script)

```
cd ~/proj/luckFox/flashTools/upgrade_tool_v2.17_for_linux
sudo ./upgrade_tool  uf ../../ultraBuildROOT/proj/luckfox-pico_OLDER_GIT_BUILD_UBU/output/image/update.img
```



## build

strings ReDimNet_no_mel.rknn   | grep RKNN  | grep target_platform

"target_platform":["rv1106"],"version":"2.3.2",


strings  yolov5s_i8.rknn   | grep RKNN  | grep target_platform

target_platform":["rv1106"],"version":"1.6.0+81f21f4d"


## runtime

/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api
|-- aarch64
|   -- librknnrt.so
|-- armhf
|   -- librknnrt.so
|-- armhf-uclibc
|   |-- librknnmrt.a
|   -- librknnmrt.so



librknnrt.so is for RK356X / RK3588 / RK1808 / RK3399Pro, etc.

librknnmrt.so is mandatory for RV1103 / RV1106 series
(the "m" means microcontroller-class / minimal NPU)



/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api
├── aarch64/         → RK3588 etc → uses librknnrt.so
├── armhf/           → RK3566 etc → uses librknnrt.so
└── armhf-uclibc/    → RV1106 etc → has librknnmrt.so


### librknnmrt?

/home/user/shared/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/armhf-uclibc/librknnmrt.so
librknnmrt version:1.6      // uclibc

 /home/user/shared/rknn_model_zoo/install/rv1106_linux_armv7l/rknn_yolov5_demo/lib/librknnmrt.so
librknnmrt version:2.3      // uclibc


/home/user/shared/faceIT/luckfox_pico_rknn_example/lib/glibc/librknnmrt.so
/home/user/shared/luckfox-pico/project/cfg/BoardConfig_IPC/overlay/overlay-luckfox-ubuntu-rockchip/usr/lib/librknnmrt.so
librknnmrt version: 1.6.0   // glibc

??? cant find 2.3 for glibc ???