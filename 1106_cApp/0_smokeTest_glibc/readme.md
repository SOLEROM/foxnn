# smoke test for glibc

## build

* pay attention as may build for glibc or uclibc ; this will build glibc

## cross-compile

```
sudo apt update
sudo apt install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf

which arm-linux-gnueabihf-gcc
/usr/bin/arm-linux-gnueabihf-gcc

```


### for glibc 

```
CROSS_COMPILE   =>  use cross from repo

rknn_api.h      =>  $(RKNN_SDK)/include

rknnmrt         => <find suitable version>

```

## librknnmrt:

/home/user/shared/luckfox-pico/project/cfg/BoardConfig_IPC/overlay/overlay-luckfox-ubuntu-rockchip/usr/lib/librknnmrt.so
librknnmrt version: 1.6.0   // glibc

??? cant find ` 2.3` for glibc ???

## test run

* build gcc with glibc
* build rknn with 1.6

```
strings  yolov5s_i8.rknn   | grep RKNN  | grep target_platform

target_platform":["rv1106"],"version":"1.6.0+81f21f4d"
```

* use librknnmrt.so of 1.6.0 build for glib

run ok on ubuRock1106: 

```
pico@luckfox:~/out$ sudo ./npu_probe  yolov5s_i8.rknn 
[sudo] password for pico: 
╔═══════════════════════════════════════════════
║ NPU probe test
╠═══════════════════════════════════════════════
║ Loading model: yolov5s_i8.rknn
╚→ rknn_init returned 0

```
