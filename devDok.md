# dev docker

## part1 - sdk

* https://wiki.luckfox.com/Luckfox-Pico/Luckfox-Pico-SDK/
* compile build root

```

sudo apt-get install -y git ssh make gcc gcc-multilib g++-multilib module-assistant expect g++ gawk texinfo libssl-dev bison flex fakeroot cmake unzip gperf autoconf device-tree-compiler libncurses5-dev pkg-config bc python-is-python3 passwd openssl openssh-server openssh-client vim file cpio rsync

cd luckfox-pico/
> ./build.sh lunch

[8] RV1106_Luckfox_Pico_Ultra_W
[0] EMMC
[0] Buildroot
[build.sh:info] Running build_select_board succeeded.

./build.sh

output/image/
|-- boot.img
|-- download.bin
|-- env.img
|-- idblock.img
|-- oem.img
|-- rootfs.img
|-- sd_update.txt
|-- tftp_update.txt
|-- uboot.img
|-- update.img
`-- userdata.img

```



## part2 - rknn-toolkit2

```
git clone https://github.com/rockchip-linux/rknn-toolkit2


sudo apt-get update
sudo apt-get install python3 python3-dev python3-pip
sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc


python3 --version
Python 3.10.12   => _cp310-1.6.0.txts_cp310-1.6.0


pip3 install -r  rknn-toolkit2/rknn-toolkit2/packages/requirements_cp310-1.6.0.txts_cp310-1.6.0

pip3 install rknn-toolkit2/packages/rknn_toolkit2-1.6.0+81f21f4d-cp310-cp310-linux_x86_64.whl

pip install "numpy<2"


```

```
>>> python3
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'python3' is not defined
>>> from rknn.api import RKNN
```