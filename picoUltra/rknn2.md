# rknn-toolkit2 miniconda

* deps

```
sudo apt-get install python3 python3-dev python3-pip
sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc
```


### install conda

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh
chmod 777 Miniconda3-4.6.14-Linux-x86_64.sh
bash Miniconda3-4.6.14-Linux-x86_64.sh

!!! provide path on share folder outside the docker !!!

will add to ~/.bashrc

```
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/user/shared/XXXX/condaEnvs/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/user/shared/XXXX/condaEnvs/etc/profile.d/conda.sh" ]; then
        . "/home/user/shared/XXXX/condaEnvs/etc/profile.d/conda.sh"
    else
        export PATH="/home/user/shared/XXXX/condaEnvs/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

```

###  create conda env for toolkit2

```
conda create -n RKNN-Toolkit2 python=3.8

conda activate RKNN-Toolkit2

```


```
git clone https://github.com/rockchip-linux/rknn-toolkit2.git
cd rknn-toolkit2

## instal deps:
pip install tf-estimator-nightly==2.8.0.dev2021122109 
pip install -r rknn-toolkit2/packages/requirements_cp38-1.6.0.txt -i https://pypi.mirrors.ustc.edu.cn/simple/

## install
pip install rknn-toolkit2/packages/rknn_toolkit2-1.6.0+81f21f4d-cp38-cp38-linux_x86_64.whl

```

test

```
python
Python 3.8.13 (default, Mar 28 2022, 11:38:47) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from rknn.api import RKNN
>>> 


```

## usage

```
conda activate RKNN-Toolkit2


```