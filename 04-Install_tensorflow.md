# Install TensorFlow
---
## update 2018.10.01
## 0. Install Nvidia driver first.
[driver](https://www.geforce.cn/drivers)
* follow CUDA_Installation_Guide_Linux.pdf disable `Disabling Nouveau` & `x server`.
```shell
$ sudo ./NVIDIA-Linux-x86_64-390.87.run -no-x-check -no-nouveau-check -no-opengl-files
```
1. 报错the distribution-providedpre-install script failed!不必理会，继续安装。
2. 提示32位兼容问题，不用理会。
3. 安装时选择不启用Nvidia Xorg.conf服务

## 0. Ubuntu Reverting to a Previous Kernel 4.4.0 
* **Problem:**
> The driver installation is unable to locate the kernel source. when install cuda->nvidia drivers
> See follow CUDA_Installation_Guide_Linux.pdf `Table1.Native Linux Distribution Support in CUDA9.0 for details.
* **Solution:**
> Reverting from 4.15 to 4.4
```shell
$ uname -r
$ sudo apt-get install linux-headers-4.4.0-98-generic linux-image-4.4.0-98-generic
$ sudo reboot
```
1. when you see the Grub screen, Choose `Advanced options for Ubuntu`->`linux-headers-4.4.0-98-generic` 
2. hit `Enter`
```shell
$ uname -r
```
You can see `4.4.0-98-generic`,Now you can remove `linux-headers-4.15`.
```shell
$ sudo apt-get purge linux-image-4.15.0-64-generic
$ sudo apt-get purge linux-headers-4.15.0-64-generic
```
**Reboot**
And you will enter kernel `4.4.0-98-generic`, if not
```shell
$ sudo vim /etc/default/grub
$ GRUB_DEFAULT=”Ubuntu，Linux 4.4.0-98-generic“
$ sudo update-grub
```
Done!

## 1. install cuda9.0 on ubuntu 16.04
* follow CUDA_Installation_Guide_Linux.pdf
* **NO OpenGL, NO xorg.conf**
* add follow two lines to ~/.bashr
```shell 
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 
```

## 2. install cudnn
* [download link](https://developer.nvidia.com/rdp/cudnn-download)
* version: cuDNN v7.1.2 Library for Linux
```shell
$ sudo cp cudnn.h /usr/local/cuda/include/ #复制头文件
$ sudo cp lib* /usr/local/cuda/lib64/ #复制动态链接库
$ cd /usr/local/cuda/lib64/
$ sudo rm -rf libcudnn.so libcudnn.so.7  #删除原有动态文件
$ sudo ln -s libcudnn.so.7.2.1 libcudnn.so.7  #生成软衔接
$ sudo ln -s libcudnn.so.7 libcudnn.so  #生成软链接
$ sudo vim /etc/ld.so.conf.d/cuda.conf
>/usr/local/cuda/lib64
```
## 3. TensorFlow
* [TensorFlow_link](https://www.tensorflow.org/install/install_linux)
* Choose `Use pip in your system environment`
```shell
$ python -V  # or: python3 -V
$ pip -V     # or: pip3 -V
$ sudo apt-get install python-pip python-dev   # for Python 2.7
$ sudo apt-get install python3-pip python3-dev # for Python 3.n
$ pip install --upgrade pip
```
* **Problem**:
* pip error ocured!
> Import Error:cannot import name main
**Solution**
```shell
$ sudo vim /usr/bin/pip
```
Change from:
```python
from pip import main
if __name__ == '__main__':
    sys.exit(main())
```
To:
```python
from pip import __main__
if __name__ == '__main__':
    sys.exit(__main__._main())
```
* Install `tensorflow-gpu`
```shell
$ pip install --upgrade --user tensorflow-gpu   # Python 2.7
$ pip3 install --upgrade --user tensorflow-gpu  # Python 3.n
```
* Test install 
```shell
$ python -c "import tensorflow as tf; print(tf.__version__)"
```
And it will print `1.10.1`.
* **Done!**

