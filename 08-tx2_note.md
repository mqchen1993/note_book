# Error: 
> InvalidArgumentError (see above for traceback): Number of ways to split should evenly divide the split dimension, but got split_dim 3 (size = 4) and num_split 3

# Solution:
1. rewrite *.png file use matlab
> rewrite_img.m
2. png file name unfit, need `pre_num.png`, say `1_1.png`, not `1 (1).png`
>  rename.py
3. update `testing.txt`, copy from `testing/image_2/testing.txt`
> prepare_testing_file.py

------
# runwaydetection 'incl'
```shell
$ mkdir incl
$ ln -s ../submodules/tensorflow-fcn/ tensorflow_fcn
$ ln -s ../submodules/evaluation/kitti_devkit/ seg_utils
$ ln -s ../submodules/evaluation/ evaluation
$ ln -s ../submodules/TensorVision/tensorvision tensorvision
```

------
## Nvidia TX2 install
# sudo pip install -r requirements.txt error
```shell
$ sudo apt-get install libjpeg-dev zlib1g-dev
$ apt-get install libatlas-base-dev 
```

------
# OpenCV compile with python
##1 cmake-qt-gui
* http://ports.ubuntu.com/ubuntu-ports/pool/universe/c/cmake/

##2 cmake ../
* exe: /usr/bin/python2.7
* include: /usr/include/python2.7
* lib: /usr/lib/aarch64-linux-gnu/libpython2.7.so
* num_include: /usr/local/lib/python2.7/dist-packages/numpy/core/include

##3 copy

------
# enable all of the CPU cores (6 core) on TX2
```python
$ sudo nvpmodel -m 0
```

------
# swap GB
[swap](https://www.cnblogs.com/EasonJim/p/7487596.html)
```shell
$ cd /
$ sudo fallocate -l 4G /swapfile 
$ sudo chmod 600 /swapfile
$ sudo mkswap /swapfile
$ sudo swapon /swapfile
$ free -m
$ sudo vim /etc/fstab
> /swapfile none swap sw 0 0
$ sudo swapoff /swapfile
```

------
# tx2 install tensorflow
* [pip install tf](https://devtalk.nvidia.com/default/topic/1031300/jetson-tx2/tensorflow-1-8-wheel-with-jetpack-3-2-/)
* jetpack 3.2 choose r1.9
* python 2.7
```shell
$ sudo apt install python-pip
$ sudo pip install tensorflow-1.9.0rc0-cp27-cp27mu-linux_aarch64.whl
```
