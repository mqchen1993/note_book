# 安装Anaconda环境

-------------------
## useful links.
* [blog](https://blog.csdn.net/qq_17534301/article/details/80869998)
* [Installing on Linux](http://docs.anaconda.com/anaconda/install/linux/)
* [Anaconda installer archive](https://repo.anaconda.com/archive/)
* [清华源](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

-------------------
## 1. 下载[Anaconda3](https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh)
* Install
```shell
$ sudo chmod +x Anaconda3-5.3.0-Linux-x86_64.sh
$ ./Anaconda3-5.3.0-Linux-x86_64.sh
# Anaconda3 will now be installed into this location: `/home/jun/anaconda3`.
# Python 3.7.0.
# No Microsoft VSCode? `no`
```

* Check anaconda installation.
```shell
$ conda -V
# conda 4.5.11
```

* 更换清华源
```shell
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
$ conda config --set show_channel_urls yes
```

---------------------
## 2. 创建虚拟python环境
### 2.1 基于 python2.7 创建一个名为py2 的环境:
```shell
$ conda create --name py27 python=2.7
# The following NEW packages will be INSTALLED:
# Proceed ([y]/n)? y
---
# To activate this environment, use
$ conda activate py27	# for `added by Anaconda3 5.3.0 installer` in '~/.bashrc'
$ source activate py27	# for `export PATH="/home/jun/anaconda3/bin:$PATH"` in '~/.bashrc'
# To deactivate an active environment, use
$ conda deactivate
$ source deactivate
```

### 2.2 基于 python3.6 创建一个名为py3 的环境：
```shell
$ conda create --name py3 python=3.6
```

# 2.3 删除虚拟环境
```shell
$ conda remove -n py27 --all
```

# 2.4 查看当前设置了哪些虚拟环境
```shell
$ conda env list
```

## 切换系统python2.7和anconda3环境
```shell
# added by Anaconda3 4.2.0 installer
$ export PATH="/home/king/anaconda3/bin:$PATH"
# python2.7
$ export PATH="/usr/bin/python2.7:$PATH"
```

## 在ubuntu上卸载anaconda
```shell
$ rm -rf ~/anaconda3/
# 在`~/.bashrc`文件中注释掉之前添加的anaconda路径.
$ source ~/.bashrc
```

## 常用命令
```shell
# 查看安装了哪些包
$ conda list
# 安装包
$ conda install package_name
# 查看当前存在哪些虚拟环境 
$ conda env list
$ conda info -e
# 检查更新当前conda
$ conda update conda
# 虚拟环境中安装额外的包
$ conda install -n py27 package_name
# 删除环境中的某个包
$ conda remove --name py27 package_name
```

---------------
# Build caffe
## Modify `Makefile.config`.
```shell
# Anaconda Python distribution is quite popular. Include path:
# Verify anaconda location, sometimes it's in root.
ANACONDA_HOME := $(HOME)/anaconda3/envs/py27
PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
                  $(ANACONDA_HOME)/include/python2.7 \
                  $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include
PYTHON_LIB := $(ANACONDA_HOME)/lib
```
