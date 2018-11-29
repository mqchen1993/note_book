# 1. 用户手册/1-8-基于虚拟机安装Ubuntu14.04.3操作系统.pdf
* 安装ubuntu虚拟机
* 共享文件目录 `/mnt/hgfs/00`

#2. 用户手册/1-5-Linux Processor-SDK安装.pdf
* 交叉编译环境

#3.  用户手册/2-1-AM437x开发板快速体验.pdf
* 创龙板子插上网线
* udhcpc
* ifconfig
* ubuntu虚拟机ping 通板子，即可scp文件。

#4. 用户手册/3-1-基于AM437x的Linux应用程序开发步骤演示.pdf
* helloworld 演示程序。

# 5.  用户手册/3-3-基于AM437x的OpenCV移植与开发例程.pdf
* opencv 移植。
* error: `../../3rdparty/lib/libzlib.a: could not read symbols: Bad value`
> arm-linux-gnueabihf.cmake中第一行SET(CMAKE_SYSTEM_NAME `Linux`) 
* scp from tronlong to ubuntu
```shell
$ sudo scp root@192.168.1.106:~/facedetect/result.bmp ~/Documents/am437x/facedetect/
```
