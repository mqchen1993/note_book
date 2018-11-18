# matlab install on ubuntu
---
[TOC]
#1 mount matlab.iso
```shell
$ mkdir matlab 
$ sudo mount -t auto -o loop Linux/R2016b_glnxa64_dvd1.iso matlab/
```
#2 install
```shell
$ sudo ./matlab/install
```
#3 mount dvd2
```shell
$ sudo mount -t auto -o loop Linux/R2016b_glnxa64_dvd2.iso matlab/
```
#4 unmount 
```shell
$ umount matlab/
$ sudo rm -r matlab/ # 删除空的文件夹
```
#5 激活Matlab
```shell
$ cd /usr/local/MATLAB/R2016b/bin
$ ./matlab # 如果是第一次运行，建议加sudo
```
* 载入激活文件license_standalone.lic
```shell
$ sudo cp Crack/R2016b/bin/glnxa64/libcufft.so.7.5.18 /usr/local/MATLAB/R2016b/bin/glnxa64
$ sudo cp Crack/R2016b/bin/glnxa64/libinstutil.so /usr/local/MATLAB/R2016b/bin/glnxa64
$ sudo cp Crack/R2016b/bin/glnxa64/libmwlmgrimpl.so /usr/local/MATLAB/R2016b/bin/glnxa64
$ sudo cp Crack/R2016b/bin/glnxa64/libmwservices.so  /usr/local/MATLAB/R2016b/bin/glnxa64
```
#6 命令行启动matlab
```shell
$ vim ~/.bashrc
$ export PATH=/usr/local/MATLAB/R2016b/bin:$PATH
```
#7 设置快捷方式
```shell
* sudo vim /usr/share/applications/Matlab2016b.desktop
```
输入：
```shell
[Desktop Entry]
Encoding=UTF-8
Name=Matlab 2016b
Comment=MATLAB
Exec=/usr/local/MATLAB/R2016b/bin/matlab
Icon=/usr/local/MATLAB/R2016b/toolbox/nnet/nnresource/icons/matlab.png
Terminal=true
StartupNotify=true
Type=Application
Categories=Application;
```
