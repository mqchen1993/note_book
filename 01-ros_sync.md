# 0. ros hostname
* pc: book.local
* ELAR-Systems.local
* robot
> export ROS_HOSTNAME=tegra-ubuntu.local
* desktop
> export ROS_HOSTNAME=localhost
> export ROS_MASTER_URI=http://tegra-ubuntu.local:11311

--------------------------------------------------
# 1. ssh 
## ssh to robot
* ssh tinker@10.42.0.129
* ssh tinker@10.42.0.43
## ssh cp
* scp /home/king/.vimrc tinker@10.42.0.129:~/
* scp king@10.106.18.46
## key_twist
* rosrun teleop_twist_keyboard teleop_twist_keyboard.py

SUBSYSTEM=="tty", ENV{ID_SERIAL_SHORT}=="0001",MODE="0666", OWNER="tinker", GROUP="tinker",SYMLINK+="rplidar"

## dl
* ssh lin@10.106.18.118

## king desk
* king@10.106.26.139

## king book
* king@10.106.26.216

## autolabor
* ssh root@192.168.2.1
	* key: autolabor

## nvidia
* ssh nvidia@10.42.0.43

--------------------------------------------------
# 2. udev rules

## Get idVendor&idProduct
```shell
$ lsusb
> ID `0403:6001` # (idVendor:idProduct)
# or 
$ udevadm info -q all -n /dev/ttyUSB0 | grep ID_VENDOR_ID
> ID_VENDOR_ID=0403
$ udevadm info -q all -n /dev/ttyUSB0 | grep ID_MODEL_ID
> ID_MODEL_ID=6001
```

## set the udev rule , make the device_port be fixed autolabor_pro1
```shell
KERNEL=="ttyUSB*", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", MODE:="0666", SYMLINK+="autolabor_pro1"
```
--------------------------------------------------
# tx2
sudo apt-get install ros-kinetic-dwa-local-planner
sudo apt-get install ros-kinetic-global-planner

# tx2_33
ssh nvidia@10.42.0.74

# 2018.09.01
## 33_tx2_vncviewer
wlan: 192.168.1.102
eth0: 10.42.0.1

--------------------------------------------------
# install x11vnc
* [blog1](https://blog.csdn.net/longhr/article/details/51657610)
* [blog2](https://blog.csdn.net/styshoo/article/details/52706138)

--------------------------------------------------
# ssh IPV6 
ssh -6 jun@fe80::3640:fb18:95b0:a93%enp3s0

--------------------------------------------------
# 3. linux cutecom
## install cutecom
```shell
$ sudo apt-get install cutecom
```
## install xgcom
* http://code.google.com/p/xgcom/ 下载xgcom源代码. 最新版本为 xgcom-0.04.2.tgz.
* install dependence.
```shell
$ sudo apt-get install automake
$ sudo apt-get install libglib2.0-dev
$ sudo apt-get install libvte-dev
$ sudo apt-get install libgtk2.0-dev
```
* build xgcom
```shell
$ ./autogen.sh
$ make
$ sudo make install
```

--------------------------------------------------
# 4. nvidia GPU FAN PWM
## 4.1 设置多显卡降温
```shell
$ nvidia-xconfig --enable-all-gpus
```
## 4.2. xorg.conf
```shell
$ cd /etc/X11
$ cp -p xorg.conf xorg.conf.origin
$ sudo vim xorg.conf
# 找到”Section Device” 这2块,添加： Option “Coolbits” “4”
Section "Device"
    Identifier     "Device0"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BoardName      "GeForce GTX 1080 Ti"
    BusID          "PCI:1:0:0"
    `Option         "Coolbits" "4"`
EndSection
```
## 4.3 设置主GPU
```shell
Section "ServerLayout"
    Identifier     "Layout0"
    Screen      0  "Screen0" RightOf "Screen1"
    Screen      1  "Screen1" 
    InputDevice    "Keyboard0" "CoreKeyboard"
    InputDevice    "Mouse0" "CorePointer"
EndSection
```
## 4.4 重启机器

## 4.5 手动修改GPU Fan
* Nvidia X Server Settings.
--------------------------------------------------
# 5. 设置xorg.conf 默认虚拟屏幕
```shell
$ cd /etc/X11
$ sudo vim xorg.conf
# add the line to Section "Device"
$ Option      "AllowEmptyInitialConfiguration" "true"
```

--------------------------------------------------
# Teamviewer
* 1078057326

# jun-pc
ip: 192.168.1.121

--------------------------------------
# 6. ubuntu统计程序代码量
```shell
$ sudo apt-get install cloc
$ cd $(Code_dir)/
$ cloc ./
      28 text files.
      28 unique files.
     493 files ignored.

http://cloc.sourceforge.net v 1.60  T=0.06 s (364.8 files/s, 88426.2 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
C++                              7            390            724           2211
C/C++ Header                     7            285            358           1380
CMake                            8             40             31            152
Bourne Shell                     1              1              0              3
-------------------------------------------------------------------------------
SUM:                            23            716           1113           **3746**
-------------------------------------------------------------------------------
```

# 2019.03.03
## images num

***		|	0901***
------	|	------
Upper	|	lower

