# see build wiki
[boteye wiki](https://github.com/baidu/boteye/wiki/Build)

# 1. Dependency Installation
* Download 3rdparty_lib_lean according to your OS:
> Ubuntu 14.04(Updated 11-19-2018): [3rdparty_lib_lean.tar.gz](http://xteam.bj.bcebos.com/Product/ubuntu_14/3rdparty_lib_lean_x86_64_14.04_11-19-18.tar.gz?authorization=bce-auth-v1/d5982a7be6d84c7bb39c118d0b48cab8/2018-11-19T10:38:28Z/-1/host/defc767d12cbaa60adf360b77fcddff1ad057b73a997eecc47b110ba365933fe)

> Ubuntu 16.04(Updated 11-19-2018): [3rdparty_lib_lean.tar.gz](http://xteam.bj.bcebos.com/Product/ubuntu_16/3rdparty_lib_lean_x86_64_16.04_11-19-18.tar.gz?authorization=bce-auth-v1/d5982a7be6d84c7bb39c118d0b48cab8/2018-11-19T10:40:32Z/-1/host/911edc84ac2cec7d766b3e4e75af4ab9b84f9bd90a80d4a0b2fd1699315a2614)

* Run the following command to install dependencies from 3rdparty_lib_lean.tar.gz:
```shell
$ mkdir ~/XP_release
$ tar -xvf [3rdparty_lib_lean_mmddyyy.tar.gz] -C ~/XP_release
$ source 3rdparty_lib_lean/update.sh
```
# 2. Pre-built libraries
* Download pre-built libraries according to your OS: 
> Ubuntu 14.04(Updated 11-19-2018): [lib_x86_64.tar.gz](http://xteam.bj.bcebos.com/Product/ubuntu_14/lib_x86_64.tar.gz?authorization=bce-auth-v1/d5982a7be6d84c7bb39c118d0b48cab8/2018-11-19T10:38:51Z/-1/host/ea2da691c4242ba38955ee94f1bcd23103f33556dd6c3e663ee7425998a3100d)

> Ubuntu 16.04(Updated 11-19-2018): [lib_x86_64.tar.gz](http://xteam.bj.bcebos.com/Product/ubuntu_16/lib_x86_64.tar.gz?authorization=bce-auth-v1/d5982a7be6d84c7bb39c118d0b48cab8/2018-11-19T10:41:05Z/-1/host/5b5276e9165a85f9d689e1e2eb121eae5cdbe063429eff4a463dd28c121d13b5)

* Extract it to your `boteye` root directory(make sure the entire folder is included in the root directory, not just the contents within): 
```shell
$ tar -xvf lib_x86_64.tar.gz
```

# 3. Build
```shell
cd apps
mkdir build
cd build
cmake ..
make -j2
```

# run app 
```shell
$ source config_environment.sh
$ ./app_tracking/app_tracking -sensor_type XP3 -calib_file ../../calib_params/XP3.yaml -show_depth
```
