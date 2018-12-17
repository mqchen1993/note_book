# 1. jupyter install
1. sudo pip install jupyter
2. "export PATH=$PATH:~/.local/bin" to ~/.bashrc
3. jupyter notebook `or` jupyter-notebook

# 2. remote login jupyter service.
## method 1: ssh远程使用jupyter notebook
1. 在远程服务器上，启动jupyter notebooks服务
```shell
$ jupyter notebook --no-browser --port=8889
```
2. 在本地终端中启动SSH
```shell
$ ssh -N -f -L localhost:8888:localhost:8889 username@serverIP
```
> 其中： -N 告诉SSH没有命令要被远程执行； -f 告诉SSH在后台执行； -L 是指定port forwarding的配置，远端端口是8889，本地的端口号的8888。
3. 最后打开浏览器，访问：http://localhost:8888/ 

## method 2: 利用jupyter notebook自带的远程访问功能
1. 生成默认配置文件
```shell
$ jupyter notebook --generate-config
```
2. 生成访问密码(token)
* 终端输入ipython，设置你自己的jupyter访问密码，注意复制输出的sha1:xxxxxxxx密码串.
```shell
In [1]: from notebook.auth import passwd
In [2]: passwd()
Enter password:
Verify password:
Out[2]: 'sha1:xxxxxxxxxxxxxxxxx'
```
3. 修改~/.jupyter/jupyter_notebook_config.py中对应行如下:
```shell
c.NotebookApp.ip='0.0.0.0'
c.NotebookApp.password = u'sha:ce...刚才复制的那个密文'
c.NotebookApp.open_browser = False
c.NotebookApp.port =8888 #可自行指定一个端口, 访问时使用该端口
```
4. 在服务器上启动jupyter notebook.
5. 在本机打开浏览器，访问：http://ip:8888/，输入设置的密码。
