# 1 win10安装软碟通uiso9_cn_9.6.6.3300.exe
* ubuntu: startup disk creator

# 2 制作启动U盘

# 3	system settings->software&updates->软件源选tsinghua.edu.cn

# 4	安装搜狗输入法（百度），完成后添加sogou-pinyin

# 5	屏幕backlight调节（https://itsfoss.com/ubuntu-mint-brightness-settings/）

# 6	在文件夹打开终端 （restart）
		sudo apt-get install nautilus-open-terminal	

# 7	安装 foxit reader

# 8 	安装 wps （卸载自带）

# 9	安装 Adobe flash player(ubuntu software center)

# 10 	install shadowsocks-qt5(https://github.com/shadowsocks/shadowsocks-qt5/wiki/)
```shell
###ubuntu####
$ sudo add-apt-repository ppa:hzwhuang/ss-qt5
$ sudo apt-get install shadowsocks-qt5
### Settings ####
Server Address:		107.191.108.144(168.235.74.242)
Server Port:		8388
PassWord:		aqrose_vpn
Local Address:		127.0.0.1
Local Port:		1080
Local Server Type:	SOCKS5
Encryption Method:	AES-256-CFB
Automation:		yes
```
# 11 Firefox add-ons:
* Firefox
```shell
#1	adaware ad block
#2	Zoom Page
#3	FoxyProxy Standard
	-> Use proxy "Default" for all URLs 
	-> Edit Selection 
	-> Manual Proxy Configuration 
	-> IP:127.0.0.1 ; Port: 1080 ; SOCKS v5
firefox search box:
	-> Edit -> Preferences -> Search 
	-> Add more search engines -> baidu add-ons
```
* 终端使用ss
```shell
$ sudo apt-get install proxychains
$ sudo vim /etc/proxychains.conf
$ 将socks4         127.0.0.1 9050注释，增加socks5 127.0.0.1 1080
```
# 12 安装 qtcreator5(http://blog.csdn.net/win_turn/article/details/50465127)
* key: qaz1230@

# 13 Install `terminator`
```shell 
$ sudo apt-get install terminator
```
# 14 安装 ros

#	下载 Git 代码	

#	Qt & sublime

# 15 Install `smplayer`
```shell
$ sudo apt-get install smplayer
```

# 16 Install `simplescreenrecorder`
```sehll
$ sudo apt-get install simplescreenrecorder
```

# 17 开机开启数字小键盘(https://www.linuxidc.com/Linux/2016-09/135365.htm)

# 18 firefox 翻译插件
* ImTranslator

# 19 Install `PyCharm`
```sehll
$ sudo snap install [pycharm-professional|pycharm-community] --classic
```

# 20 sublime 安装 MarkDown 插件
* https://www.cnblogs.com/james-lee/p/6847906.html
