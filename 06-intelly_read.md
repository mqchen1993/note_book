[进制转换工具](https://tool.lu/hexconvert/)
[在线异或校验](http://www.ip33.com/bcc.html)

# 1.安装 mNetAssist-release-amd64.deb

# 2.设置网口
	* ip：192.168.0.x (x=0~255,else 111)
	* netmask: 255.255.255.0
	* gate: 0.0.0.0

# 3.打开mNetAssist，配置如下：
	* 协议类型：TCP客户端
	* 服务器ip地址：192.168.0.111
	* 服务器端口：4001 数据，4002调试
	* 本地ip地址：192.168.0.2
	* 连接网络
	* 先在4002端口配置雷达自动上传，发送十六进制：
	  AA 77 77 AA A1 02 00 00 00 10 00 B3 77 AA AA 77
	* 再在4001端口接收数据，接收设置重导向文件。用sublime查看。
