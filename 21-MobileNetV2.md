
-------------------------------------------------------
# 2019.03.14
## MobileNetV2
- [x] **Paper Reading**
### MobileNetV2
* Based on an `inverted residual` structure where the shortcut connections are between the thin bottleneck layers. The intermediate expansion layer uses `lightweight depthwise convolutions` to filter features as a source of non-linearity. 
* Additionally, we find that it is important to `remove non-linearities in the narrow layers` in order to maintain representational power.

### Semantic Segmentation
* we compare `MobileNetV1` and `MobileNetV2` models used as feature extractors with `DeepLabv3`.
* we experimented with three design variations: 
1. different feature extractors, 
2. simplifying the DeepLabv3 heads for faster computation,
3. different inference strategies for boosting the performance.

* We have observed that (`Table 7`): 
1. the inference strategies, including multi-scale inputs and adding leftright flipped images, significantly increase the MAdds and thus are not suitable for on-device applications,
2. using `output stride = 16` is more efficient than `output stride = 8`
3.  `MobileNetV1` is already a `powerful feature extractor` and only requires about 4.9−5.7 times fewer `MAdds`(number of multiply-adds operators) than `ResNet-101` (e.g., mIOU:78.56 vs 82.70, and MAdds: 941.9B vs 4870.6B)
4. It is more efficient to build DeepLabv3 heads on top of `the second last feature map` of `MobileNetV2` than on the original last-layer feature map, since the second to last feature map contains 320 channels instead of 1280
5. `DeepLabv3` heads are computationally expensive and `removing the ASPP module` significantly reduces the MAdds with only a slight performance degradation. 

* Notably, our architecture combined with the `SSDLite detection module` is 20× less computation and 10× less parameters than YOLOv2.




---------------------------------------------------------
# 2019.03.18
## MobileNetV2
- [x] [谷歌发布MobileNetV2](http://baijiahao.baidu.com/s?id=1596801861393928295&wfr=spider&for=pc) 
* 发现 Tensoflow Object API: 'models/research/object_detection'
* 在语义分割基准 PASCAL VOC 2012 上，`MobileNetV1` 与 `MobileNetV2` 作为特征提取器表现相当，但是后者所需的参数量减少了 5.3 倍，在 Multiply-Adds 方面 operations 也减少了 5.2 倍。

|	Model					|	Params	| 	Multiply-Adds	|	mIOU	|
|	------					|	------	|	------			|	------	|
| MobileNetV1 + DeepLabV3	|	11.15M	|	14.25B			|	75.29%	|
| MobileNetV2 + DeepLabV3	|	2.11M	|	2.75B			|	75.32%	|
	


--------------------
## MobileNetV2
- [x] [如何评价MobileNetV2?](https://www.zhihu.com/question/265709710)
*  在实际使用的时候， 我们发现Depthwise 部分的kernel比较容易训废掉： 训完之后发现depthwise训出来的kernel有不少是空的

### Inverted residual 以及Linear Bottlenecks的影响：
* 缓解特征退化：
Linear Bottleneck 通过去掉Eltwise+ 的特征去掉ReLU， 减少ReLU对特征的破坏； Invered residual 有两个好处： 1. 复用特征， 2. 旁支block内先通过1x1升维， 再接depthwise conv以及ReLU,  通过增加ReLU的InputDim， 来缓解特征的退化情况.

* 效率优先的网络结构设计：
以前大多数的模型结构加速的工作都停留在压缩网络参数量上。 其实这是有误导性的： 参数量少不代表计算量小； 计算量小不代表理论上速度快（带宽限制）。 ResNet的结构其实对带宽不大友好： 旁路的计算量很小，eltwise+ 的特征很大，所以带宽上就比较吃紧。由此看Inverted residual 确实是个非常精妙的设计！   其实业界在做优化加速， 确实是把好几个层一起做了， 利用时空局部性，减少访问DDR来加速。 所以 Inverted residual  带宽上确实是比较友好。

* 近两年是深度可分离卷积（Depth-wise Separable Convolution）大红大紫的两年，甚至有跟ResNet的Skip-connection一样成为网络基本组件的趋势。Xception论文指出，这种卷积背后的假设是跨channel相关性和跨spatial相关性的`解耦`。

* Evolution of separable convolution blocks.

![separable convolution blocks](https://github.com/kinglintianxia/note_book/blob/master/imgs/MobileNetV2_SCB.png)

图a中普通卷积将channel和spatial的信息同时进行映射，参数量较大； 图b为可分离卷积，解耦了channel和spatial，化乘法为加法，有一定比例的参数节省；图c中进行可分离卷积后又添加了bottleneck，映射到低维空间中；图d则是从低维空间开始，进行可分离卷积时扩张到较高的维度（前后维度之比被称为expansion factor，扩张系数），之后再通过1x1卷积降到原始维度。

> 这里要指出的一点是，观察和设计一个网络时，隐含的会有“状态层”和“变换层”的划分。当上图中c和d的结构堆叠起来时，其事实上是等价的。在这个视角下，我们可以把channel数少的张量看做`状态层`，而channel数多的张量看做`变换层`。这种视角下，在网络中传递的特征描述是压缩的，进行新一轮的变换时被映射到channel数相对高的空间上进行运算（文中称这一扩张比例为扩张系数，实际采用的数值为6），之后再压缩回原来的容量。

### Inverted Residual & Linear Bottleneck
* 第一点是skip-connection位置的迁移，即从连接channel数多的特征表述迁移为连接channel数少的特征表述。
* 第二点是用线性变换层替换channel数较少的层中的ReLU，这样做的理由是ReLU会对channel数低的张量造成较大的信息损耗。我个人的理解是ReLU会使负值置零，channel数较低时会有相对高的概率使某一维度的张量值全为0，即张量的维度减小了，而且这一过程无法恢复。张量维度的减小即意味着特征描述容量的下降。

![separable convolution blocks](https://github.com/kinglintianxia/note_book/blob/master/imgs/MobileNetV2_Inverted_residual_block.png)

因而，在需要使用ReLU的卷积层中，将channel数扩张到足够大，再进行激活，被认为可以降低激活层的信息损失。文中举了这样的例子：

![separable convolution blocks](https://github.com/kinglintianxia/note_book/blob/master/imgs/MobileNetV2_Linear_Bottleneck.png)

> 上图中，利用`nxm`的矩阵B将张量（2D，即m=2）变换到`n维`的空间中，通过ReLU后（y=ReLU(Bx)），再用此矩阵之逆恢复原来的张量。可以看到，当n较小时，恢复后的张量坍缩严重，n较大时则恢复较好。

* [Caffe Implementation of Google's MobileNets (v1 and v2)](https://github.com/shicai/MobileNet-Caffe)

[vis of MobileNetV2(NetScope)](http://ethereon.github.io/netscope/#/gist/d01b5b8783b4582a42fe07bd46243986) <br>

[vis of MobileNetV2(Netron)](http://lutzroeder.github.io/netron/?gist=d01b5b8783b4582a42fe07bd46243986)
* conv2_1/expand: 1x1 Conv, 6xchannels, point-wise
* conv2_1/dwise:  3x3 Conv, depth-wise
* conv2_1/linear: 1x1 Conv, 1/6channels, point-wise


### ReLU6
首先说明一下`ReLU6`，卷积之后通常会接一个ReLU非线性激活，在Mobile v1里面使用ReLU6，ReLU6就是普通的ReLU但是限制最大输出值为6（对输出值做clip），这是为了在移动端设备float16的低精度的时候，也能有很好的数值分辨率，如果对ReLU的激活范围不加限制，输出范围为0到正无穷，如果激活值非常大，分布在一个很大的范围内，则低精度的float16无法很好地精确描述如此大范围的数值，带来精度损失。

### 网络结构

![](https://pic3.zhimg.com/80/v2-b5af2ae3e210901ec31c79dc1e395fab_hd.jpg)
> `左边是v1`的没有residual connection并且带最后的ReLU，`右边是v2`的带residual connection并且去掉了最后的ReLU：


--------------------
## MobileNetV2
- [x] [MobileNet V2 论文初读](https://zhuanlan.zhihu.com/p/33075914)
* 之前用过 `MobileNet V1` 的准确率不错，更重要的是速度很快，在 `Jetson TX2`上都能达到`38 FPS`的帧率.

### 1. 对比 MobileNet V1 与 V2 的微结构

![MobileNetV1_V2](https://github.com/kinglintianxia/note_book/blob/master/imgs/MobileNetV1_V2.png)

* 都采用 Depth-wise (DW) 卷积搭配 Point-wise (PW) 卷积的方式来提特征。这两个操作合起来也被称为 `Depth-wise Separable Convolution`，之前在 Xception 中被广泛使用。
这么做的好处是理论上可以成倍的减少卷积层的`时间复杂度和空间复杂度`, 因为`卷积核的尺寸K` 通常远小于输出通道数 $C_{out}$，因此标准卷积的计算复杂度近似为`DW + PW`组合卷积的 $K^2$倍。

* V2 在 DW 卷积之前新加了一个 PW 卷积。
这么做的原因，是因为 DW 卷积由于本身的计算特性决定它自己没有改变通道数的能力，上一层给它多少通道，它就只能输出多少通道。所以如果上一层给的通道数本身很少的话，DW 也只能很委屈的在低维空间提特征，因此效果不够好。现在 V2 为了改善这个问题，给每个 DW 之前都配备了一个 PW，专门用来升维，定义升维系数 t = 6，这样不管输入通道数 $C_{in}$ 是多是少，经过第一个 PW 升维之后，DW 都是在相对的更高维 ( $t \cdot C_{in}$ ) 进行着辛勤工作的。
* V2 去掉了第二个 PW 的激活函数。
论文作者称其为 Linear Bottleneck。这么做的原因，是因为作者认为`激活函数在高维空间能够有效的增加非线性，而在低维空间时则会破坏特征，不如线性的效果好`。由于第二个 PW 的主要功能就是降维，因此按照上面的理论，降维之后就不宜再使用 ReLU6 了。

### 2. 对比 ResNet 与 MobileNet V2 的微结构

![MobileNetV2_ResNet](https://github.com/kinglintianxia/note_book/blob/master/imgs/MobileNetV2_ResNet.png)

* MobileNet V2 借鉴 ResNet，都采用了 [1x1 -> 3x3 -> 1x1] 的模式.
* MobileNet V2 借鉴 ResNet，同样使用 `Shortcut`将输出与输入相加.
* ResNet 使用`标准卷积`提特征，MobileNet 始终使用`DW卷积`提特征。
* ResNet 先降维 (0.25倍)、卷积、再升维，而 MobileNet V2 则是 先升维 (6倍)、卷积、再降维。直观的形象上来看，`ResNet 的微结构是沙漏形`，而 `MobileNet V2 则是纺锤形`，刚好相反。因此论文作者将 MobileNet V2 的结构称为`Inverted Residual Block`。这么做也是因为使用DW卷积而作的适配，希望特征提取能够在高维进行。

### 3. 对比 MobileNet V1 和 V2 的宏结构
* MobileNetV1

![MobileNetV1](https://pic2.zhimg.com/80/v2-81afb871cfd4a36220c33c7282ebeb3d_hd.jpg)

* MobileNetV2

![MobileNetV2](https://pic4.zhimg.com/80/v2-22299048d725a902a84010675fe84a13_hd.jpg)

* 基本参数汇总:

![](https://pic1.zhimg.com/80/v2-06186cd3bf792c1c2e4fdce13246cc1c_hd.jpg)

### 4. 在复现时遇到的问题
* 表述前后不一致。`论文`里面文字描述说有`19个 Bottleneck Residual Block`，但是之后给出的网络结构表（`论文中的Table 2`）里却只列出了`17个`。Table 2 第五行的 stride 和第六行的输入尺寸也是矛盾的。最后一行的输入通道数应该是1280而不是k。最后似乎也没有用 Softmax，不知道是否有意为之等等。


### MobileNetV2 网络结构图:

[MobileNetV2](https://pic1.zhimg.com/v2-8387d7ca2bed54e6f55bc0d5984bc6d4_r.jpg)


--------------------
## MobileNetV2
- [x] [Google AI Blog](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html)

* MobileNetV2: The Next Generation of On-Device Computer Vision Networks 
* [Official tensorflow code](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)
* MobileNetV2 is a very effective `feature extractor` for `object detection` and `segmentation`.

* [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)


![MobileNetV2 Building block](https://1.bp.blogspot.com/-M8UvZJWNW4E/WsKk-tbzp8I/AAAAAAAAChw/OqxBVPbDygMIQWGug4ZnHNDvuyK5FBMcQCLcBGAs/s1600/image5.png)



-----------------------
# 2019.03.18
## MobileNetV2
- [x] **MobileNetV2 Code Reading**
* [Official tensorflow code](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)
* Keep in mind that warm-starting from `a checkpoint` affects the model's weights `only during the initialization` of the model. Once a model has started training, `a new checkpoint` will be created in ${TRAIN_DIR}. If the fine-tuning training is `stopped and restarted`, this new checkpoint will be the one from which weights are restored and `not the ${checkpoint_path}$`.

* Typically for `fine-tuning` one only want train a sub-set of layers, so the flag `--trainable_scopes` allows to specify which subsets of layers should trained, the rest would remain frozen.

* models/tree/master/research/slim/README.md
### Run label image in C++

* Run `mobilenet_example.ipynb` Succeed!
Note: `base_name` must be `Absolute directory`.

* @slim.add_arg_scope
```python
"""
并不是所有的方法都能用arg_scope设置默认参数, 只有用@slim.add_arg_scope修饰过的方法才能使用arg_scope. 
所以, 要使slim.arg_scope正常运行起来, 需要两个步骤:

1. 用@add_arg_scope修饰目标函数
2. 用with arg_scope(...) 设置默认参数.
"""
@slim.add_arg_scope 
def fn(a, b, c=3): 
	d = c + b 
	print("a={}, b={}".format(a, b)) 
	return d 

with slim.arg_scope([fn], a = 1): 
	fn(b = 2)

```

* mobilenet.py/mobilenet_base()
`output_stride`: An integer that specifies the requested ratio of input to output spatial resolution. `If not None`, then we invoke `atrous convolution` if necessary to prevent the network from reducing the spatial resolution of the activation maps. 
`Allowed values` are 1 or any even number, `excluding zero`. Typical values are 8 (accurate fully convolutional mode), 16 (fast fully convolutional mode), and 32 (classification mode).

* `mobilenet_example.ipynb` -> `mobilenet_v2.py` -> `mobilenet.py` -> `conv_blocks.py`










