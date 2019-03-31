
------------------------------------------------------
# 2019.03.06
## DANet
* [github](https://github.com/junfu1115/DANet)
* [CityScapes detailed-results](https://www.cityscapes-dataset.com/detailed-results/)
-------------------
- [x] [DANet&CCNet](https://segmentfault.com/a/1190000018271713)
* 两篇文章都是将self-attention机制应用到分割当中，扩大感受野。第二篇文章采用了更巧妙的方法来减少参数。
* `self-attention`在分割中应用的大致思想是：`特征图`与`特征图的转置`进行矩阵相乘，由于特征图有`channel维度`，相当于是每个像素与另外每个元素都进行`点乘操作`，而向量的点乘几何意义为计算`两个向量的相似度`，两个向量越相似，它们点乘越大。看下图，特征图转置与特征图矩阵相乘后用softmax进行归一化就得到了`Attention map S`。S再与特征图的转置进行矩阵相乘，这个操作`把相关性信息重新分布到原始特征图上`，最后再将这个信息与特征图A相加，得到最终输出，这个输出结合了`整张图的相关性结果`。

![](https://i.loli.net/2019/02/25/5c73485641635.png)

* 整个网络的框架如下图：非常简单，特征提取->attention module->上采样得到分割图

![](https://c2.staticflickr.com/8/7896/40180410623_4f9679fd0e_c.jpg)

> 除了上面说的那一部分attention，作者还加了`蓝色channel attention`，在这里计算特征图与特征图转置矩阵相乘操作时，相乘的顺序调换了一下，这相当于是让channel与channel之间进行点乘操作，计算channel之间的相似性，在这里我认为每张channel map代表了不同类别，这样让类别与类别计算距离，来进行辅助。
----------------------
## DANet
- [x] [几篇较新的计算机视觉Self-Attention](https://zhuanlan.zhihu.com/p/44031466?utm_source=wechat_session&utm_medium=social&utm_oi=963402370776072192&from=timeline&isappinstalled=0)
* 总的来说，就是区域权值学习问题：
1. `Hard-attention`，就是0/1问题，哪些区域是被 attentioned，哪些区域不关注.
2. `Soft-attention`，[0,1]间连续分布问题，每个区域被关注的程度高低，用0~1的score表示.
3. `Self-attention`自注意力，就是 feature map 间的自主学习，分配权重（可以是 spatial，可以是 temporal，也可以是 channel间）

### Non-local NN, CVPR2018
* `主要思想`也很简单，CNN中的 convolution单元每次只关注`邻域 kernel size 的区域`，就算后期感受野越来越大，终究还是局部区域的运算，这样就`忽略了全局`其他片区（比如很远的像素）对当前区域的贡献。
* 所以 `non-local blocks` 要做的是，`捕获这种 long-range关系`：对于`2D图像`，就是图像中任何像素对当前像素的关系权值；对于`3D视频`，就是所有帧中的所有像素，对当前帧的像素的关系权值。
* **网络框架图!!!**:

![](https://pic4.zhimg.com/80/v2-b7805f52179e0313c97b67984866a98f_hd.jpg)
* 在这里简单说说在DL框架中最好实现的 Matmul 方式：
1. 首先对输入的 feature map X 进行线性映射（说白了就是[1x1x1]卷积来压缩通道数),然后得到$\theta$,$\phi$, $g$ 特征
2. 通过reshape操作，强行合并上述的三个特征除通道数外的维度，然后对$\theta$和$\phi$进行矩阵点乘操作，得到类似协方差矩阵的东西（这个过程很重要，计算出特征中的自相关性，即得到每帧中每个像素对其他所有帧所有像素的关系）
3. 然后对自相关特征 以列or以行（具体看矩阵$g$的形式而定） 进行 Softmax 操作，得到0~1的weights，这里就是我们需要的 Self-attention 系数
4. 最后将 attention系数，对应乘回特征矩阵$g$中，然后`再上扩channel数`，与原输入feature map X `残差`一下，完整的 bottleneck.

### Interaction-aware Attention, ECCV2018
* 就是在 non-local block 的协方差矩阵基础上，设计了基于 PCA 的新loss，更好地进行特征交互。作者认为，这个过程，特征会在channel维度进行更好的 non-local interact，故称为 Interaction-aware attention.
* 文中不直接使用`协方差矩阵的特征值分解`来实现, 通过`PCA`来获得 `Attention weights`
* 有点`小区别`是，在 X 和 Watten 点乘后，`还加了个b项`，文中说这里可看作 data central processing (subtracting mean) of PCA.

### CBAM: Convolutional Block Attention Module, ECCV2018
* 基于 `SE-Net`中的 Squeeze-and-Excitation module 来进行进一步拓展
* 文中把 `channel-wise attention` 看成是教网络 `Look 'what’`；而`spatial attention` 看成是教网络 `Look 'where'`，所以它比 SE Module 的主要优势就多了后者.
* 先看看SE-module流程：
1. 将输入特征进行`Global AVE pooling`，得到 `[1x1xChannel]` 
2. 然后`bottleneck`特征交互一下，`先压缩channel数`，`再重构回channel数`最后接个`sigmoid`，生成channel间`0~1`的 attention weights，最后scale乘回原输入特征.

![](https://pic4.zhimg.com/80/v2-2e8c37ad7e40b7f1cdfd81ecbae4e95f_hd.jpg)

* 再看看CBAM：
1. `Channel Attention Module`: 基本和 SE-module 是一致的，就额外加入了 Maxpool 的 branch。在 Sigmoid 前，两个 branch 进行 element-wise summation 融合。
2. `Spatial Attention Module`: 对输入特征进行 channel 间的 AVE 和 Max pooling，然后 concatenation，再来个7*7大卷积，最后`Sigmoid`.

![](https://pic1.zhimg.com/80/v2-a5ada5fb9ee0355b44e6a78f81ac1c58_hd.jpg)

### DANet, CVPR2019
* 很早就挂在了arXiv，`最近被CVPR2019接收`，把`Self-attention`的思想用在图像分割，可通过`long-range上下文关系`更好地做到精准分割。
* 把deep feature map进行`spatial-wise self-attention`，同时也进行`channel-wise self-attetnion`，最后将两个结果进行`element-wise sum`融合。
* 好处是：借鉴CBAM`分别进行空间和通道`self-attention的思想上，直接使用了 non-local 的自相关矩阵Matmul的形式进行运算，避免了CBAM手工设计pooling，多层感知器等复杂操作。

----------------------
## DANet
## Paper Reading
- [x] [DANet PPT](https://blog.csdn.net/mieleizhi0522/article/details/83111183) 
* SOTA(State of the art).
* 位置注意力模块(spatial-wise self-attention)通过所有位置的特征加权总和选择的性的聚集每个位置的特征，无论距离远近，相似的特征都会相互关联。(**类似于全连接条件随机场CRF**)
* 通道注意力模块(channel-wise self-attetnion)通过整合所有通道中的相关特征，有选择的性的强调相关联的通道。
* 基础网络为**DeepLab**
* **PSPNet** & **DeepLabV3**
* 位置注意力模块:

![](https://img-blog.csdn.net/20181017154326441?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pZWxlaXpoaTA1MjI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

* 通道注意力模块:

![](https://img-blog.csdn.net/20181017154336438?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pZWxlaXpoaTA1MjI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

* Position Attention Module:

![](https://img-blog.csdn.net/20181017154404699?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pZWxlaXpoaTA1MjI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

* Channel Attention Module:

![](https://img-blog.csdn.net/20181017154426685?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pZWxlaXpoaTA1MjI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

* DANet 网络整体结构

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/DANet.png)
> 我们的注意力模块很简单，可以直接插入现有的FCN模块中，不会增加太多参数，但会有效的增强特征表示。

* Dataset简介：

![](https://img-blog.csdn.net/20181017154447811?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pZWxlaXpoaTA1MjI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

* 训练参数和评价指标：

![](https://img-blog.csdn.net/20181017154457283?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pZWxlaXpoaTA1MjI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

* Visualization of Attention Module:
We observe that the `position attention module` could capture `clear semantic similarity` and longrange relationships.

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/DANet_Vis.png)

![](https://img-blog.csdn.net/20181017154534719?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pZWxlaXpoaTA1MjI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## Update 2019.03.17
* 对于通道注意力模块来说，很难直接给出关于注意力图的可视化理解:
Instead, We find that the response of `specific semantic` is noticeable after `channel attention module` enhances. For example, `11th channel map` responds to the ’car’ class in all three examples, and `4th channel map` is for the ’vegetation’ class, which benefits for the segmentation of two scene categories

![](https://img-blog.csdn.net/2018101715462287?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pZWxlaXpoaTA1MjI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


* MultiGrid:

![](https://img-blog.csdn.net/20181017154717952?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pZWxlaXpoaTA1MjI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

---------------------
## Update 2019.03.17

### Implementation Details
* we employ a `poly` learning rate policy
The `base learning rate` is set to `0.01` for `Cityscapes` dataset. `Momentum` and `weight decay` coefficients are set to `0.9 and 0.0001` respectively.

* We train our model with `Synchronized BN`. `Batchsize` are set to `8 for Cityscapes` and
`16 for other datasets`.When adopting multi-scale augmentation, 

* we set `training time` to `180 epochs for COCO` Stuff and `240 epochs for other datasets`.

* we also adopt `auxiliary supervision` on the top of two `attention modules`.

-------------------
## Update 2019.03.25
### From github
Note that: We adopt `multiple losses` in end of the network for `better training`.


-------------------
# Update 2019.03.25
- [ ] **Code Reading**
* `danet/train.py/get_segmentation_model()` -> `encoding/models/danet.py/DANetHead()` -> `encoding/nn/attention.py`
```python
# encoding/models/danet.py/DANetHead()
self.sa = PAM_Module(inter_channels)
self.sc = CAM_Module(inter_channels)

# encoding/nn/attention.py
class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))      # gamma: tensor([0.])

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        # proj_query: [batch, height*width, channels]
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        # proj_key: [batch, channels, height*width]
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        # torch.bmm: batch matrix multiply
        # energy: [batch, height*width, height*width]
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        # proj_value: [batch, channel, height*width]
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        # out: [batch, channel, height*width]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out: [batch, channel, height, width]
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        # proj_query: [batch, channels, height*width]
        proj_query = x.view(m_batchsize, C, -1)
        # proj_key: [batch, height*width, channels]
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # energy: [batch, channels, channels]
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        # proj_value: [batch, channel, height*width]
        proj_value = x.view(m_batchsize, C, -1)
        # out: [batch, channels, height*width]
        out = torch.bmm(attention, proj_value)
        # out: [batch, channels, height, width]
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
```

* why add "energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy" <br>
Ans: Prevent loss divergence during training(防止训练期间的损失不收敛)

* [I want to apply danet module to deeplab](https://github.com/junfu1115/DANet/issues/28) <br>
Que: I want to apply danet module to deeplab, so i am not sure how to insert it, wheater the front/middle/end of backbone(xception),or before ASPP module and behind ASPP, can you help me? <br>

Ans: Y can apply danet module at the end of backbone. If u want to use both aspp and danet, u can try to apply danet `before or behind aspp`, since we dont try aspp in our network, i can not give u a clear answer.

* junfu1115: We run our code for several times with the training script, the performances are above 79.5% on val set. <br>
I could get `77~78% on validation set` when `training result is 73%~74%`.

* Find a `concurrent work [OCNet](https://github.com/PkuRainBow/OCNet.pytorch)` <br>

`OCNet` achieves `81.67 on the test set of Cityscapes` with multi-scale method [0.75x, 1x, 1.25x]

![](https://raw.githubusercontent.com/PkuRainBow/OCNet.pytorch/master/OCNet.png)

* it seems you(DANet) used multi-grid by following psp rather than deeplabv3?

* Find `tensorflow self-attention implementation` [Self-Attention-GAN-Tensorflow
](https://github.com/taki0112/Self-Attention-GAN-Tensorflow)
```python
# SAGAN.py
def attention(self, x, ch, sn=False, scope='attention', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        f = conv(x, ch // 8, kernel=1, stride=1, sn=sn, scope='f_conv') # [bs, h, w, c']
        g = conv(x, ch // 8, kernel=1, stride=1, sn=sn, scope='g_conv') # [bs, h, w, c']
        h = conv(x, ch, kernel=1, stride=1, sn=sn, scope='h_conv') # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
        x = gamma * o + x

return x

```

* `encoding/models/danet.py/DANetHead()` <br>
```python
# Attention module `in_channels=2018`, `out_channels=num_classes`
        
class DANetHead(nn.Module):
    """
    in_channels: 2048
    out_channels: num_classes
    """
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4   # inter_channels: 512
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)                  # 3x3 conv, in_channels // 4 = 512
        sa_feat = self.sa(feat1)                # position attention, in_channels // 4 = 512
        sa_conv = self.conv51(sa_feat)          # 3x3 conv, in_channels // 4 = 512
        sa_output = self.conv6(sa_conv)         # 1x1 conv, num_classes

        feat2 = self.conv5c(x)                  # 3x3 conv, in_channels // 4 = 512
        sc_feat = self.sc(feat2)                # channel attention, in_channels // 4 = 512
        sc_conv = self.conv52(sc_feat)          # 3x3 conv, in_channels // 4 = 512
        sc_output = self.conv7(sc_conv)         # 1x1 conv, num_classes

        feat_sum = sa_conv+sc_conv              # self-attention merge, in_channels // 4 = 512
        
        sasc_output = self.conv8(feat_sum)      # self-attention out, 1x1 conv, num_classes

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)

```

* DANet 网络结构

|	Input			|	 Shape  						| 
|	------			|	------		  					|
| ResNet101(block4)	| [batch, height, width, 2048]		|
|	self-attention	| [batch, height/8, width/8, num_class]	|
|	upsample		| [batch, height, width]			|


* `DANet/encoding/models/base.py` <br>
pred = pred[0]


