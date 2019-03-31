## **DeepLab V3+论文代码**
- [x] **deeplab_v2.docx**
- [x] **deeplab_v3.docx**
-------------------------
- [x] **deeplab_v3+.docx**
### DeepLabv3+ 网络结构

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/deeplab_v3+.png)

### DeepLabv3+使用Modified Aligned Xception，对Xception的改进如下:

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/Modified_Aligned_Xception.png)

* 层数变深了.
* 所有的`最大池化`都被`替换`成了`3x3 with stride 2` 的 Separable `Convolution`.
* 在每个 3x3 Depthwise Separable Convolution 的后面加了 BN 和 ReLU.
* 最终测试结果`Xception+decoder`的结构性能最优.

---------------------------
## DeepLabv3+
- [x] [deeplabv3+Xception](https://blog.csdn.net/u013711082/article/details/80415376)
## 1. Xception
### Inception的理念:
* 首先通过一系列的 1x1 卷积来学习`cross-channel correlations`，同时将输入的维度降下来；再通过常规的 3x3 和 5x5 卷积来学习`spatial correlations`。这样一来，两个卷积模块分工明确。Inception V3 中的 module 如下图:

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/xception_fig1.jpg)

### Inception的假设:
* `corss-channels correlations` 和 `spatial correlations` 是`分开学习`的，而不是在某一个操作中共同学习的。

### Inception到Xception(Extreme Inception) 的转变
1. 简版的Inception module:拿掉所有的pooling，并且只用一层3x3的卷积来提取spatial correlations，如Figure2。

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/xception_fig2.jpg)

2. 简版Inception:可以将这些1x1的卷积用一个较大的 1x1 卷积来替代（也就是在`channel上进行triple`），再在这个较大卷积产生的feature map上分出三个不重叠的部分，进行`separable convolution`，如 Figure3。 

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/xception_fig3.jpg)

> 这样一来就自然而然地引出：为什么不是`分出多个`不重叠的部分，而是`分出三个`部分来进行 separable convolution 呢？如果加强一下 Inception 的假设，假设 cross-channel correlations 和 spatial correlations 是完全无关的呢？

> 沿着上面的思路，一种`极端的情况`就是，在`每个channel`上进行 separable convolution，假设 1x1 卷积输出的 `feature map的channel有128个`，那么极端版本的Inception 就是在`每个channel`上进行3x3的卷积，而不是学习一个 3x3x128的kernel，取而代之的是`学习128个3x3的kernel`。 

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/xception_fig4.jpg)

3. Xception Architecture:一种`Xception module` 的线性堆叠，并且使用了`residual connection`(`残差单元`的输出由多个卷积层级联的输出和输入元素间相加)，数据依次流过`Entry flow`，`Middle flow` 和 `Exit flow`。

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/Xception.png)

### 顺便写一点读 Xception 时的小发现
* Xception 的实验有一部分是关于应不应该在`1x1卷积后面只用激活层的讨论`，实验结果是：如果在1x1卷积后`不加以激活`直接进行depthwise separable convolution，无论是在收敛速度还是效果上都`优于`在1x1 卷积后加以 ReLU 之类激活函数的做法。
> 这可能是因为，在对很浅的 feature（比如这里的 1-channel feature）进行激活会导致一定的信息损失，而对很深的 feature，比如 Inception module 提取出来的特征，进行激活是有益于特征的学习的，个人理解是这一部分特征中有大量冗余信息。

## 2. DeepLab V3+
* 论文里，作者直言不讳该框架参考了`spatial pyramid pooling (SPP) module`和`encoder-decoder` 两种形式的分割框架。前一种就是`PSPNet`那一款，后一种更像是`SegNet`的做法.
* `ASPP(Atrous Spatial Pyramid Pooling)`方法的优点是该种结构可以提取比较 dense 的特征，因为参考了不同尺度的 feature，并且 atrous convolution 的使用加强了提取 dense 特征的能力。但是在该种方法中由于 pooling 和有 stride 的 conv 的存在，使得分割目标的边界信息丢失严重.
* Encoder-Decoder 方法的 decoder 中就可以起到修复尖锐物体边界的作用。

### 关于Encoder中卷积的改进
* DeepLab V3+ 效仿了Xception中使用的`depthwise separable convolution`，在 DeepLab V3 的结构中使用了`atrous depthwise separable convolution`，降低了计算量的同时保持了相同（或更好）的效果。

### **Decoder的设计**
1. Encoder 提取出的特征首先被 x4 上采样，称之为 F1；
2. Encoder 中提取出来的与 F1 同尺度的特征 F2’ 先进行 1x1 卷积，降低通道数得到 F2，再进行 F1 和 F2 的 concatenation，得到 F3；
> 为什么要进行通道降维？因为在 encoder 中这些尺度的特征通常通道数有 256 或者 512 个，而 encoder 最后提取出来的特征通道数没有这么多，如果不进行降维就进行 concate 的话，无形之中加大了 F2’ 的权重，加大了网络的训练难度。
3. 对 F3 进行常规的 3x3 convolution 微调特征，最后直接 x4 upsample 得到分割结果。

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/deeplab_v3+_1.png)


### DeepLabv3+使用Modified Aligned Xception(BackBone),对Xception的改进如下:

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/Modified_Aligned_Xception.png)

* 层数变深了.
* 所有的`最大池化`都被`替换`成了`3x3 with stride 2` 的 Separable `Convolution`.
* 在每个 3x3 Depthwise Separable Convolution 的后面加了 BN 和 ReLU.
* 最终测试结果`Xception+decoder`的结构`性能最优`.


-------------------------
## DeepLabv3+
- [x] [PSPNet VS DeepLabv3](https://zhuanlan.zhihu.com/p/51132008)
### 网络结构
1. PSPNet:

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/PSPNet.png)

2. DeepLabv3:

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/DeepLabv3.png)

### 基础网络(Backbone)
1. PSPNet:
带`dilation卷积`的`ResNet系列`：ResNet50，ResNet101，ResNet152，ResNet269，以ResNet50为例，后面两个Block是dilation=2和4的。

2. DeepLabv3:
带有Multi-Grid（Multi-Grid=（1,2,4）最优）的dilation的ResNet系列。

### Loss
PSPNet:在Block4后面有一个辅助的分类Loss,而DeepLabv3：就一个最后的Loss

### 输出
PSPNet和DeepLabv3原论文中应该是**将label下采样到8倍后与输出进行比较**。

### 数据增强
1. PSPNet
> we adopt `random mirror` and `random resize` between 0.5 and 2 for all datasets, and additionally add `random rotation` between -10 and 10 degrees, and `random Gaussian blur` for ImageNet and PASCAL VOC.

2. DeepLabv3
> We apply data augmentation by `randomly scaling` the input images (from 0.5 to 2.0) and `randomly left-right flipping` during training.

### 训练策略
1. PSPNet
poly,初始学习率乘以$(1-\frac{iter}{maxiter})^{power}$, where power=0.9,Lr=0.01.
2. DeepLabv3
poly,初始学习率乘以$(1-\frac{iter}{maxiter})^{power}$, where power=0.9,The batch.


-------------------------
## DeepLabv3+ 
- [x] **Paper Reading**
* [语义分割：DeepLabV3+翻译](https://zhuanlan.zhihu.com/p/41150415)

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/SPP&Encoder-Decoder&E-D_Atrous_Conv.png)

### Tensorflow `models/research`:
1. deeplab
2. inception
3. resnet
4. models/samples/core/tutorials/keras
5. models/samples/core/tutorials/estimators
6. models/tutorials/image
7. models/research/slim


--------------------------------------------
# 2019.03.08
## DeepLabv3+ Train&Test
- [x] [TensorFlow DeepLabV3+训练自己的数据分割](https://zhuanlan.zhihu.com/p/42756363)
### CamVid数据集
当然出来cityspcapes数据集之外你也可以放许多其他的数据集。包括我自己的车道线数据集，分割效果也还不错，连左右车道线都能分割出来。

### 数据集制作
* 假设你用`labelme`或者其他工具标注了你的数据，你的保存标注可能是polygon的点，也可能是mask。这里我`推荐保存polygon`，因为deeplab中使用的`label`是单通道以你的`类别的id为像素值`的标签.

* 关于label有几点需要注意的，`像素值就是label的index`，从我的map也能看的出来.除此之外没了。另外，如果你的类别里面没有`ignore_label`, 那就直接是idx和0,0就是背景。`如果有ignore_label就是255`,相应的类别写进去，颜色值为255就是ignore了。

* 接下来你的生成相应的`tfrecords`




--------------------------
## DeepLabv3+ Train&Test
- [x] [在tensorflow上用其他数据集训练DeepLabV3+](https://www.jianshu.com/p/dcca31142b99)
### Apollo数据集
1. clone 谷歌官方源码到本地
2. 添加Python环境变量
$ export PYTHONPATH=$PYTHONPATH:/home/xxx/Downloads/models-master/research/slim
3. 测试一下
```shell
#在deeplab/research目录下运行
python deeplab/model_test.py
```
4. 生成图像mask。
5. 生成voctrain.txt vocval.txt为接下来生成tfrecord做准备。
6. 生成tfrecord

> When `fine_tune_batch_norm=True`, use at least batch size larger than 12 (batch size more than 16 is better). Otherwise, one could use smaller batch size and set fine_tune_batch_norm=False.

> --eval_crop_size=2710 这里=后不能有空格，--eval_crop_size= 2710 不然报错 所有的=后面都不能有空格 不然报错。

> if you want to `fine-tune` DeepLab on your own dataset, then you can modify some parameters in train.py, here has some options:

    you want to `re-use all the trained wieghts`, set `initialize_last_layer=True`
    you want to `re-use only the network backbone`, set `initialize_last_layer=False` and `last_layers_contain_logits_only=False`
    you want to `re-use all the trained weights except the logits`(since the num_classes may be different), set `initialize_last_layer=False` and `last_layers_contain_logits_only=True`

Finally, my setting is as follows:

    `initialize_last_layer=False`
    `last_layers_contain_logits_only=True`


* [ aquariusjay commented on Apr 11, 2018](https://github.com/tensorflow/models/issues/3730#issuecomment-380168917)

> When you want to fine-tune DeepLab on other datasets, there are a few cases:

1. You want to `re-use ALL the trained weigths`: set `initialize_last_layer = True` (last_layers_contain_logits_only does not matter in this case).

2. You want to `re-use ONLY the network backbone` (i.e., exclude ASPP, decoder and so on): set `initialize_last_layer = False` and `last_layers_contain_logits_only = False`.

3. You want to `re-use ALL the trained weights EXCEPT the logits` (since the num_classes may be different): `set initialize_last_layer = False` and `last_layers_contain_logits_only = True`.


--------------------------
## DeepLabv3+ Train&Test
- [x] [火焰识别--重新标注后的Deeplabv3+训练](https://blog.csdn.net/w_xiaowen/article/details/85289750)
* 将使用`labelme生成的json文件`转换成标注后的图片


--------------------------
## DeepLabv3+ Train&Test
- [x] [使用deeplabv3+训练自己的数据集经验总结](https://blog.csdn.net/Kelvin_XX/article/details/81946091)
* 为了能在`windows系统上运行shell脚本`，这里强烈推荐`[Git Bash](https://www.git-scm.com/download/)`。它是Git（没错，就是那个代码管理系统）软件的一个子程序，感觉比windows自带的cmd和powershell要好用！ 
* 然后`正常安装`，完毕后在任何文件夹的`空白处鼠标右键`，点击`Git Bash Here`选项，就可以在当前右键的路径下打开一个 Git Bash 控制台，长这个样子:

![](https://img-blog.csdn.net/20180822195005838?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0tlbHZpbl9YWA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

* VOC数据集的目录组织形式应当这样:
```python
+ Database # 自己的数据集名称 
	+ JPEGImages 
	+ SegmentationClass 
	+ ImageSets 
		+ Segmentation 
			- train.txt
			- val.txt
			- trainval.txt
+ tfrecord
```
其中：
1. `JPGImages` 文件夹存放RGB图像；
2. `SegmentationClass` 存放转换为class的标签，格式为单通道的png图像。对应的图像和标签的文件名相同！！扩展名分别为.jpg和.png
3. `ImageSets/Segmentation` 存放有图像文件名的 .txt 文件，这里我按文件名将数据分为 train, val, trainval； 
4. `tfrecord` 存放转换的 tfrecord 格式的数据。
5. `.txt` 文件中的内容应当为对应图像的文件名，不带扩展名：

* 转换为TFRecord 格式
可在代码中调节参数 `_NUM_SHARDS` （默认为4），改变`数据分块的数目`。（一些文件系统有最大单个文件大小的限制，如果数据集非常大，增加 _NUM_SHARDS 可减小单个文件的大小）

* **注册数据集**

* **Colormap**

--------------------------
## DeepLabv3+ Train&Test
- [x] [Deeplab V3+训练自己数据集全过程](https://blog.csdn.net/jairana/article/details/83900226)
* [labelme制作数据集](https://note.youdao.com/ynoteshare1/index.html?id=032620eac64634508cd4f9e65be4617c&type=note#/)

* 数据不平衡问题 <br>
在train_utils.py的70行修改权重

-------------------------
### Run DeepLabv3+
## 1. Install libraries
```shell
$ sudo apt-get install python-pil python-numpy
$ pip install --user jupyter
$ pip install --user matplotlib
$ pip install --user PrettyTable
```
## 2. Add Libraries to `PYTHONPATH`
```shell
# add this line to '~/.bashrc'
export PYTHONPATH=$PYTHONPATH:/home/jun/Documents/king/models/research:/home/jun/Documents/king/models/research/slim
```
## 3. Testing the Installation
```shell
# From tensorflow/models/research/deeplab
$ python model_test.py
# ---- It will print ----
Ran 5 tests in 16.648s

OK
```
-----------
## 4. VOC dataset
### Recommended Directory Structure for Training and Evaluation:
```shell
+ datasets
  + pascal_voc_seg
    + tfrecord						# convert from voc2012
    + exp	
      + train_on_train_set			# train_on_train_set stores the train/eval/vis events and results
        + train
        + eval
        + vis
```

### 4.1 Uncompress VOC2012 dataset.
```shell
$ cd /media/jun/ubuntu/datasets/VOC
# -x, extract files from an archive; -f, use archive file or device ARCHIVE;
# -v, verbosely list files processed
# -C, change to directory DIR
$ tar -xvf VOCtrainval_11-May-2012.tar -C ../
# cd to deeplab
$ cd /home/jun/Documents/king/models/research/deeplab/datasets
# Removes the color map from the ground truth segmentation annotations and save the results to output_dir.
$ python remove_gt_colormap.py --original_gt_folder=/media/jun/ubuntu/datasets/VOCdevkit/VOC2012/SegmentationClass/ --output_dir=/media/jun/ubuntu/datasets/VOCdevkit/VOC2012/SegmentationClassRaw
# mkdir 
$ mkdir -p pascal_voc_seg/tfrecord
# convert
$ python build_voc2012_data.py --image_folder=/media/jun/ubuntu/datasets/VOCdevkit/VOC2012/JPEGImages/ --semantic_segmentation_folder=/media/jun/ubuntu/datasets/VOCdevkit/VOC2012/SegmentationClassRaw/ --list_folder=/media/jun/ubuntu/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/ --image_format=jpg --output_dir=./pascal_voc_seg/tfrecord
## 各参数意义如下：
    `image_folder`： 保存images的路径
    `semantic_segmentation_folder`： 保存labels的路径
    `list_folder`： 保存train\val.txt文件的路径
    `image_format`： image的格式
    `output_dir`： 生成tfrecord格式的数据所要保存的位置

## Terminal print
>> Converting image 363/1449 shard 0
>> Converting image 726/1449 shard 1
>> Converting image 1089/1449 shard 2
>> Converting image 1449/1449 shard 3
```
------------------
### 4.2 A local `evaluation` job using `xception_65` can be run with the following command:
```shell
$ cd /home/jun/Documents/king/models/research/deeplab
# Download checkpoint from https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
$ mkdir pascal_voc_seg/model_zoo && cd pascal_voc_seg/model_zoo
$ wget http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz

# tar it & touch `checkpoint` file:
model_checkpoint_path: "./model.ckpt"
all_model_checkpoint_paths: "./model.ckpt"

# run eval, From tensorflow/models/research/deeplab
$ python eval.py --logtostderr --eval_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=513 --eval_crop_size=513 --dataset="pascal_voc_seg" --checkpoint_dir=./datasets/pascal_voc_seg/exp/train_on_train_set/train --eval_logdir=./datasets/pascal_voc_seg/exp/train_on_train_set/eval --dataset_dir=./datasets/pascal_voc_seg/tfrecord --max_number_of_iterations=1

# run eval, mobilenet_v2
$ python eval.py --logtostderr --eval_split="val" --model_variant="mobilenet_v2"  --output_stride=16 --eval_crop_size=513 --eval_crop_size=513 --dataset="pascal_voc_seg" --checkpoint_dir=./datasets/pascal_voc_seg/exp/train_on_train_set/train --eval_logdir=./datasets/pascal_voc_seg/exp/train_on_train_set/eval --dataset_dir=./datasets/pascal_voc_seg/tfrecord --max_number_of_iterations=1

## Terminal print
INFO:tensorflow:Finished evaluation at 2019-03-08-07:49:00
miou_1.0[0.935834229]
## Get output file: 'datasets/pascal_voc_seg/exp/train_on_train_set/eval/events.out.tfevents.1552031267.jun-pc'
$ tensorboard --logdir ./
```
---------------------
### 4.3 A local `visualization` job using `xception_65` can be run with the following command:
```shell
# run vis, From tensorflow/models/research/deeplab
$ python vis.py --logtostderr --vis_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=513 --vis_crop_size=513 --dataset="pascal_voc_seg" --checkpoint_dir=./datasets/pascal_voc_seg/exp/train_on_train_set/train --vis_logdir=./datasets/pascal_voc_seg/exp/train_on_train_set/vis --dataset_dir=./datasets/pascal_voc_seg/tfrecord --max_number_of_iterations=1

# run vis, mobilenet_v2
$ python vis.py --logtostderr --vis_split="val" --model_variant="mobilenet_v2" --output_stride=16 --vis_crop_size=513 --vis_crop_size=513 --dataset="pascal_voc_seg" --checkpoint_dir=./datasets/pascal_voc_seg/exp/train_on_train_set/train --vis_logdir=./datasets/pascal_voc_seg/exp/train_on_train_set/vis --dataset_dir=./datasets/pascal_voc_seg/tfrecord --max_number_of_iterations=1


## Get output file: 'datasets/pascal_voc_seg/exp/train_on_train_set/vis/segmentation_results'
```

---------------------
### 4.4 A local `training` job using `xception_65` can be run with the following command::
```shell
# run training, `re-use all the trained wieghts`. [OK]
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=8 --dataset="pascal_voc_seg" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_pascal_trainval/model.ckpt --train_logdir=./datasets/pascal_voc_seg/exp/train_on_train_set/train --dataset_dir=./datasets/pascal_voc_seg/tfrecord --num_clones=2 --fine_tune_batch_norm=False

# run training, `re-use all the trained weights except the logits` [OK] 
# step 19760: loss = 0.1650 (0.924 sec/step), miou_1.0[0.904196858]
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=8 --dataset="pascal_voc_seg" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_pascal_trainval/model.ckpt --train_logdir=./datasets/pascal_voc_seg/exp/train_on_train_set/train --dataset_dir=./datasets/pascal_voc_seg/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=True

# run training, `re-use all the trained weights except the logits` 	[OK]
# step 19760: loss = 0.1650 (0.924 sec/step), miou_1.0[0.904196858]
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=8 --dataset="pascal_voc_seg" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_pascal_trainval/model.ckpt --train_logdir=./datasets/pascal_voc_seg/exp/train_on_train_set/train --dataset_dir=./datasets/pascal_voc_seg/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=True

# run training, `re-use only the network backbone`	[OK]
# global step 20260: loss = 0.2941 (0.934 sec/step), miou_1.0[0.818994045]
# global step 30000: loss = 0.2861 (0.878 sec/step), miou_1.0[0.826865256]
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=8 --dataset="pascal_voc_seg" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_pascal_trainval/model.ckpt --train_logdir=./datasets/pascal_voc_seg/exp/train_on_train_set/train --dataset_dir=./datasets/pascal_voc_seg/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=False

---------------------------------------------------
# mobilenet_v2 on VOC dataset performance(val): 75.32% (OS=16), 77.33 (OS=8)
# 2019.03.15
## mobilenet_v2, `re-use all the trained weights except the logits` [OK]
## global step 8890: loss = 0.3322 (0.205 sec/step), miou_1.0[0.706052244]
## global step 30000: loss = 0.2057 (0.196 sec/step), miou_1.0[0.727050483]
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="mobilenet_v2" --output_stride=16 --train_crop_size=513 --train_crop_size=513 --train_batch_size=8 --dataset="pascal_voc_seg" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_pascal_train_aug/model.ckpt-30000 --train_logdir=./datasets/pascal_voc_seg/exp/train_on_train_set/train --dataset_dir=./datasets/pascal_voc_seg/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=True

# 2019.03.16
## mobilenet_v2, `re-use only the network backbone` 
## 
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="mobilenet_v2" --output_stride=16 --train_crop_size=513 --train_crop_size=513 --train_batch_size=8 --dataset="pascal_voc_seg" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_pascal_train_aug/model.ckpt-30000 --train_logdir=./datasets/pascal_voc_seg/exp/train_on_train_set/train --dataset_dir=./datasets/pascal_voc_seg/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=True


## Get output file: 'datasets/pascal_voc_seg/exp/train_on_train_set/train'
```

-----------
### 5. CityScapes dataset
### Recommended Directory Structure for Training and Evaluation:
```shell
+ datasets
  + cityscapes
    + tfrecord
    + exp
      + train_on_train_set
        + train
        + eval
        + vis
```

### 5.1 Prepare dataset and convert to TFRecord
```shell
# From tensorflow/models/research/deeplab/datasets
$ ./convert_cityscapes_me.sh
# This shell script run 'cityscapesscripts/preparation/createTrainIdLabelImgs.py'
# And then run 'build_cityscapes_data.py'

-------------------------
# Prepare `fine+coarse` data
## Coarse dataset 
* 'gt_Coarse/train_extra': 19998 files.
* fine + coarse: 22973 = 19998+2975
* `gt_Coarse` 生成的`*_gtCoarse_labelTrainIds.png`只有黑白两色? 视觉错误。

1. 修改`createTrainIdLabelImgs.py`只保留`searchCoarse` data.
$ python cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py 
2. 将`leftImg8bit`文件夹下`train_extra` copy 合并为`fine+coarse`; 将`gtFine`文件夹下`train_extra` copy 合并为`fine+coarse`;
3. 运行转换脚本
$ ./convert_cityscapes_fine_coarse.sh

4. 注册`fine+coarse` data.
_CITYSCAPES_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 2975,        # train images.
        'val': 500,           # val images.
        'train_extra':22973   # fine + coarse. 22973 = 19998+2975
    },
    num_classes=19,
    ignore_label=255,
)

```

------------------
### 5.2 A local `evaluation` job using `xception_65` can be run with the following command:
```shell
$ cd deeplab/datasets/model_zoo
# Download checkpoint from https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
$ wget http://download.tensorflow.org/models/deeplabv3_cityscapes_train_2018_02_06.tar.gz

# tar it & touch `checkpoint` file:
model_checkpoint_path: "./model.ckpt"
all_model_checkpoint_paths: "./model.ckpt"
-----------------------
# run eval, From tensorflow/models/research/deeplab
$ python eval.py --logtostderr --eval_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=1025 --eval_crop_size=2049 --dataset="cityscapes" --checkpoint_dir=./datasets/cityscapes/exp/train_on_train_set/train --eval_logdir=./datasets/cityscapes/exp/train_on_train_set/eval --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1

-----------------------
# run eval, xception_65
$ python eval.py --logtostderr --eval_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=1025 --eval_crop_size=2049 --dataset="cityscapes" --checkpoint_dir=./datasets/cityscapes/exp/train_on_train_set/xception_65/train --eval_logdir=./datasets/cityscapes/exp/train_on_train_set/xception_65/eval --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1

------------------------
## run eval, mobilenet_v2
$ python eval.py --logtostderr --eval_split="val" --model_variant="mobilenet_v2" --output_stride=16 --eval_crop_size=1025 --eval_crop_size=2049 --dataset="cityscapes" --checkpoint_dir=./datasets/cityscapes/exp/train_on_train_set/train --eval_logdir=./datasets/cityscapes/exp/train_on_train_set/eval --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1
### ASPP & Decoder
$ python eval.py --logtostderr --eval_split="val" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=1025 --eval_crop_size=2049 --dataset="cityscapes" --checkpoint_dir=./datasets/cityscapes/exp/train_on_train_set/train --eval_logdir=./datasets/cityscapes/exp/train_on_train_set/eval --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1

## Get 'Waiting for new checkpoint at...' 
## Terminal print
INFO:tensorflow:Finished evaluation at 2019-03-08-07:49:00
miou_1.0[0.935834229]
## Get output file: 'datasets/cityscapes/exp/train_on_train_set/eval/events.out.tfevents.1552031267.jun-pc'
$ tensorboard --logdir ./

```

---------------------
### 5.3 A local `visualization` job using `xception_65` can be run with the following command:
```shell
# run vis, From tensorflow/models/research/deeplab
$ python vis.py --logtostderr --vis_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=1025 --vis_crop_size=2049 --dataset="cityscapes" --colormap_type="cityscapes" --checkpoint_dir=./datasets/model_zoo/deeplabv3_cityscapes_train --vis_logdir=./datasets/cityscapes/exp/train_on_train_set/vis --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1

# run vis, mobilenet_v2
$ python vis.py --logtostderr --vis_split="val" --model_variant="mobilenet_v2"  --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=8  --decoder_output_stride=4 --vis_crop_size=1025 --vis_crop_size=2049 --dataset="cityscapes" --colormap_type="cityscapes" --checkpoint_dir=./datasets/cityscapes/exp/train_on_train_set/mobilenet_v2-backbone-aspp-decoder --vis_logdir=./datasets/cityscapes/exp/train_on_train_set/vis --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1

# run vis, mobilenet_v2
$ python vis.py --logtostderr --vis_split="val" --model_variant="mobilenet_v2"  --output_stride=16  --vis_crop_size=1025 --vis_crop_size=2049 --dataset="cityscapes" --colormap_type="cityscapes" --checkpoint_dir=./datasets/cityscapes/exp/train_on_train_set/train --vis_logdir=./datasets/cityscapes/exp/train_on_train_set/vis --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1

## Get output file: 'datasets/cityscapes/exp/train_on_train_set/vis/segmentation_results'
```

---------------------
### 5.4 A local `training` job using `xception_65` can be run with the following command::
```shell
# *9W* iters if fine for training.
# run training, From tensorflow/models/research/deeplab
# --train_batch_size <= 2

# Multi GPUs Training
## Fine-tuning `re-use all the trained wieghts`, train_batch_size=1, [OK]
$ python train.py --logtostderr --training_number_of_steps=90000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=1 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --fine_tune_batch_norm=False


## Fine-tuning  `re-use all the trained weights except the logits`
$ python train.py --logtostderr --training_number_of_steps=10000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=2 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False
                                
-------------------------------------------
## xception_65(val): 78.79%(OS=16), 80.42%(OS=8) 
## Training with Batch norm is essential to attain high performance.

# 2019.03.18           
## Fine-tuning  `re-use only the network backbone`		[0.72 VS 0.80]
## global step 11550: loss = 0.3878 (0.594 sec/step), miou_1.0[0.542078793]
## step 90000: loss = 0.4648 (0.570 sec/step), miou_1.0[0.718897283], miou_1.0[0.722017109](OS=8)      
$ python train.py --logtostderr --training_number_of_steps=90000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=2 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/xception_65/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=False

-------------------------------------------
## mobilenet_v2: miou_1.0[0.716534734](output_stride=8), miou_1.0[0.706700146](output_stride=16)

## mobilenet_v2, `re-use all the trained wieghts`, set `initialize_last_layer=True` [OK]
## Total number of params: 2194963 = 2.195M
## ('GFLOPs after freezing: ', 13.8740999319)
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="mobilenet_v2" --output_stride=8 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False

----------------
# 2019.03.16
## mobilenet_v2, `re-use all the trained wieghts`, 	[OK]
## Recording summary at step 2559, miou_1.0[0.67876631]
## global step 10000: loss = 0.1832 (0.433 sec/step), miou_1.0[0.698925376](OS=16), miou_1.0[0.707758665](OS=8)
$ python train.py --logtostderr --training_number_of_steps=10000 --train_split="train" --model_variant="mobilenet_v2" --output_stride=16 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False --save_summaries_images=True

----------------
# 2019.03.16
## mobilenet_v2, `re-use all the trained weights except the logits`  [OK]
## Recording summary at step 4643, miou_1.0[0.573157489]
## global step 10000: loss = 0.2033 (0.949 sec/step), miou_1.0[0.645294428]
## Recording summary at step 13278, loss = 0.2239 (0.881 sec/step), miou_1.0[0.668720305]
## Recording summary at step 20000, loss = 0.2694 (0.868 sec/step), miou_1.0[0.671730518]
## global step 43850: loss = 0.1665 (0.934 sec/step), miou_1.0[0.704537928]
## global step 90000: loss = 0.1702 (0.831 sec/step), miou_1.0[0.706493258](-1.0%), miou_1.0[0.687527776](OS=16, -1.9%)
$ python train.py --logtostderr --training_number_of_steps=90000 --train_split="train" --model_variant="mobilenet_v2" --output_stride=8 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=True --save_summaries_images=True


----------------
# 2019.03.18
## mobilenet_v2, `re-use only the network backbone`	[OK]
## global step 4630: loss = 0.2648 (0.477 sec/step), miou_1.0[0.556890249]
## global step 30000: loss = 0.2261 (0.442 sec/step), miou_1.0[0.667330086], miou_1.0[0.678636968](OS=8)
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="mobilenet_v2" --output_stride=16 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True

----------------
# 2019.03.19
## mobilenet_v2, `re-use only the network backbone`	[Failed], '0.588809609' VS '0.706700146'
## train_batch_size=16, fine_tune_batch_norm=True
## global step 7640: loss = 0.6005 (0.390 sec/step), miou_1.0[0.307944208]
## global step 28800: loss = 0.3123 (1.062 sec/step), miou_1.0[0.428906143]
## global step 65738, miou_1.0[0.581575751]
## global step 90000: loss = 0.3171 (1.109 sec/step), miou_1.0[0.588809609], miou_1.0[0.597531855](OS=8)
## `base_learning_rate` too small ? YEAH !!!
$ python train.py --logtostderr --training_number_of_steps=90000 --train_split="train" --model_variant="mobilenet_v2" --output_stride=16 --train_crop_size=769 --train_crop_size=769 --train_batch_size=16 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=True --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True

----------------
# 2019.03.20
## mobilenet_v2, `re-use only the network backbone`		[OK]
## train_batch_size=16, fine_tune_batch_norm=True, base_learning_rate=0.001
## global step 10440: loss = 0.2543 (0.965 sec/step), miou_1.0[0.684512675]
## global step 60000: loss = 0.2023, miou_1.0[0.699472129](-0.7%), miou_1.0[0.708855033](OS=8,-0.76%)
## global step 69490: loss = 0.1956, miou_1.0[0.699715734](-0.698%), miou_1.0[0.709764659](OS=8,-0.669%)
## global step 72490: loss = 0.1723, miou_1.0[0.701163888](-0.553%), miou_1.0[0.711194](OS=8, -0.534%)
## global step 90000: loss = 0.2030, miou_1.0[0.701635838](-0.506%), miou_1.0[0.711272776](OS=8, -0.526%) 
$ python train.py --logtostderr --training_number_of_steps=60000 --train_split="train" --model_variant="mobilenet_v2" --output_stride=16 --train_crop_size=769 --train_crop_size=769 --train_batch_size=16 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=True --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.001

----------------
# 2019.03.21
## mobilenet_v2, `re-use only the network backbone` 	
## atrous_rates, decoder_output_stride=4, train_batch_size=8, 
## fine_tune_batch_norm=False, base_learning_rate=0.001,
## With `ASPP` & `Decoder`.
## Total number of params: 2806979 = 2.807M, + 612016 params.
## global step 60000: loss = 0.2365 (0.624 sec/step), 
## OS=16, miou_1.0[0.713084042], class_0_iou[0.977024138], class_1_iou[0.819241762], class_2_iou[0.909438729], class_3_iou[0.47108227], class_4_iou[0.557567596], class_5_iou[0.526857793], class_6_iou[0.605363429], class_7_iou[0.704800606], class_8_iou[0.912902653], class_9_iou[0.604719758], class_10_iou[0.93508476], class_11_iou[0.759098768], class_12_iou[0.501966774], class_13_iou[0.931289], class_14_iou[0.662572622], class_15_iou[0.784451902], class_16_iou[0.65018934], class_17_iou[0.514163], class_18_iou[0.720781744]

## OS=8, miou_1.0[0.718111694], class_0_iou[0.976072133],class_1_iou[0.81612581],class_2_iou[0.910968959], class_3_iou[0.474383414], class_4_iou[0.555778503], class_5_iou[0.551861942], class_6_iou[0.616956949], class_7_iou[0.718308449], class_8_iou[0.91384697], class_9_iou[0.605608463], class_10_iou[0.936656296], class_11_iou[0.768783], class_12_iou[0.513123572], class_13_iou[0.932872117], class_14_iou[0.662885666], class_15_iou[0.784798861], class_16_iou[0.651456356], class_17_iou[0.526634455], class_18_iou[0.727000773]

## global step 90000: loss = 0.2403, miou_1.0[0.715924084], miou_1.0[0.721019745](OS=8)
$ python train.py --logtostderr --training_number_of_steps=60000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.001


-----------------
## Cityscapes dataset train class ID
[128, 64, 128],  # trainId 0: 'road'			[0.977024138]		+
[244, 35, 232],  # trainId 1: 'sidewalk'		[0.819241762]		+
[70, 70, 70],    # trainId 2: 'building'		[0.909438729]		
[102, 102, 156], # trainId 3: 'wall'			[0.47108227]			-
[190, 153, 153], # trainId 4: 'fence'			[0.557567596]			-
[153, 153, 153], # trainId 5: 'pole'			[0.526857793]			-
[250, 170, 30],  # trainId 6: 'traffic light'	[0.605363429]		+
[220, 220, 0],   # trainId 7: 'traffic sign'	[0.704800606]		+
[107, 142, 35],  # trainId 8: 'vegetation'		[0.912902653]		+
[152, 251, 152], # trainId 9: 'terrain'			[0.604719758]		
[70, 130, 180],  # trainId 10: 'sky'			[0.93508476]		
[220, 20, 60],   # trainId 11: 'person'			[0.759098768]		+
[255, 0, 0],     # trainId 12: 'rider'			[0.501966774]		+	-
[0, 0, 142],     # trainId 13: 'car'			[0.931289]			+
[0, 0, 70],      # trainId 14: 'truck'			[0.662572622]		+
[0, 60, 100],    # trainId 15: 'bus'			[0.784451902]		+
[0, 80, 100],    # trainId 16: 'train'			[0.651456356]
[0, 0, 230],     # trainId 17: 'motorcycle'		[0.514163]			
[119, 11, 32],   # trainId 18: 'bicycle'		[0.720781744]		+

------------------
# 2019.03.22
## mobilenet_v2, `re-use only the network backbone` 	[OK]
## train_batch_size=8, fine_tune_batch_norm=False, base_learning_rate=0.001,
## global step 60000: loss = 0.1840 (0.454 sec/step), miou_1.0[0.703004062], miou_1.0[0.71064](OS=8)
## global step 80100: loss = 0.1852 (0.482 sec/step), miou_1.0[0.699271679]
## global step 90000: loss = 0.1706 (0.452 sec/step), miou_1.0[0.702552319], miou_1.0[0.710787654](OS=8) 
$ python train.py --logtostderr --training_number_of_steps=60000 --train_split="train" --model_variant="mobilenet_v2" --output_stride=16 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.001



----------------
# 2019.03.23
## mobilenet_v2, `re-use only the network backbone` 	
## atrous_rates, decoder_output_stride=4, train_batch_size=8, 
## fine_tune_batch_norm=False, base_learning_rate=0.001,
## With `ASPP` & `Decoder`, `Multi loss`
## global step 10930: loss = 0.2160 (0.647 sec/step), miou_1.0[0.658394158]
## global step 30000: loss = 0.2130, miou_1.0[0.690611124], miou_1.0[0.698419273](OS=8)
## global step 39220: loss = 0.2180, miou_1.0[0.692829728], miou_1.0[0.699058115](OS=8)
## global step 90000: loss = 0.2137, miou_1.0[0.712212205], miou_1.0[0.716619968](OS=8) 
$ python train.py --logtostderr --training_number_of_steps=60000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.001



----------------
# 2019.03.24
## mobilenet_v2, `re-use only the network backbone` 	
## atrous_rates, decoder_output_stride=4, train_batch_size=8, 
## fine_tune_batch_norm=False, base_learning_rate=0.001,
## With `ASPP` & `Decoder`, `Updated Multi loss`
## global step 53430: loss = 0.2190 (0.702 sec/step), miou_1.0[0.70634681], miou_1.0[0.708874226](OS=8)
## global step 57660: loss = 0.2031, miou_1.0[0.710215807], miou_1.0[0.712354183](OS=8)
## --base_learning_rate=0.0001
## global step 60000: loss = 0.2219, miou_1.0[0.710162222], miou_1.0[0.712677181](OS=8)
## global step 90000: loss = 0.2795, miou_1.0[0.711302519], miou_1.0[0.714012325](OS=8)
$ python train.py --logtostderr --training_number_of_steps=90000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.001


----------------
# 2019.03.25
## mobilenet_v2, `re-use only the network backbone` 	
## atrous_rates, decoder_output_stride=4, train_batch_size=8, 
## fine_tune_batch_norm=False, base_learning_rate=0.001,
## With `dense_prediction_cell` & `Decoder`
## global step 10000: loss = 0.1748, miou_1.0[0.617310345]
## global step 30000: loss = 0.1748, miou_1.0[0.669960856] 
## global step 60000, miou_1.0[0.680237591]
$ python train.py --logtostderr --training_number_of_steps=60000 --train_split="train" --model_variant="mobilenet_v2" --dense_prediction_cell_json="./core/dense_prediction_cell_branch5_top1_cityscapes.json" --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.001

## eval
$ python eval.py --logtostderr --eval_split="val" --model_variant="mobilenet_v2" --dense_prediction_cell_json="./core/dense_prediction_cell_branch5_top1_cityscapes.json" --decoder_output_stride=4 --eval_crop_size=1025 --eval_crop_size=2049 --dataset="cityscapes" --checkpoint_dir=./datasets/cityscapes/exp/train_on_train_set/train --eval_logdir=./datasets/cityscapes/exp/train_on_train_set/eval --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1


----------------
# 2019.03.25
## mobilenet_v2, `re-use only the network backbone`		[OK] !!!!!!!!!!!
## train_batch_size=8, fine_tune_batch_norm=True, base_learning_rate=0.01
## With `ASPP` & `Decoder`. 
## global step 21100: loss = 0.1626 (0.614 sec/step), miou_1.0[0.689448178], miou_1.0[0.694613576](OS=8)
## global step 30000: loss = 0.1628, miou_1.0[0.726237476], miou_1.0[0.725398242](OS=8)
## global step 60000: loss = 0.1858, miou_1.0[0.730198562], miou_1.0[0.729129791](OS=8)
## 
$ python train.py --logtostderr --training_number_of_steps=60000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=True --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.01


----------------
# 2019.03.25
## mobilenet_v2, `re-use only the network backbone`		
## train_batch_size=8, fine_tune_batch_norm=True, base_learning_rate=0.01
## With `ASPP` & `Decoder`. `sigmoid loss`
## global step 3240: loss = 0.1294 (0.603 sec/step), miou_1.0[0.293790758]
## global step 26390: loss = 0.0978, miou_1.0[0.549788833]
## global step 30000: loss = 0.0924, miou_1.0[0.550878525]
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train_test --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=True --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.01


----------------
# 2019.03.26
## mobilenet_v2, `re-use only the network backbone`		
## train_batch_size=8, fine_tune_batch_norm=False, base_learning_rate=0.001
## With `ASPP` & `Decoder`. `sigmoid loss`
## global step 11210: loss = 0.1100 (0.758 sec/step), miou_1.0[0.31730026]
## 
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train_test --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.001


----------------
# 2019.03.27
## mobilenet_v2, `re-use only the network backbone`		[OK] !!!!!!!!!!!
## train_batch_size=8, fine_tune_batch_norm=True, base_learning_rate=0.01
## With `ASPP` & `Decoder` &　`self-attention v2` 
## Total number of params: 3316547 = 3.316M, + 1121584 params.
## global step 5830, miou_1.0[0.60104394], 
## global step 38980: loss = 0.2598 (0.633 sec/step), miou_1.0[0.718122]
## global step 60000: loss = 0.2327 (0.598 sec/step), miou_1.0[0.734421551]
## global step 90000: loss = 0.2347, miou_1.0[0.737797499]
$ python train.py --logtostderr --training_number_of_steps=50000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train_self_attention --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=True --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.01 --use_self_attention=True

## eval `self-attention`
$ python eval.py --logtostderr --eval_split="val" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=1025 --eval_crop_size=2049 --dataset="cityscapes" --checkpoint_dir=./datasets/cityscapes/exp/train_on_train_set/train_self_attention --eval_logdir=./datasets/cityscapes/exp/train_on_train_set/eval --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1 --use_self_attention=True

## vis `self-attention`
$ python vis.py --logtostderr --vis_split="val" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=1025 --vis_crop_size=2049 --dataset="cityscapes" --colormap_type="cityscapes" --checkpoint_dir=./datasets/cityscapes/exp/train_on_train_set/train_self_attention --vis_logdir=./datasets/cityscapes/exp/train_on_train_set/vis --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1 --use_self_attention=True

## frozen graph
$ python export_model.py --logtostderr --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --crop_size=1025 --crop_size=2049 --checkpoint_path=./datasets/cityscapes/exp/train_on_train_set/0328-backbone-aspp-decoder-bn-self_attentionv3/model.ckpt-90000 --export_path=./datasets/cityscapes/frozen_graph.pb --num_classes=19 --use_self_attention=True

----------------
# 2019.03.28
## mobilenet_v2, `re-use only the network backbone`		[OK] !!!!!!!!!
## train_batch_size=8, fine_tune_batch_norm=True, base_learning_rate=0.01
## With `ASPP` & `Decoder` &　`self-attention v3` 
## Total number of params: 3316547 = 3.316M, + 1121584 params. 
## global step 20890: loss = 0.2968 (0.600 sec/step), miou_1.0[0.686801]
## global step 90000: loss = 0.1979, miou_1.0[0.752497077]
## class_0_iou[0.981986761], class_1_iou[0.85065], class_2_iou[0.91842705], class_3_iou[0.538971126], class_4_iou[0.565963566], class_5_iou[0.601120472], class_6_iou[0.65283227], class_7_iou[0.743324935],class_8_iou[0.920270562], class_9_iou[0.638566077], class_10_iou[0.944234], class_11_iou[0.787149727], class_12_iou[0.550466299], class_13_iou[0.943454146], class_14_iou[0.743885], class_15_iou[0.808109], class_16_iou[0.75095], class_17_iou[0.612785161], class_18_iou[0.7442981]     
$ python train.py --logtostderr --training_number_of_steps=90000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train_self_attention --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=True --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.01 --use_self_attention=True 



----------------
# 2019.03.29
## mobilenet_v2, `re-use only the network backbone`		[NOT Better]
## train_batch_size=8, fine_tune_batch_norm=True, base_learning_rate=0.01
## With `ASPP` & `Decoder` &　`self-attention v3` & better decoder
## global step 90000: loss = 0.1917 (0.659 sec/step), miou_1.0[0.734410882]
## class_0_iou[0.981430709], class_1_iou[0.843066216], class_2_iou[0.91083771], class_3_iou[0.524650693], class_4_iou[0.545776486], class_5_iou[0.596264482], class_6_iou[0.594103634], class_7_iou[0.717876732], class_8_iou[0.919632912], class_9_iou[0.610619128], class_10_iou[0.945383906], class_11_iou[0.780628443], class_12_iou[0.464673], class_13_iou[0.940704107], class_14_iou[0.782208502], class_15_iou[0.813224137], class_16_iou[0.662017465], class_17_iou[0.588823795], class_18_iou[0.731884599] 
$ python train.py --logtostderr --training_number_of_steps=90000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train_self_attention --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=True --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.01 --use_self_attention=True


----------------
# 2019.03.30
## mobilenet_v2, `re-use only the network backbone`		
## train_batch_size=8, fine_tune_batch_norm=True, base_learning_rate=0.01
## With `ASPP` & `Decoder` &　`self-attention v3` & `multi-loss`
## 
$ python train.py --logtostderr --training_number_of_steps=90000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train_multi_loss --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=True --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.01 --use_self_attention=True 


----------------
# 2019.03.30
## mobilenet_v2, `re-use only the network backbone`		
## train_batch_size=8, fine_tune_batch_norm=True, base_learning_rate=0.01
## With `ASPP` & `Decoder` &　`self-attention v3` & `fine_coarse`
## 200 epochs, 22973*200/8 = 574325

$ python train.py --logtostderr --training_number_of_steps=500000 --train_split="train_extra" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train_self_attention --dataset_dir=/media/jun/ubuntu/datasets/CityScapes/tfrecord --num_clones=2 --fine_tune_batch_norm=True --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.01 --use_self_attention=True

```


-----------------
### 6. CamVid dataset
### Recommended Directory Structure for Training and Evaluation:
```shell
+ datasets
  + camvid
    + tfrecord
    + exp
      + train_on_train_set
        + train
        + eval
        + vis


### 6.1 Prepare dataset and convert to TFRecord
```shell
# From tensorflow/models/research/deeplab/datasets
$ ./convert_camvid.sh
# This shell script run 'build_camvid_data.py'

# 修改训练脚本
# segmentation_dataset.py line 110
_CAMVID_INFORMATION = DatasetDescriptor( 
    splits_to_sizes={ 
      'train': 367,   # num of samples in images/training 
      'val': 101,   # num of samples in images/validation 
    }, 
    num_classes=12, 
    ignore_label=255, 
    )
# _DATASETS_INFORMATION
_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'camvid': _CAMVID_INFORMATION,  #camvid示例
}
# ==TODO==
# 修改Colormap
```

------------------
### 6.2 A local `evaluation` job using `xception_65` can be run with the following command:
```shell
# run eval, From tensorflow/models/research/deeplab
$ python eval.py --logtostderr --eval_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=361 --eval_crop_size=481 --dataset="camvid" --checkpoint_dir=./datasets/camvid/exp/train_on_train_set/train --eval_logdir=./datasets/camvid/exp/train_on_train_set/eval --dataset_dir=./datasets/camvid/tfrecord --max_number_of_iterations=1

```

---------------------
### 6.3 A local `visualization` job using `xception_65` can be run with the following command:
```shell
# run vis, From 'deeplabv3_cityscapes_train'
$ python vis.py --logtostderr --vis_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=361 --vis_crop_size=481 --dataset="camvid" --colormap_type="cityscapes" --checkpoint_dir=./datasets/camvid/exp/train_on_train_set/train --vis_logdir=./datasets/camvid/exp/train_on_train_set/vis --dataset_dir=./datasets/camvid/tfrecord --max_number_of_iterations=1

## Get output file: 'datasets/camvid/exp/train_on_train_set/vis/segmentation_results'
```

---------------------
### 6.4 A local `training` job using `xception_65` can be run with the following command::

```shell
# run training, From tensorflow/models/research/deeplab

## Fine-tuning From 'deeplabv3_cityscapes_train', `re-use ALL the trained weights EXCEPT the logits`
## step 3050: loss = 0.6069 (0.192 sec/step), miou_1.0[0.0278994814], train_batch_size=1
$ python train.py --logtostderr --training_number_of_steps=90000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=361 --train_crop_size=361 --train_batch_size=1 --dataset="camvid" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_cityscapes_train/model.ckpt --train_logdir=./datasets/camvid/exp/train_on_train_set/train --dataset_dir=./datasets/camvid/tfrecord --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=True

## Fine-tuning From 'deeplabv3_cityscapes_train', `re-use ALL the trained weights EXCEPT the logits` [OK]
## global step 5290: loss = 0.5241 (0.518 sec/step), miou_1.0[0.638597667]
## global step 30000: loss = 0.5131 (0.506 sec/step), miou_1.0[0.667017698]
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=361 --train_crop_size=361 --train_batch_size=10 --dataset="camvid" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_cityscapes_train/model.ckpt --train_logdir=./datasets/camvid/exp/train_on_train_set/train --dataset_dir=./datasets/camvid/tfrecord --fine_tune_batch_norm=False --num_clones=2 --initialize_last_layer=False --last_layers_contain_logits_only=True

## Fine-tuning From 'deeplabv3_cityscapes_train', `re-use only the network backbone` [OK]
## global step 8960: loss = 0.6623 (0.619 sec/step), miou_1.0[0.497921795]
## global step 13710: loss = 0.6310 (0.572 sec/step), miou_1.0[0.537117]
## global step 30000: loss = 0.5538 (0.643 sec/step), miou_1.0[0.57864368]
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=361 --train_crop_size=361 --train_batch_size=12 --dataset="camvid" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_cityscapes_train/model.ckpt --train_logdir=./datasets/camvid/exp/train_on_train_set/train --dataset_dir=./datasets/camvid/tfrecord --fine_tune_batch_norm=False --num_clones=2 --initialize_last_layer=False --last_layers_contain_logits_only=False

``` 


-----------
## 7. kitti road dataset
### Recommended Directory Structure for Training and Evaluation:
```shell
+ datasets
  + pascal_voc_seg
    + tfrecord						# convert from voc2012
    + exp	
      + train_on_train_set			# train_on_train_set stores the train/eval/vis events and results
        + train
        + eval
        + vis
```

### 7.1 Uncompress VOC2012 dataset.
```shell
# cd to deeplab
$ cd /home/jun/Documents/king/models/research/deeplab/datasets
# Removes the color map from the ground truth segmentation annotations and save the results to output_dir.
$ python remove_gt_colormap_kitti.py --original_gt_folder=/media/jun/ubuntu/datasets/Kitti/Kitti_road/data_road/training/gt_image_2/ --output_dir=/media/jun/ubuntu/datasets/Kitti/Kitti_road/data_road/training/gt_image_raw
# mkdir 
$ mkdir -p kitti_road/tfrecord
# convert
$ python build_kitti_data.py --image_folder=/media/jun/ubuntu/datasets/Kitti/Kitti_road/data_road/ --semantic_segmentation_folder=/media/jun/ubuntu/datasets/Kitti/Kitti_road/data_road/ --list_folder=/media/jun/ubuntu/datasets/Kitti/Kitti_road/data_road/ --image_format=png --output_dir=./kitti_road/tfrecord/
## 各参数意义如下：
    `image_folder`： 保存images的路径
    `semantic_segmentation_folder`： 保存labels的路径
    `list_folder`： 保存train\val.txt文件的路径
    `image_format`： image的格式
    `output_dir`： 生成tfrecord格式的数据所要保存的位置

## Terminal print
# train
>> Converting image 121/241 shard 0
>> Converting image 241/241 shard 1
# trainval
>> Converting image 145/289 shard 0
>> Converting image 289/289 shard 1
# val
>> Converting image 24/48 shard 0
>> Converting image 48/48 shard 1

# 修改训练脚本
# segmentation_dataset.py line 110

# king@2019.03.28
_KITTIROAD_INFORMATION = DatasetDescriptor( 
    splits_to_sizes={ 
      'train': 241,     # num of samples in images/training 
      'trainval': 289,  # num of trainval 
      'val': 48,        # num of samples in images/validation 
    }, 
    num_classes=2, 
    ignore_label=255, 
    )


_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'camvid': _CAMVID_INFORMATION,  #camvid示例
    'kitti_road': _KITTIROAD_INFORMATION,
}

# 修改Colormap
`deeplab/utils/get_dataset_colormap.py`
1. _KITTIROAD = 'kitti_road'
2. def create_label_colormap(dataset=_PASCAL):
3. def create_kittiroad_label_colormap():
4. For vis:
# king@2019.03.28
_KITTIROAD = 'kitti_road'

# Max number of entries in the colormap for each dataset.
_DATASET_MAX_ENTRIES = {
    _ADE20K: 151,
    _CITYSCAPES: 19,
    _MAPILLARY_VISTAS: 66,
    _PASCAL: 256,
    _KITTIROAD: 2,
}

## 解决数据(backgroud & road)不平衡问题
# Handle data balance problem.
irgore_weight = 0 
bk_weight = 1 #background 
road_weight = 10 #object1 
not_ignore_mask = tf.to_float(tf.equal(scaled_labels, 0)) * bk_weight + \
                  tf.to_float(tf.equal(scaled_labels, 1)) * road_weight + \
                  tf.to_float(tf.equal(scaled_labels, ignore_label)) * irgore_weight 



```

---------------------
### 7.4 A local `training` job using `mobilenetv2` can be run with the following command::

```shell

----------------
# run eval mobilenet_v2
$ python eval.py --logtostderr --eval_split="val" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=376 --eval_crop_size=1243 --dataset="kitti_road" --checkpoint_dir=./datasets/kitti_road/exp/train_on_train_set/train_kitti --eval_logdir=./datasets/kitti_road/exp/train_on_train_set/eval --dataset_dir=./datasets/kitti_road/tfrecord --max_number_of_iterations=1 --use_self_attention=True

# run vis, mobilenet_v2
## add `colormap_type', 'pascal', ['pascal', 'cityscapes', 'kitti_road']`
$ python vis.py --logtostderr --vis_split="val" --model_variant="mobilenet_v2"  --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16  --decoder_output_stride=4 --vis_crop_size=376 --vis_crop_size=1243 --dataset="kitti_road" --colormap_type="kitti_road" --checkpoint_dir=./datasets/kitti_road/exp/train_on_train_set/train_kitti --vis_logdir=./datasets/kitti_road/exp/train_on_train_set/vis --dataset_dir=./datasets/kitti_road/tfrecord --max_number_of_iterations=1 --use_self_attention=True

# frozen graph
$ python export_model.py --logtostderr --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --crop_size=376 --crop_size=1243 --checkpoint_path=./datasets/kitti_road/exp/train_on_train_set/train_kitti/model.ckpt-12000 --export_path=./datasets/kitti_road/frozen_graph.pb --num_classes=2 --use_self_attention=True


# run training, From tensorflow/models/research/deeplab
----------------
# 2019.03.28
## 'deeplabv3_mnv2_cityscapes_train', `re-use ALL the trained weights EXCEPT the logits` [OK]
## global step 10000: loss = 0.1312 (0.517 sec/step), miou_1.0[0.931432247]
$ python train.py --logtostderr --training_number_of_steps=10000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=376 --train_crop_size=1243 --train_batch_size=8 --dataset="kitti_road" --tf_initial_checkpoint=./datasets/cityscapes/exp/train_on_train_set/0327-backbone-aspp-decoder-bn-self_attention/model.ckpt-90000 --train_logdir=./datasets/kitti_road/exp/train_on_train_set/train_kitti --dataset_dir=./datasets/kitti_road/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True

----------------
# 2019.03.29
## 'deeplabv3_mnv2_cityscapes_train', `re-use only the network backbone` 
## atrous_rates & train_batch_size=8 & BN=True & --base_learning_rate=0.001
## ASPP & Decoder & self-attention v3
## Total number of params: 3321490 = 3.321M
## ('GFLOPs after freezing: ', 2.2490016435)
## miou_1.0[0.954717636], class_0_iou[0.983766], class_1_iou[0.925669253]
$ python train.py --logtostderr --training_number_of_steps=12000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=376 --train_crop_size=1243 --train_batch_size=8 --dataset="kitti_road" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/kitti_road/exp/train_on_train_set/train_kitti --dataset_dir=./datasets/kitti_road/tfrecord --num_clones=2 --fine_tune_batch_norm=True --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.001 --use_self_attention=True



----------------
# 2019.03.29
## 'deeplabv3_mnv2_cityscapes_train', `re-use only the network backbone` 
## atrous_rates & train_batch_size=8 & BN=True & --base_learning_rate=0.001
## ASPP & Decoder & self-attention v3 & label balance
## miou_1.0[0.960260928], class_0_iou[0.985344708], class_1_iou[0.935177147]
$ python train.py --logtostderr --training_number_of_steps=12000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=376 --train_crop_size=1243 --train_batch_size=8 --dataset="kitti_road" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/kitti_road/exp/train_on_train_set/train_kitti --dataset_dir=./datasets/kitti_road/tfrecord --num_clones=2 --fine_tune_batch_norm=True --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.001 --use_self_attention=True

``` 

--------------------------------------

## 8. runway dataset
### Recommended Directory Structure for Training and Evaluation:

### 8.1 Prepare dataset.
```shell
# cd to deeplab
$ cd /home/jun/Documents/king/models/research/deeplab/datasets
# Removes the color map from the ground truth segmentation annotations and save the results to output_dir.
$ python remove_gt_colormap_kitti.py --original_gt_folder=/media/jun/ubuntu/datasets/Kitti/Kitti_road/data_road/training/gt_image_2/ --output_dir=/media/jun/ubuntu/datasets/Kitti/Kitti_road/data_road/training/gt_image_raw
# prepare list
$ python prepare_train_file.py
# mkdir 
$ mkdir -p runway/tfrecord
# convert
$ python build_kitti_data.py --image_folder=/media/jun/ubuntu/datasets/airport_runway/runway/ --semantic_segmentation_folder=/media/jun/ubuntu/datasets/airport_runway/runway/ --list_folder=/media/jun/ubuntu/datasets/airport_runway/runway/ --image_format=png --output_dir=./runway/tfrecord/

## 各参数意义如下：
    `image_folder`： 保存images的路径
    `semantic_segmentation_folder`： 保存labels的路径
    `list_folder`： 保存train\val.txt文件的路径
    `image_format`： image的格式
    `output_dir`： 生成tfrecord格式的数据所要保存的位置

# 修改训练脚本
# segmentation_dataset.py line 110

# king@2019.03.28
_KITTIROAD_INFORMATION = DatasetDescriptor( 
    splits_to_sizes={ 
      'train': 241,     # num of samples in images/training 
      'trainval': 289,  # num of trainval 
      'val': 48,        # num of samples in images/validation 
    }, 
    num_classes=2, 
    ignore_label=255, 
    )


_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'camvid': _CAMVID_INFORMATION,  #camvid示例
    'kitti_road': _KITTIROAD_INFORMATION,
}

# 修改Colormap
`deeplab/utils/get_dataset_colormap.py`
1. _KITTIROAD = 'kitti_road'
2. def create_label_colormap(dataset=_PASCAL):
3. def create_kittiroad_label_colormap():
4. For vis:
# king@2019.03.28
_KITTIROAD = 'kitti_road'

# Max number of entries in the colormap for each dataset.
_DATASET_MAX_ENTRIES = {
    _ADE20K: 151,
    _CITYSCAPES: 19,
    _MAPILLARY_VISTAS: 66,
    _PASCAL: 256,
    _KITTIROAD: 2,
}

## 解决数据(backgroud & road)不平衡问题
# Handle data balance problem.
irgore_weight = 0 
bk_weight = 1 #background 
road_weight = 10 #object1 
not_ignore_mask = tf.to_float(tf.equal(scaled_labels, 0)) * bk_weight + \
                  tf.to_float(tf.equal(scaled_labels, 1)) * road_weight + \
                  tf.to_float(tf.equal(scaled_labels, ignore_label)) * irgore_weight 



```

---------------------
### 8.4 A local `training` job using `mobilenetv2` can be run with the following command::

```shell

----------------
# run eval mobilenet_v2
$ python eval.py --logtostderr --eval_split="val" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=376 --eval_crop_size=1243 --dataset="runway" --checkpoint_dir=./datasets/runway/exp/train_on_train_set/train --eval_logdir=./datasets/runway/exp/train_on_train_set/eval --dataset_dir=./datasets/runway/tfrecord --max_number_of_iterations=1 --use_self_attention=True

# run vis, mobilenet_v2
## add `colormap_type', 'pascal', ['pascal', 'cityscapes', 'kitti_road']`
$ python vis.py --logtostderr --vis_split="val" --model_variant="mobilenet_v2"  --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16  --decoder_output_stride=4 --vis_crop_size=376 --vis_crop_size=1243 --dataset="runway" --colormap_type="kitti_road" --checkpoint_dir=./datasets/runway/exp/train_on_train_set/train --vis_logdir=./datasets/runway/exp/train_on_train_set/vis --dataset_dir=./datasets/runway/tfrecord --max_number_of_iterations=1 --use_self_attention=True

# frozen graph
$ python export_model.py --logtostderr --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --crop_size=376 --crop_size=1243 --checkpoint_path=./datasets/runway/exp/train_on_train_set/train/model.ckpt-12000 --export_path=./datasets/runway/frozen_graph.pb --num_classes=2 --use_self_attention=True


# run training, From tensorflow/models/research/deeplab

----------------
# 2019.03.29
## 'deeplabv3_mnv2_cityscapes_train', `re-use only the network backbone` 
## atrous_rates & train_batch_size=8 & BN=True & --base_learning_rate=0.001
## ASPP & Decoder & self-attention v3 & label balance
## miou_1.0[0.976282358], class_0_iou[0.987392783], class_1_iou[0.965171933]
$ python train.py --logtostderr --training_number_of_steps=12000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=376 --train_crop_size=1243 --train_batch_size=8 --dataset="runway" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/runway/exp/train_on_train_set/train --dataset_dir=./datasets/runway/tfrecord --num_clones=2 --fine_tune_batch_norm=True --initialize_last_layer=False --last_layers_contain_logits_only=False --save_summaries_images=True --base_learning_rate=0.001 --use_self_attention=True

``` 



--------------------------
## DeepLabv3+ Code Reading

- [x] **deeplab_demo.ipynb**
* 修改不用每次下载模型，从'ubuntu'盘读取模型。
* `model_variant`
```python
_PREPROCESS_FN = {
    'mobilenet_v2': _preprocess_zero_mean_unit_range,
    'resnet_v1_50': _preprocess_subtract_imagenet_mean,
    'resnet_v1_50_beta': _preprocess_zero_mean_unit_range,
    'resnet_v1_101': _preprocess_subtract_imagenet_mean,
    'resnet_v1_101_beta': _preprocess_zero_mean_unit_range,
    'xception_41': _preprocess_zero_mean_unit_range,
    'xception_65': _preprocess_zero_mean_unit_range,
    'xception_71': _preprocess_zero_mean_unit_range,
}
```

--------------------------
# 2019.03.10
## DeepLabv3+ Code Reading
- [x] [TF.slim简单用法](https://www.jianshu.com/p/18747374ec28)
* slim这个模块是在16年新推出的，其主要目的是来做所谓的“代码瘦身”。
* 撇开`Keras，TensorLayer，tfLearn`这些个`高级库`不谈，光用tensorflow能不能写出简洁的代码？当然行，`有slim就够了`！
* slim被放在tensorflow.contrib这个库下面，导入的方法如下：
> import tensorflow.contrib.slim as slim

* `slim`是一个使构建，训练，评估神经网络变得简单的库。它可以消除原生tensorflow里面很多重复的模板性的代码，让代码更紧凑，更具备可读性。另外slim提供了很多计算机视觉方面的著名模型（VGG, AlexNet等），我们不仅可以直接使用，甚至能以各种方式进行扩展。


---------------------------
# 2019.03.10
### slim的子模块及功能介绍：
* `arg_scope`: provides a new scope named arg_scope that allows a user to define default arguments for specific operations within that scope.
除了基本的`namescope，variabelscope`外，又加了`argscope`，它是用来控制每一层的默认超参数的。

* `data`: contains TF-slim's dataset definition, data providers, parallel_reader, and decoding utilities.

* `evaluation`: contains routines for evaluating models.

* `layers`: contains high level layers for building models using tensorflow.
这个比较重要，slim的核心和精髓，一些复杂层的定义

* `learning`: contains routines for training models.

* `losses`: contains commonly used loss functions.

* `metrics`: contains popular evaluation metrics.
评估模型的度量标准

* `nets`: contains popular network definitions such as VGG and AlexNet models.
包含一些经典网络，VGG等，用的也比较多

* `queues`: provides a context manager for easily and safely starting and closing QueueRunners.
文本队列管理，比较有用。

* `regularizers`: contains weight regularizers.
包含一些正则规则

* `variables`: provides convenience wrappers for variable creation and manipulation.
这个比较有用，我很喜欢slim管理变量的机制

* slim中实现一个层：
> net = slim.conv2d(input, 128, [3, 3], scope='conv1_1')

### slim中的`repeat`操作：
```python
## 定义三个相同的卷积层
net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')
# slim中的repeat操作可以减少代码量
net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')
```

### `stack`是处理卷积核或者输出不一样的情况：
```python
# Verbose way:
x = slim.fully_connected(x, 32, scope='fc/fc_1')
x = slim.fully_connected(x, 64, scope='fc/fc_2')
x = slim.fully_connected(x, 128, scope='fc/fc_3')
# 使用stack操作：
slim.stack(x, slim.fully_connected, [32, 64, 128], scope='fc')
# 卷积层
# 普通方法:
x = slim.conv2d(x, 32, [3, 3], scope='core/core_1')
x = slim.conv2d(x, 32, [1, 1], scope='core/core_2')
x = slim.conv2d(x, 64, [3, 3], scope='core/core_3')
x = slim.conv2d(x, 64, [1, 1], scope='core/core_4')
# 简便方法:
slim.stack(x, slim.conv2d, [(32, [3, 3]), (32, [1, 1]), (64, [3, 3]), (64, [1, 1])], scope='core')
```

### slim中的argscope
```python
with slim.arg_scope([slim.conv2d], padding='SAME',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.conv2d(inputs, 64, [11, 11], scope='conv1')
	# 若想特别指定某些层的参数，可以重新赋值（相当于重写）
    net = slim.conv2d(net, 128, [11, 11], padding='VALID', scope='conv2')
    net = slim.conv2d(net, 256, [11, 11], scope='conv3')
```

### 定义一个VGG网络
```python
def vgg16(inputs):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    net = slim.fully_connected(net, 4096, scope='fc6')
    net = slim.dropout(net, 0.5, scope='dropout6')
    net = slim.fully_connected(net, 4096, scope='fc7')
    net = slim.dropout(net, 0.5, scope='dropout7')
    net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
  return net
```

### 训练模型
```python
import tensorflow as tf
vgg = tf.contrib.slim.nets.vgg
 
# Load the images and labels.
images, labels = ...
 
# Create the model.
predictions, _ = vgg.vgg_16(images)
 
# Define the loss functions and get the total loss.
loss = slim.losses.softmax_cross_entropy(predictions, labels)
```

### 定义自己的loss的方法
```python
# Load the images and labels.
images, scene_labels, depth_labels, pose_labels = ...
 
# Create the model.
scene_predictions, depth_predictions, pose_predictions = CreateMultiTaskModel(images)
 
# Define the loss functions and get the total loss.
classification_loss = slim.losses.softmax_cross_entropy(scene_predictions, scene_labels)
sum_of_squares_loss = slim.losses.sum_of_squares(depth_predictions, depth_labels)
pose_loss = MyCustomLossFunction(pose_predictions, pose_labels)
slim.losses.add_loss(pose_loss) # Letting TF-Slim know about the additional loss.
 
# The following two ways to compute the total loss are equivalent:
regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
total_loss1 = classification_loss + sum_of_squares_loss + pose_loss + regularization_loss
 
# (Regularization Loss is included in the total loss by default).
total_loss2 = slim.losses.get_total_loss()
```

### 读取保存模型变量
```python
# Create some variables.
v1 = slim.variable(name="v1", ...)
v2 = slim.variable(name="nested/v2", ...)
...
 
# Get list of variables to restore (which contains only 'v2').
variables_to_restore = slim.get_variables_by_name("v2")
 
# Create the saver which will be used to restore the variables.
restorer = tf.train.Saver(variables_to_restore)
 
with tf.Session() as sess:
  # Restore variables from disk.
  restorer.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")

###########################################
# 加载到不同名字的变量中
def name_in_checkpoint(var):
  return 'vgg16/' + var.op.name
 
variables_to_restore = slim.get_model_variables()
variables_to_restore = {name_in_checkpoint(var):var for var in variables_to_restore}
restorer = tf.train.Saver(variables_to_restore)
 
with tf.Session() as sess:
  # Restore variables from disk.
  restorer.restore(sess, "/tmp/model.ckpt")
```


--------------------------
# 2019.03.11
## DeepLabv3+ Code Reading

- [x] [deeplabV3+源码分解学习](https://www.jianshu.com/p/d0cc35b3f100)
github上deeplabV3+的源码是基于tensorflow（slim）简化的代码，是一款非常值得学习的标准框架结构


--------------------------
# 2019.03.11
## DeepLabv3+ run on CamVid
* [TensorFlow实战：Chapter-9下(DeepLabv3+在自己的数据集训练)](https://blog.csdn.net/u011974639/article/details/80948990)

* [deeplab/faq](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/faq.md)

* aquariusjay关于训练参数的设置：

![](https://img-blog.csdn.net/20180707103243314?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTE5NzQ2Mzk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

* aquariusjay关于imblance的设置：

![](https://img-blog.csdn.net/20180707103252841?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTE5NzQ2Mzk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

* github-@zhaolewen的关于数据标签的回答:

![](https://img-blog.csdn.net/20180707103303371?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTE5NzQ2Mzk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
















# 2019.03.14
## is:issue deeplab train is:closed 

* `aquariusjay` says:
> ASPP should still work for `MobileNet-V2 backbone`. `We do not use it` because we target at faster inference speed instead of high performance when using MobileNet-V2.


* `aquariusjay` commented on May 26, 2018
> We use `batch size 8` with `crop size = 769x769`, and `output_stride = 16` on Cityscapes.
Training with `Batch norm is essential` to attain high performance. <br>

> To `get the 82.1% performance` (on test set), you need to further train the model on all the `fine + coarse` annotations.

* The issue here is that your training batch size (--train_batch_size=4) is too small. You need to have a larger batch size. 


* It seems I needed a much larger step length than the default. `1e-2` gives results closer to the published results, with batch size 15 and a smaller crop window size.

* I want to train Cityscape with a trained model but summurize some Classes. For that i changed the `trainid` of some classes.

* If I use the `export_model.py` without --decoder_output_stride=4 flag (as used in training) it always outputs 0.
solution: adding the `--decoder_output_stride=4 flag`

* Issues: Is it possible to get `94% mIOU` of pascal voc 'val' data?
Ans: You are training with `trainval` dataset which means you've already include the `val set` during `training`. That usually results in `overfitting` in val dataset with `mIoU over 90%`.


* `deeplabv3+` how to print `per_class_iou`?
1. [TensorFlow: How Can I get the total_cm in tf.contrib.metrics.streaming_mean_iou](https://stackoverflow.com/questions/40340728/tensorflow-how-can-i-get-the-total-cm-in-tf-contrib-metrics-streaming-mean-iou)

2. [Deeplab——How to evaluate each class of iou](https://blog.csdn.net/zsf442553199/article/details/82217717)


* I have trained deeplab on my custom dataset(`VOC like`). The `training works without any problem`.
```shell
python train.py
--logtostderr
--training_number_of_steps=30000
--train_split="train"
--model_variant="mobilenet_v2"
--atrous_rates=12
--atrous_rates=24
--atrous_rates=36
--output_stride=8
--decoder_output_stride=4
--train_crop_size=241
--train_crop_size=321
--train_batch_size=24
--dataset="${DATASET}"
--initialize_last_layer=false 
```

# Update 2019.03.20
## [deeplab/g3doc/cityscapes.md](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/cityscapes.md)
* Note that for `{train,eval,vis}.py`:

1. In order to `reproduce our results`, one needs to use large `batch size (> 8)`, and set `fine_tune_batch_norm = True`. Here, we simply use small batch size during training for the purpose of demonstration. If the users have `limited GPU memory` at hand, please fine-tune from our provided checkpoints whose batch norm parameters have been trained, and `use smaller learning rate` with `fine_tune_batch_norm = False`.

2. The users should change `atrous_rates` from [6, 12, 18] to [12, 24, 36] if setting `output_stride=8`.

3. The users could skip the flag, `decoder_output_stride`, if you do not want to use the decoder structure.

4. Change and add the following flags in order to use the provided `dense prediction cell`. Note we need to set `decoder_output_stride` if you want to use the provided checkpoints which include the decoder module.
```python
--model_variant="xception_71"
--dense_prediction_cell_json="deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json"
--decoder_output_stride=4
```


-----------------------------------
# 2019.03.16

## I see `checkerboard pattern` in segmap when training `mobilenet_v2`:

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/deeplabv3_checkerboard_pattern.jpg)


---------------------------------------
# 2019.03.18
## MobileNetV2
- [x] **DeepLab Code Reading**
## `deeplab_demo.ipynb`
Done!


---------------------------------------
# 2019.03.19
## `train.py`
`train.py/_build_deeplab()/multi_scale_logits()` -> `model.py/_get_logits()/extract_features()` -> `core/feature_extractor.py` 

* From the code 'model/research/deeplab/train.py#L101'
> When `fine_tune_batch_norm=True`, use `at least batch size larger than 12` (batch size more than 16 is better). Otherwise, one could use smaller batch size and set fine_tune_batch_norm=False. <br>

* From the code 'model/research/deeplab/common.py#L47'
```python
"""
When using 'mobilent_v2', we set `atrous_rates = decoder_output_stride = None`.
When using 'xception_65' or 'resnet_v1' model variants, we set
atrous_rates = [6, 12, 18] (output stride 16) and decoder_output_stride = 4.
"""
```

* From the code 'model/research/deeplab/utils/train_utils.py#L56'
'upsample_logits', True
> `Label` is not downsampled, and instead we `upsample logits`.

* Image and label preprocess:
In 'deeplab/input_preprocess.py/preprocess_image_and_label()'
```python
# image & label type.
processed_image = tf.cast(image, tf.float32)
label = tf.cast(label, tf.int32)
```

* Data augmentation
In 'deeplab/input_preprocess.py/preprocess_image_and_label()'
1. random_scale
[0.5, 2.0], step:0.25

2. Pad image with mean pixel value.
For 'mobilenet_v2', mean_pixel: [127.5, 127.5, 127.5] <br>
target_height: [crop_height, 2*origin_height]

3. Randomly crop the image and label.
For Cityscapes: crop_size: [769, 769]

4. Randomly left-right flip the image and label.
> _PROB_OF_FLIP = 0.5

* `Default setup` for 'mobilenet_v2':
> No `ASPP` and No `Decoder`.


----------------------------------------
# 2019.03.21
## Code Reading
- [x] **eval.py**
`eval.py/predict_labels()` -> `model.py`
* add `per-class IoU`.

* `logits`: decoder_output_stride (resize到) label.shape(resize_bilinear) -> `tf.metrics.mean_iou` with logits.
* `tf.argmax`
predictions[output] = tf.argmax(logits, 3)

----------------
- [x] **vis.py**
`vis.py/predict_labels()` -> `model.py` get `predictions` -> `vis.py/_process_batch()`

* set `also_save_raw_predictions` to raw predictions(evaluation id)


----------------------------------------
# 2019.03.21
- [x] add `sigmoid_cross_entropy` in `deeplab/utils/train_utils.py`

In `train.py#L239`
> train_utils.add_softmax_cross_entropy_loss_for_each_scale( 


--------------------
- [x] **add `multi_loss` seg map**

* `deeplab/utils/train_utils.py/add_softmax_cross_entropy_for_auxiliary_loss()`

* `deeplab/model.py#L523`
loss_weight=0.4

* `deeplab/train.py`
loss_weight=0.6



----------------------------------------
# 2019.03.26
- [x] **self-attention**

* add `attention.py`

* modify `common.py`
```python
# king@2019.03.26
flags.DEFINE_boolean('use_self_attention', False,
                     'Use self attention module or not')

```

* modify `model.py#L603`:
```python
# king@2019.03.26
  if model_options.use_self_attention:    # Use self attention module.
    features = attention.self_attention_module(features,
                                  model_options.outputs_to_num_classes['semantic'],   # 19 classes.
                                  weight_decay = weight_decay,
                                  is_training = is_training,
                                  reuse = reuse,
                                  fine_tune_batch_norm = fine_tune_batch_norm)
```


-------------------------------------------
# 2019.03.28
- [x] **better decoder**

* `tf.reshape`应用场合比较广泛，当我们需要创建新的tensor或者动态地改变原有tensor的shape的时候可以使用; 而当我们只是想更新图中某个tensor的shape或者补充某个tensor的shape信息可以使用`tf.set_shape`来进行更新。
* short cut ``

* `deeplab/core/feature_extractor.py`
```python
'mobilenet_v2': {
        DECODER_END_POINTS: ['layer_7/depthwise_output',    # OS=8
                             'layer_4/depthwise_output'     # OS=4                        
        ],     
    },
```

## tf.image.resize_images
```python
tf.image.resize_images(
    images,
    size,
    method=ResizeMethod.BILINEAR,		# ResizeMethod.NEAREST_NEIGHBOR
    align_corners=False,
    preserve_aspect_ratio=False
)

# 
tf.image.resize_nearest_neighbor(
    images,
    size,
    align_corners=False,
    name=None
)
# Resize images to size using nearest neighbor interpolation.
decode = tf.image.resize_nearest_neighbor(d_layer1, tf.shape(inputs)[1:3])
decode = tf.layers.conv2d(
	decode,
	inputs.shape[-1],
	kernel_size=(3, 3),
	strides=(1, 1),
	activation=tf.nn.tanh,
	padding="SAME",
	name="decode")

```
* tf.image.resize_images() 

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/tf.image.resize_images.png)


## Modify
1. `model.py#L741`
```python
# model.py
# Resize to [decoder_height, decoder_width].
upsample_height = scale_dimension(decoder_height, 1.0/(2-i))  # OS=8, 4
upsample_width = scale_dimension(decoder_width, 1.0/(2-i))    # OS=8, 4
for j, feature in enumerate(decoder_features_list):
  decoder_features_list[j] = tf.image.resize_bilinear(
      feature, [upsample_height, upsample_width], align_corners=True)

# deeplab/core/feature_extractor.py
'mobilenet_v2': {
        DECODER_END_POINTS: ['layer_7/depthwise_output',    # OS=8
                             'layer_4/depthwise_output'     # OS=4                        
        ],     
    },
```
2. `tf.image.resize_bilinear` -> `tf.image.resize_nearest_neighbor`












