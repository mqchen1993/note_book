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

![](https://github.com/kinglintianxia/note_book/blob/master/imgs/Modified Aligned Xception.png)

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
$ mkdir pascal_voc_seg/tfrecord
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

# run training, `re-use all the trained weights except the logits` 
# step 19760: loss = 0.1650 (0.924 sec/step), miou_1.0[0.904196858]
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=8 --dataset="pascal_voc_seg" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_pascal_trainval/model.ckpt --train_logdir=./datasets/pascal_voc_seg/exp/train_on_train_set/train --dataset_dir=./datasets/pascal_voc_seg/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=True

# run training, `re-use only the network backbone`
# global step 20260: loss = 0.2941 (0.934 sec/step), miou_1.0[0.818994045]
# global step 30000: loss = 0.2861 (0.878 sec/step), miou_1.0[0.826865256]
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=8 --dataset="pascal_voc_seg" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_pascal_trainval/model.ckpt --train_logdir=./datasets/pascal_voc_seg/exp/train_on_train_set/train --dataset_dir=./datasets/pascal_voc_seg/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=False

## Get output file: 'datasets/pascal_voc_seg/exp/train_on_train_set/train'
```


## aquariusjay says:
> ASPP should still work for `MobileNet-V2 backbone`.
We do not use it because we target at faster inference speed instead of high performance when using MobileNet-V2.

## Get output file: 'datasets/cityscapes/exp/train_on_train_set/train'

```

## `aquariusjay` commented on May 26, 2018
> We use `batch size 8` with `crop size = 769x769`, and `output_stride = 16` on Cityscapes.
Training with Batch norm is essential to attain high performance.

> To `get the 82.1% performance` (on test set), you need to further train the model on all the `fine + coarse` annotations.



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

# run eval, From tensorflow/models/research/deeplab
$ python eval.py --logtostderr --eval_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=1025 --eval_crop_size=2049 --dataset="cityscapes" --checkpoint_dir=./datasets/cityscapes/exp/train_on_train_set/train --eval_logdir=./datasets/cityscapes/exp/train_on_train_set/eval --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1
## Get 'Waiting for new checkpoint at...' 
## Terminal print
INFO:tensorflow:Finished evaluation at 2019-03-08-07:49:00
miou_1.0[0.935834229]
## Get output file: 'datasets/cityscapes/exp/train_on_train_set/eval/events.out.tfevents.1552031267.jun-pc'
$ tensorboard --logdir ./

## run eval, mobilenet_v2
$ python eval.py --logtostderr --eval_split="val" --model_variant="mobilenet_v2" --output_stride=8 --eval_crop_size=1025 --eval_crop_size=2049 --dataset="cityscapes" --checkpoint_dir=./datasets/cityscapes/exp/train_on_train_set/train --eval_logdir=./datasets/cityscapes/exp/train_on_train_set/eval --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1

$ python eval.py --logtostderr --eval_split="val" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=1025 --eval_crop_size=2049 --dataset="cityscapes" --checkpoint_dir=./datasets/cityscapes/exp/train_on_train_set/train --eval_logdir=./datasets/cityscapes/exp/train_on_train_set/eval --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1
```

---------------------
### 5.3 A local `visualization` job using `xception_65` can be run with the following command:
```shell
# run vis, From tensorflow/models/research/deeplab
$ python vis.py --logtostderr --vis_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=1025 --vis_crop_size=2049 --dataset="cityscapes" --colormap_type="cityscapes" --checkpoint_dir=./datasets/model_zoo/deeplabv3_cityscapes_train --vis_logdir=./datasets/cityscapes/exp/train_on_train_set/vis --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1

# run vis, mobilenet_v2
$ python vis.py --logtostderr --vis_split="val" --model_variant="mobilenet_v2"  --output_stride=8  --vis_crop_size=1025 --vis_crop_size=2049 --dataset="cityscapes" --colormap_type="cityscapes" --checkpoint_dir=./datasets/cityscapes/exp/train_on_train_set/train --vis_logdir=./datasets/cityscapes/exp/train_on_train_set/vis --dataset_dir=./datasets/cityscapes/tfrecord --max_number_of_iterations=1
## Get output file: 'datasets/cityscapes/exp/train_on_train_set/vis/segmentation_results'
```

---------------------
### 5.4 A local `training` job using `xception_65` can be run with the following command::
```shell
# 9W iters if fine for training.
# run training, From tensorflow/models/research/deeplab
# --train_batch_size <= 2

# Multi GPUs Training
## Fine-tuning `re-use all the trained wieghts`, train_batch_size=1, [OK]
$ python train.py --logtostderr --training_number_of_steps=90000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=1 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --fine_tune_batch_norm=False

## Fine-tuning  `re-use all the trained wieghts`
## global step 3550: loss = 1.6651 (0.626 sec/step), miou_1.0[0.149772376]
$ python train.py --logtostderr --training_number_of_steps=10000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=2 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False
 
## Fine-tuning From 'deeplabv3_cityscapes_train', `re-use only the network backbone`
$ python train.py --logtostderr --training_number_of_steps=90000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=4 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=False

## mobilenet_v2: miou_1.0[0.716534734](output_stride=8), miou_1.0[0.706700146](output_stride=16)
## mobilenet_v2, `re-use all the trained wieghts`, set `initialize_last_layer=True` [OK]
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="mobilenet_v2" --output_stride=8 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False

## mobilenet_v2, `re-use all the trained weights except the logits`
## global step 800: loss = 1.0841 (0.842 sec/step), miou_1.0[0.013041365]
## global step 2930: loss = 1.2053 (0.882 sec/step), miou_1.0[0.140691563]
## global step 30000: loss = 1.2253 (0.839 sec/step), miou_1.0[0.146490291]
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="mobilenet_v2" --output_stride=8 --train_crop_size=769 --train_crop_size=769 --train_batch_size=8 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=True

## mobilenet_v2, `re-use all the trained weights except the logits`
## global step 30000: loss = 1.3792 (0.252 sec/step), miou_1.0[0.159074679]
## step 51803, loss = 1.4126 (0.232 sec/step), miou_1.0[0.143954247]
$ python train.py --logtostderr --training_number_of_steps=90000 --train_split="train" --model_variant="mobilenet_v2" --output_stride=8 --train_crop_size=769 --train_crop_size=769 --train_batch_size=2 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=False --initialize_last_layer=False --last_layers_contain_logits_only=True

# 2019.03.14
## mobilenet_v2, `re-use only the network backbone`
## step 2000: loss = 1.3960 (0.680 sec/step), miou_1.0[0.0641207546]
## global step 5050: loss = 1.1510 (0.632 sec/step), miou_1.0[0.0680430755]
## add learning rate. --base_learning_rate=0.01
## global step 15000: loss = 1.5850 (0.688 sec/step), miou_1.0[0.138460264]
## 2019.03.15, 10:00 done!
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="mobilenet_v2" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=16 --dataset="cityscapes" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_mnv2_cityscapes_train/model.ckpt --train_logdir=./datasets/cityscapes/exp/train_on_train_set/train --dataset_dir=./datasets/cityscapes/tfrecord --num_clones=2 --fine_tune_batch_norm=True --initialize_last_layer=False --last_layers_contain_logits_only=False



-----------
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
```

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

## Fine-tuning From 'deeplabv3_cityscapes_train', `re-use ALL the trained weights EXCEPT the logits`
## global step 5290: loss = 0.5241 (0.518 sec/step), miou_1.0[0.638597667]
## global step 30000: loss = 0.5131 (0.506 sec/step), miou_1.0[0.667017698]
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=361 --train_crop_size=361 --train_batch_size=10 --dataset="camvid" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_cityscapes_train/model.ckpt --train_logdir=./datasets/camvid/exp/train_on_train_set/train --dataset_dir=./datasets/camvid/tfrecord --fine_tune_batch_norm=False --num_clones=2 --initialize_last_layer=False --last_layers_contain_logits_only=True

## Fine-tuning From 'deeplabv3_cityscapes_train', `re-use only the network backbone`
## global step 8960: loss = 0.6623 (0.619 sec/step), miou_1.0[0.497921795]
## global step 13710: loss = 0.6310 (0.572 sec/step), miou_1.0[0.537117]
## global step 30000: loss = 0.5538 (0.643 sec/step), miou_1.0[0.57864368]
$ python train.py --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=361 --train_crop_size=361 --train_batch_size=12 --dataset="camvid" --tf_initial_checkpoint=./datasets/model_zoo/deeplabv3_cityscapes_train/model.ckpt --train_logdir=./datasets/camvid/exp/train_on_train_set/train --dataset_dir=./datasets/camvid/tfrecord --fine_tune_batch_norm=False --num_clones=2 --initialize_last_layer=False --last_layers_contain_logits_only=False

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

* The issue here is that your training batch size (--train_batch_size=4) is too small. You need to have a larger batch size. 


* It seems I needed a much larger step length than the default. `1e-2` gives results closer to the published results, with batch size 15 and a smaller crop window size.

* I want to train Cityscape with a trained model but summurize some Classes. For that i changed the `trainid` of some classes.

* If I use the `export_model.py` without --decoder_output_stride=4 flag (as used in training) it always outputs 0.
solution: adding the `--decoder_output_stride=4 flag`

* Issues: Is it possible to get `94% mIOU` of pascal voc 'val' data?
Ans: You are training with `trainval` dataset which means you've already include the `val set` during `training`. That usually results in `overfitting` in val dataset with `mIoU over 90%`.

* To get the 82.1% performance (on test set), you need to further train the model on all the `fine + coarse` annotations.

* deeplabv3+ how to print `per_class_iou`?
[TensorFlow: How Can I get the total_cm in tf.contrib.metrics.streaming_mean_iou](https://stackoverflow.com/questions/40340728/tensorflow-how-can-i-get-the-total-cm-in-tf-contrib-metrics-streaming-mean-iou)

[Deeplab——How to evaluate each class of iou](https://blog.csdn.net/zsf442553199/article/details/82217717)

