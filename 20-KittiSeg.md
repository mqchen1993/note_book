# KittiSeg note.
---

# run KittiSeg.
```shell
# run demo.py
$ python demo.py --input_image data/demo/demo.png
# run train.py
$ python train.py --hypes hypes/KittiSeg.json
# run evaluate.py
# Modify L32 `flags.DEFINE_string('RUN'` & L65 `if not FLAGS.RUN ==`.
$ python evaluate.py
```

# FCN论文中作者给出的在VOC2011数据集上的 `Pixel Accuracy`为90.3, mean IoU（即多个类别IoU的平均值，具体看我之前给你写的‘语义分割评价指标‘，IoU即单个类别计算结果，IoU=TP/(TP+FN+FP)）为62.7
# Kitti Road benchmark（http://www.cvlibs.net/datasets/kitti/eval_road.php）
* 目前最好的模型MaxF1:97.05 %， AP：93.53 %； MultiNet 分别为：93.99 % 	93.24 %
* Kitti Road 用MaxF1和AP作为评价指标，这都是像素分类的评价指标，应该时数据集只有单个类别，这样评价比较合理）

# CityScapes benchmark（https://www.cityscapes-dataset.com/benchmarks/#scene-labeling-task）
* 目前最好 Class IoU：83.6，之前跑过的DeepLabv3+为82.1。
* CityScapes 用IoU作为评价指标，这都是语义分割的评价指标，应该是数据集有多个类别，这样评价比较合理

# VOC2012 benchmark（http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_FCN-8s）
* 目前最好 DeepLabv3+_JFT	mIoU: 89.0, DeepLabv3+ 为87.8， 	原生FCN-8s（模型结构同KittiSeg）mIoU：62.2.


----------------------------------
# 自己做的实验。
* KittiSeg Deconvelution 部分
```python
Shape of Validation/scale5/block3/Relu:0[1 12 39 2048]
Shape of upscore2[1 24 78 2]
Shape of upscore4[1 48 156 2]
Shape of upscore32[1 384 1248 2]
```

# 1. KittiSeg(FCN-8s) 在 Kitti Road 数据集上训练结果(在kitti road评价)（作者给的模型，基于VGG16）：
```shell
2019-01-24 22:55:40,039 INFO Evaluation Succesfull. Results:
2019-01-24 22:55:40,040 INFO     `MaxF1  :  96.0821` 
2019-01-24 22:55:40,040 INFO     BestThresh  :  14.5098 
2019-01-24 22:55:40,040 INFO     Average Precision  :  92.3620 
2019-01-24 22:55:40,040 INFO     `Pixel Accuracy  :  97.8370` 
2019-01-24 22:55:40,040 INFO     IOU  :  89.4572 
2019-01-24 22:55:40,040 INFO     Speed (msec)  :  88.4474 
2019-01-24 22:55:40,040 INFO     Speed (fps)  :  11.3062 
```

# 2. KittiSeg(FCN) 在 Kitti Road 数据集上训练结果(在kitti road评价)（基于ResNet101）：
```shell
2019-01-24 23:20:08,677 INFO Evaluation Succesfull. Results:
2019-01-24 23:20:08,677 INFO     [train] MaxF1  :  99.6044 
2019-01-24 23:20:08,677 INFO     [train] BestThresh  :  61.5686 
2019-01-24 23:20:08,678 INFO     [train] Average Precision  :  92.5433 
2019-01-24 23:20:08,678 INFO     [train] Pixel Accuracy  :  99.4363 
2019-01-24 23:20:08,678 INFO     [train] IOU  :  98.3407 
2019-01-24 23:20:08,678 INFO     `[val] MaxF1  :  96.6176` 
2019-01-24 23:20:08,678 INFO     [val] BestThresh  :  31.7647 
2019-01-24 23:20:08,678 INFO     `[val] Average Precision  :  92.0721` 
2019-01-24 23:20:08,678 INFO     [val] Pixel Accuracy  :  98.4305 
2019-01-24 23:20:08,678 INFO     [val] IOU  :  92.8428 
2019-01-24 23:20:08,678 INFO     Speed (msec)  :  62.4584 
2019-01-24 23:20:08,678 INFO     Speed (fps)  :  16.0107 
```

# 3. KittiSeg(FCN) 在 Kitti Road + CityScapes 数据集上训练(在kitti road评价)结果（基于ResNet101）：
```shell
2019-01-24 23:24:14,275 INFO Evaluation Succesfull. Results:
2019-01-24 23:24:14,275 INFO     [train] MaxF1  :  98.1683 
2019-01-24 23:24:14,275 INFO     [train] BestThresh  :  76.8627 
2019-01-24 23:24:14,275 INFO     [train] Average Precision  :  92.5292 
2019-01-24 23:24:14,275 INFO     [train] Pixel Accuracy  :  98.7130 
2019-01-24 23:24:14,275 INFO     [train] IOU  :  94.5232 
2019-01-24 23:24:14,275 INFO     `[val] MaxF1  :  96.7849` 
2019-01-24 23:24:14,276 INFO     [val] BestThresh  :  67.0588 
2019-01-24 23:24:14,276 INFO     `[val] Average Precision  :  92.3420` 
2019-01-24 23:24:14,276 INFO     [val] Pixel Accuracy  :  98.3321 
2019-01-24 23:24:14,276 INFO     [val] IOU  :  92.4040 
2019-01-24 23:24:14,276 INFO     Speed (msec)  :  49.6830 
2019-01-24 23:24:14,276 INFO     Speed (fps)  :  20.1276 
```

# 3. KittiSeg(FCN) 在 Kitti Road + CityScapes 数据集上训练(在Kitti Road + CityScapes验证集上评价)结果（基于ResNet101）：
```shell
2019-01-25 00:20:39,907 INFO Evaluation Succesfull. Results:
2019-01-25 00:20:39,907 INFO     [val] MaxF1  :  96.3243 
2019-01-25 00:20:39,907 INFO     [val] BestThresh  :  72.9412 
2019-01-25 00:20:39,907 INFO     [val] Average Precision  :  92.6267 
2019-01-25 00:20:39,908 INFO     [val] Pixel Accuracy  :  96.9362 
2019-01-25 00:20:39,908 INFO     [val] IOU  :  91.3910 
2019-01-25 00:20:39,908 INFO     Speed (msec)  :  47.1982 
2019-01-25 00:20:39,908 INFO     Speed (fps)  :  21.1872 
```

# 4. KittiSeg(FCN) 在 Kitti Road + CityScapes 数据集上训练(在kitti road评价)结果（基于ResNet50）：
```python
2019-01-24 23:29:14,570 INFO Evaluation Succesfull. Results:
2019-01-24 23:29:14,570 INFO     [train] MaxF1  :  97.0180 
2019-01-24 23:29:14,570 INFO     [train] BestThresh  :  68.6275 
2019-01-24 23:29:14,570 INFO     [train] Average Precision  :  92.4995 
2019-01-24 23:29:14,570 INFO     [train] Pixel Accuracy  :  98.2073 
2019-01-24 23:29:14,571 INFO     [train] IOU  :  91.9377 
2019-01-24 23:29:14,571 INFO     [val] MaxF1  :  95.2738 
2019-01-24 23:29:14,571 INFO     [val] BestThresh  :  52.1569 
2019-01-24 23:29:14,571 INFO     [val] Average Precision  :  92.1730 
2019-01-24 23:29:14,571 INFO     [val] Pixel Accuracy  :  97.5686 
2019-01-24 23:29:14,571 INFO     [val] IOU  :  88.4408 
2019-01-24 23:29:14,571 INFO     Speed (msec)  :  35.1630 
2019-01-24 23:29:14,571 INFO     Speed (fps)  :  28.4390
```

# 5. KittiSeg(FCN) 在 CityScapes(only car) 数据集上训练结果（基于ResNet101）：
```python
2019-01-26 11:39:39,349 INFO Evaluation Succesfull. Results:
2019-01-26 11:39:39,350 INFO     [train] MaxF1  :  96.2657 
2019-01-26 11:39:39,350 INFO     [train] BestThresh  :  68.2353 
2019-01-26 11:39:39,350 INFO     [train] Average Precision  :  91.3601 
2019-01-26 11:39:39,350 INFO     [train] Pixel Accuracy  :  98.9277 
2019-01-26 11:39:39,350 INFO     [train] IOU  :  89.1982 
2019-01-26 11:39:39,351 INFO     [val] MaxF1  :  94.5932 
2019-01-26 11:39:39,351 INFO     [val] BestThresh  :  69.0196 
2019-01-26 11:39:39,351 INFO     [val] Average Precision  :  90.9463 
2019-01-26 11:39:39,351 INFO     [val] Pixel Accuracy  :  98.7662 
2019-01-26 11:39:39,351 INFO     [val] IOU  :  86.0095 
2019-01-26 11:39:39,351 INFO     Speed (msec)  :  50.6929 
2019-01-26 11:39:39,352 INFO     Speed (fps)  :  19.7266 
```


# 6. KittiSeg 在 CityScapes(only person&rider) 数据集上训练结果（基于ResNet101）：
```python
2019-01-26 23:32:40,141 INFO Evaluation Succesfull. Results:
2019-01-26 23:32:40,142 INFO     [train] MaxF1  :  88.4435 
2019-01-26 23:32:40,142 INFO     [train] BestThresh  :  59.6078 
2019-01-26 23:32:40,142 INFO     [train] Average Precision  :  88.6716 
2019-01-26 23:32:40,142 INFO     [train] Pixel Accuracy  :  99.1806 
2019-01-26 23:32:40,142 INFO     [train] IOU  :  71.3110 
2019-01-26 23:32:40,143 INFO     [val] MaxF1  :  83.0147 
2019-01-26 23:32:40,143 INFO     [val] BestThresh  :  57.6471 
2019-01-26 23:32:40,143 INFO     [val] Average Precision  :  85.5599 
2019-01-26 23:32:40,143 INFO     [val] Pixel Accuracy  :  98.9799 
2019-01-26 23:32:40,143 INFO     [val] IOU  :  63.6032 
2019-01-26 23:32:40,143 INFO     Speed (msec)  :  49.9547 
2019-01-26 23:32:40,144 INFO     Speed (fps)  :  20.0181 
```

# 7. KittiSeg 在 CityScapes(only bus) 数据集上训练结果（基于ResNet101）：
2019-01-30 01:33:06,219 root INFO Raw Results:
2019-01-30 01:33:06,219 root INFO     [train] MaxF1 (raw)    :  93.7513 
2019-01-30 01:33:06,219 root INFO     [train] BestThresh (raw)    :  69.4118 
2019-01-30 01:33:06,219 root INFO     [train] Average Precision (raw)    :  90.3027 
2019-01-30 01:33:06,219 root INFO     [train] Pixel Accuracy (raw)    :  99.5699 
2019-01-30 01:33:06,219 root INFO     [train] IOU (raw)    :  83.0718 
2019-01-30 01:33:06,219 root INFO     [val] MaxF1 (raw)    :  75.2567 
2019-01-30 01:33:06,219 root INFO     [val] BestThresh (raw)    :  20.3922 
2019-01-30 01:33:06,219 root INFO     [val] Average Precision (raw)    :  74.3074 
2019-01-30 01:33:06,219 root INFO     [val] Pixel Accuracy (raw)    :  99.4377 
2019-01-30 01:33:06,219 root INFO     [val] IOU (raw)    :  54.6316 
2019-01-30 01:33:06,219 root INFO     Speed (msec) (raw)    :  64.5966 
2019-01-30 01:33:06,219 root INFO     Speed (fps) (raw)    :  15.4807 
2019-01-30 01:33:06,220 root INFO Smooth Results:
2019-01-30 01:33:06,221 root INFO     [train] MaxF1 (smooth) :  91.0865 
2019-01-30 01:33:06,221 root INFO     [train] BestThresh (smooth) :  58.6275 
2019-01-30 01:33:06,221 root INFO     [train] Average Precision (smooth) :  89.4749 
2019-01-30 01:33:06,221 root INFO     [train] Pixel Accuracy (smooth) :  99.5545 
2019-01-30 01:33:06,221 root INFO     [train] IOU (smooth) :  78.3216 
2019-01-30 01:33:06,221 root INFO     [val] MaxF1 (smooth) :  69.3762 
2019-01-30 01:33:06,221 root INFO     [val] BestThresh (smooth) :  19.4118 
2019-01-30 01:33:06,221 root INFO     [val] Average Precision (smooth) :  69.2080 
2019-01-30 01:33:06,221 root INFO     [val] Pixel Accuracy (smooth) :  99.4036 
2019-01-30 01:33:06,221 root INFO     [val] IOU (smooth) :  47.0834 
2019-01-30 01:33:06,221 root INFO     Speed (msec) (smooth) :  62.1653 
2019-01-30 01:33:06,221 root INFO     Speed (fps) (smooth) :  16.1108 
