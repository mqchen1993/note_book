# CityScapes dataset.
[github](https://github.com/mcordts/cityscapesScripts#scripts)
---
# My login count.
* email: xiaomeixuehao@bit.edu.cn
* key: me+1230 
---
# blog
[csdn](https://blog.csdn.net/zz2230633069/article/details/84591532)

---
# 1. data structure.
```python
root
|
-- cityscapesScripts
|  |
|  |-- cityscapesscripts
|   -- docs
-- gtFine
|  |
|  |-- test
|   -- train 
-- leftImg8bit
   |
   |-- test
   |-- train
    -- val

# Train class id
[128, 64, 128],  # trainId 0: 'road'
[244, 35, 232],  # trainId 1: 'sidewalk'
[70, 70, 70],    # trainId 2: 'building'
[102, 102, 156], # trainId 3: 'wall'
[190, 153, 153], # trainId 4: 'fence'
[153, 153, 153], # trainId 5: 'pole'
[250, 170, 30],  # trainId 6: 'traffic light'
[220, 220, 0],   # trainId 7: 'traffic sign'
[107, 142, 35],  # trainId 8: 'vegetation'
[152, 251, 152], # trainId 9: 'terrain'
[70, 130, 180],  # trainId 10: 'sky'
[220, 20, 60],   # trainId 11: 'person'
[255, 0, 0],     # trainId 12: 'rider'
[0, 0, 142],     # trainId 13: 'car'
[0, 0, 70],      # trainId 14: 'truck'
[0, 60, 100],    # trainId 15: 'bus'
[0, 80, 100],    # trainId 16: 'train'
[0, 0, 230],     # trainId 17: 'motorcycle'
[119, 11, 32],   # trainId 18: 'bicycle'
```

---
# 2. build scripts.
## let python know script directory.
```shell
$ export PYTHONPATH="/media/jun/ubuntu/datasets/CityScapes/cityscapesScripts:$PYTHONPATH"
$ export CITYSCAPES_DATASET="/media/jun/ubuntu/datasets/CityScapes"
```

---
# 3. prepare data for MultiNet.
## change label color
* cityscapesscripts/preparation/createTrainIdLabelImgs.py, 
L67 `json2labelImg( f , dst , "color" ) # ids, trainIds, color`.
* cityscapesscripts/helpers/labels.py,
L62 Label 'road' color:`[255,0,255]`, else, `[255,0,0]`.
* python cityscapesscripts/preparation/createTrainIdLabelImgs.py
## prepare train&val&test list
* python prepare_list.py

# 4. run MultiNet.
```shell
# run demo.py
$ python demo_seg\&obj.py --input data/demo/6_8.png
# run train.py
$ python train.py --hypes hypes/multinet2.json
```

# 5 run evaluate script.
* prediction images are put in `CITYSCAPES_DATASET/results/`
* gt & prediction pair: `<city>_123456_123456*.png` & `<city>_123456_123456_gtFine_labelIds.png`
* only one image associate with gt image, otherwise it will print " ERROR: Found multiple predictions for ground truth".
* run script.
```shell
$ python cityscapesScripts/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py
```

# 6. Prepare `fine+coarse` data
## Coarse dataset 
* 'gt_Coarse/train_extra': 19998 files.
* fine + coarse: 24998
* `gt_Coarse` 生成的`*_gtCoarse_labelTrainIds.png`只有黑白两色? 视觉错误。

1. 修改`createTrainIdLabelImgs.py`只保留`searchCoarse` data.
$ python cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py 
2. 将`leftImg8bit`文件夹下`train_extra` copy 合并为`fine+coarse`; 将`gtFine`文件夹下`train_extra` copy 合并为`fine+coarse`;
3. 运行转换脚本
$ ./convert_cityscapes_fine_coarse.sh








