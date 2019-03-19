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


# Coarse dataset 
* 'gt_Coarse/train_extra': 19998 files.
* fine + coarse: 24998
* `gt_Coarse` 生成的`*_gtCoarse_labelTrainIds.png`只有黑白两色?

