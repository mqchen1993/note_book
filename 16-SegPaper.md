# 1. 修改deeplab，简化语义标签，提高实时性。
# 2. 在MultiNet基础上修改，完成道路分割、车辆、行人等检测。
# 3. 打破FCN Encoder-Decoder架构,设计one step end-to-end网络。预测语义分割每一类物体在图片上的像素轮廓，而不是端到端输出图片。
## 使用ResNet实现的分割网络效果state-of-art. 

# 2019.01.08
* 跑通KittiSeg evaluate.py程序	yes
* 整理KittiSeg和MultiNet代码		yes

# 2019.01.09计划(评价指标)
## 阅读MaxF1论文(THE KITTI-ROAD DATASET)
* [blog1](https://blog.csdn.net/sinat_28576553/article/details/80258619)
* ego-lane vs.opposing lane,(自我车道与对方车道)
* 2D Bird’s Eye View (BEV) space.
* For methods that output confidence maps (in contrast to binary road classification), the classification threshold τ is chosen to maximize the F-measure.
* 
## IOU?

# 2019.01.13
## 评价MultNet训练结果
```python
segmentation Evaluation Finished. Results
Raw Results:
[train] MaxF1 (raw)    :  98.9767 
[train] BestThresh (raw)    :  68.2353 
[train] Average Precision (raw)    :  92.5437 
[val] MaxF1 (raw)    :  95.9975 
[val] BestThresh (raw)    :  14.9020 
[val] Average Precision (raw)    :  92.3125 
Speed (msec) (raw)    :  42.8566 
Speed (fps) (raw)    :  23.3336
```
## 读AP指标（VOC DATASET）
* [AP wiki](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)
* [blog1](https://blog.csdn.net/niaolianjiulin/article/details/53098437)
* [blog2](https://blog.csdn.net/hysteric314/article/details/54093734)
* 以recall为横坐标（0~1），precision为纵坐标（0~1）作图。得到一条曲线。该曲线下的面积即为AP.
* $ AP=\int_0^1 {p(r)}\r$ 

# 2019.01.14计划
## 标注Cityscape数据集
* [blog1](https://blog.csdn.net/fabulousli/article/details/78633531)
* [labelme 工具](https://github.com/wkentaro/labelme) 				yes
* [Semantic Segmentation using Fully Convolutional Networks over the years](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html)

