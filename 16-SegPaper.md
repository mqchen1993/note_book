# 1. 修改deeplab，简化语义标签，提高实时性。
# 2. 在MultiNet基础上修改，完成道路分割、车辆、行人等检测。
# 3. 打破FCN Encoder-Decoder架构,设计one step end-to-end网络。预测语义分割每一类物体在图片上的像素轮廓，而不是端到端输出图片。
## 使用ResNet实现的分割网络效果state-of-art. 

# 2019.01.08
* 跑通KittiSeg evaluate.py程序	yes
* 整理KittiSeg和MultiNet代码		yes

# 2019.01.09计划(评价指标)
* 阅读MaxF1论文
* IOU?
