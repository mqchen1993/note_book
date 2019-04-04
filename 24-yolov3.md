
-----------------------
## YOLO
- [x] **跑通YOLO**
* [YOLO](https://pjreddie.com/darknet/yolo/)
* Darknet is an open source neural network framework written in `C and CUDA`. It is fast, easy to install, and supports `CPU and GPU computation`.

### Build & run
```shell
# Build with CPU
$ git clone https://github.com/pjreddie/darknet
$ cd darknet
$ make
$ wget https://pjreddie.com/media/files/yolov3.weights
# run with CPU
$ ./darknet detect cfg/yolov3.cfg weights/yolov3.weights data/dog.jpg

# Build with GPU
## change the first line of the `Makefile`
GPU=1		# build with CUDA to accelerate by using GPU (CUDA should be in /usr/local/cuda)
CUDNN=1		# build with cuDNN v5-v7 to accelerate training by using GPU ( /usr/local/cudnn)
OPENCV=1	# build with OpenCV 3.x/2.4.x - allows to detect on video files and video streams
OPENMP=1	# build with OpenMP support to accelerate Yolo by using multi-core CPU
DEBUG=0
ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_61,code=[sm_61,compute_61]

## run with GPU
$ ./darknet detector test cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights data/dog.jpg 
$ ./darknet detect cfg/yolov3.cfg weights/yolov3.weights data/dog.jpg
## use GPU 1
$ ./darknet -i 1 detect cfg/yolov3.cfg weights/yolov3.weights data/dog.jpg
## run with video file
### FPS:18.2
$ ./darknet detector demo cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights data/1.avi 
### FPS:180.9
$ ./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights data/1.avi

## Train on VOC
### modify 'cfg/yolov3-voc.cfg', uncomment the 'Training' parameters.
# Testing
# batch=1
# subdivisions=1
# Training
 batch=64
 subdivisions=16
###########################################
# max_batches = 50200,迭代次数
$ ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg  weights/darknet53.conv.74 -gpus 0,1
# Test training
## FPS:47.6
$ ./darknet detector -i 1 demo cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc.backup data/1.avi



## Train on COCO
# max_batches = 500200,迭代次数
$ ./darknet detector train cfg/coco.data cfg/yolov3.cfg weights/darknet53.conv.74 -gpus 0,1
```

* Where `x, y, width, and height` are `relative to the image's width and height`.

----------------------------------------------
# 2019.03.03
## YOLOv3
- [x] [How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 1](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
* `YOLO`, `SSD`, `Mask RCNN` and `RetinaNet`.

### What is YOLO?
* YOLO makes use of only convolutional layers, making it a fully convolutional network (FCN). It has `75 convolutional layers`, with `skip connections` and `upsampling` layers. `No form of pooling` is used, and a convolutional layer with stride 2 is used to downsample the feature maps. 
* `Being a FCN`, YOLO is invariant to the size of the input image. However, `in practice`, we might want to stick to a `constant input size`:
> Process our images in `batches` (images in batches can be processed in parallel by the GPU, leading to speed boosts), we need to have all images of fixed height and width. 
* The network `downsamples` the image by a factor called the **stride** of the network.

### Interpreting the output
* Now, the first thing to notice is `our output is a feature map`.
* `Depth-wise`, we have `(B x (5 + C)) entries` in the feature map.
> `B` represents the number of bounding boxes each cell can predict.  Each of the bounding boxes have 5 + C attributes, which describe the `center coordinates`, the `dimensions`, the `objectness score` and `C class confidences` for each bounding box. YOLO v3 predicts `3 bounding boxes for every cell`.
* To do that, we `divide` the `input image` into `a grid of dimensions` equal to that of `the final feature map`.
> Let us consider an example below, where the input image is 416 x 416, and stride of the network is 32. As pointed earlier, the dimensions of the feature map will be 13 x 13. We then divide the input image into 13 x 13 cells.

![](https://blog.paperspace.com/content/images/2018/04/yolo-5.png)
> Then, the cell (on the input image) containing `the center of the ground truth box` of an object is chosen to be the one `responsible for predicting the object`. In the image, it is the cell which marked `red`, which contains the center of the ground truth box (marked yellow).
* We divide the `input image` into a `grid` just to determine which cell of the prediction `feature map` is `responsible for prediction`.

### Anchor Boxes
* It might make sense to predict the width and the height of the bounding box, but `in practice`, that leads to `unstable gradients` during training. `Instead`, most of the modern object detectors predict log-space transforms, or `simply` offsets to pre-defined default bounding boxes called `anchors`.
* The bounding box `responsible for detecting the dog` will be the one whose anchor has the `highest IoU` with the `ground truth box`.

### Making Predictions
* The following formulae describe how the network output is transformed to obtain bounding box predictions.

![](https://blog.paperspace.com/content/images/2018/04/Screen-Shot-2018-04-10-at-3.18.08-PM.png)

> `bx, by, bw, bh` are the x,y center cvoc persono-ordinates(坐标), width and height of our prediction. `tx, ty, tw, th` is what the `network outputs`. `cx and cy` are the top-left co-ordinates of the grid. `pw and ph` are anchors dimensions for the box.

### Center Coordinates
* Notice we are running our `center coordinates prediction` through a `sigmoid` function. This forces the value of the `output` to be between `0 and 1`. 
> For example, consider the case of our `dog image`. If the `prediction for center` is (0.4, 0.7), then this means that the center lies at (6.4, 6.7) on the 13 x 13 feature map. (Since the top-left co-ordinates of the red cell are (6,6)).

### Dimensions of the Bounding Box
* The `dimensions(bw, bh)` of the bounding box are predicted by applying a log-space transform to the `output(tw, th)` and then multiplying with an `anchor(pw, ph)`.
$$ {bw = pw*e^{tw}}, {bh = ph*e^{th}}  $$

![](https://blog.paperspace.com/content/images/2018/04/yolo-regression-1.png)

### Objectness Score
* `Object score` represents the `probability` that an object is contained inside a bounding box. It should be nearly 1 for the red and the neighboring grids, whereas almost 0 for, say, the grid at the corners.

* The `objectness score` is also passed through a `sigmoid`, as it is to be interpreted as a `probability`.

### Class Confidences
* Class confidences represent the probabilities of the detected object belonging to a particular class (Dog, cat, banana, car etc).
* Before v3, YOLO used to `softmax` the class scores.
* In v3, and authors have opted for using `sigmoid` instead.
* The reason is that `Softmaxing` class scores assume that the classes are `mutually exclusive`(相互排斥).
> In simple words, if an object belongs to one class, then it's guaranteed it cannot belong to another class. This is true for COCO database on which we will base our detector.

> `However`, this assumptions may not hold when we have classes like `Women` and `Person`. This is the reason that authors have steered clear of using a Softmax activation.

### Prediction across different scales.
* YOLO v3 makes prediction across `3 different scales`. The detection layer is used make detection at feature maps of three different sizes, having `strides 32, 16, 8 respectively`. This means, with an `input of 416 x 416`, we make detections on `scales 13 x 13, 26 x 26 and 52 x 52`.
* The network downsamples the input image until the first detection layer, where a detection is made using feature maps of a layer with strivoc personde 32. Further, layers are upsampled by a factor of 2 and concatenated with feature maps of a previous layers having identical feature map sizes. Another detection is now made at layer with stride 16. The same upsampling procedure is repeated, and a final detection is made at the layer of stride 8.

![](https://blog.paperspace.com/content/images/2018/04/yolo_Scales-1.png)

* `Upsampling` can help the network learn `fine-grained(细粒度) features` which are instrumental for detecting `small objects`.

### Output Processing
* For an image of size `416 x 416`, YOLO predicts ((52 x 52) + (26 x 26) + 13 x 13)) x 3 = `10647 bounding boxes`. 
* Thresholding by Object Confidence.
* Non-maximum Suppression.

---------------------------------------------
# 2019.03.05
## YOLOv3
- [x] [How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 2](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/)
* Configuration File(cfg/文件下)详细解析。
```python
[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=16
```

- [x] [YOLOv3网络结构细致解析](https://blog.csdn.net/sum_nap/article/details/80568873)

### layer filters size input output

#### VOC dataset
```python
0 conv 32 3 x 3 / 1  416 x 416 x 3 -> 416 x 416 x 32  0.299 BFLOPs	
1 conv 64 3 x 3 / 2  416 x 416 x 32 -> 208 x 208 x 64  1.595 BFLOPs	
2 conv 32 1 x 1 / 1 208 x 208 x 64 -> 208 x 208 x 32 0.177 BFLOPs
3 conv 64 3 x 3 / 1 208 x 208 x 32 -> 208 x 208 x 64 1.595 BFLOPs
4 res 1 208 x 208 x 64 -> 208 x 208 x 64
5 conv 128 3 x 3 / 2 208 x 208 x 64 -> 104 x 104 x 128 1.595 BFLOPs
6 conv 64 1 x 1 / 1 104 x 104 x 128 -> 104 x 104 x 64 0.177 BFLOPs
7 conv 128 3 x 3 / 1 104 x 104 x 64 -> 104 x 104 x 128 1.595 BFLOPs
8 res 5 104 x 104 x 128 -> 104 x 104 x 128
9 conv 64 1 x 1 / 1 104 x 104 x 128 -> 104 x 104 x 64 0.177 BFLOPs
10 conv 128 3 x 3 / 1 104 x 104 x 64 -> 104 x 104 x 128 1.595 BFLOPs
11 res 8 104 x 104 x 128 -> 104 x 104 x 128
12 conv 256 3 x 3 / 2 104 x 104 x 128 -> 52 x 52 x 256 1.595 BFLOPs
13 conv 128 1 x 1 / 1 52 x 52 x 256 -> 52 x 52 x 128 0.177 BFLOPs
14 conv 256 3 x 3 / 1 52 x 52 x 128 -> 52 x 52 x 256 1.595 BFLOPs
15 res 12 52 x 52 x 256 -> 52 x 52 x 25voc person6
16 conv 128 1 x 1 / 1 52 x 52 x 256 -> 52 x 52 x 128 0.177 BFLOPs
17 conv 256 3 x 3 / 1 52 x 52 x 128 -> 52 x 52 x 256 1.595 BFLOPs
18 res 15 52 x 52 x 256 -> 52 x 52 x 256
19 conv 128 1 x 1 / 1 52 x 52 x 256 -> 52 x 52 x 128 0.177 BFLOPs
20 conv 256 3 x 3 / 1 52 x 52 x 128 -> 52 x 52 x 256 1.595 BFLOPs
21 res 18 52 x 52 x 256 -> 52 x 52 x 256
22 conv 128 1 x 1 / 1 52 x 52 x 256 -> 52 x 52 x 128 0.177 BFLOPs
23 conv 256 3 x 3 / 1 52 x 52 x 128 -> 52 x 52 x 256 1.595 BFLOPs
24 res 21 52 x 52 x 256 -> 52 x 52 x 256
25 conv 128 1 x 1 / 1 52 x 52 x 256 -> 52 x 52 x 128 0.177 BFLOPs
26 conv 256 3 x 3 / 1 52 x 52 x 128 -> 52 x 52 x 256 1.595 BFLOPs
27 res 24 52 x 52 x 256 -> 52 x 52 x 256
28 conv 128 1 x 1 / 1 52 x 52 x 256 -> 52 x 52 x 128 0.177 BFLOPs
29 conv 256 3 x 3 / 1 52 x 52 x 128 -> 52 x 52 x 256 1.595 BFLOPs
30 res 27 52 x 52 x 256 -> 52 x 52 x 256
31 conv 128 1 x 1 / 1 52 x 52 x 256 -> 52 x 52 x 128 0.177 BFLOPs
32 conv 256 3 x 3 / 1 52 x 52 x 128 -> 52 x 52 x 256 1.595 BFLOPs
33 res 30 52 x 52 x 256 -> 52 x 52 x 256
34 conv 128 1 x 1 / 1 52 x 52 x 256 -> 52 x 52 x 128 0.177 BFLOPs
35 conv 256 3 x 3 / 1 52 x 52 x 128 -> 52 x 52 x 256 1.595 BFLOPs
36 res 33 52 x 52 x 256 -> 52 x 52 x 256
37 conv 512 3 x 3 / 2 52 x 52 x 256 -> 26 x 26 x 512 1.595 BFLOPs
38 conv 256 1 x 1 / 1 26 x 26 x 512 -> 26 x 26 x 256 0.177 BFLOPs
39 conv 512 3 x 3 / 1 26 x 26 x 256 -> 26 x 26 x 512 1.595 BFLOPs
40 res 37 26 x 26 x 512 -> 26 x 26 x 512
41 conv 256 1 x 1 / 1 26 x 26 x 512 -> 26 x 26 x 256 0.177 BFLOPs
42 conv 512 3 x 3 / 1 26 x 26 x 256 -> 26 x 26 x 512 1.595 BFLOPs
43 res 40 26 x 26 x 512 -> 26 x 26 x 512
44 conv 256 1 x 1 / 1 26 x 26 x 512 -> 26 x 26 x 256 0.177 BFLOPs
45 conv 512 3 x 3 / 1 26 x 26 x 256 -> 26 x 26 x 512 1.595 BFLOPs
46 res 43 26 x 26 x 512 -> 26 x 26 x 512
47 conv 256 1 x 1 / 1 26 x 26 x 512 -> 26 x 26 x 256 0.177 BFLOPs
48 conv 512 3 x 3 / 1 26 x 26 x 256 -> 26 x 26 x 512 1.595 BFLOPs
49 res 46 26 x 26 x 512 -> 26 x 26 x 512
50 conv 256 1 x 1 / 1 26 x 26 x 512 -> 26 x 26 x 256 0.177 BFLOPs
51 conv 512 3 x 3 / 1 26 x 26 x 256 -> 26 x 26 x 512 1.595 BFLOPs
52 res 49 26 x 26 x 512 -> 26 x 26 x 512
53 conv 256 1 x 1 / 1 26 x 26 x 512 -> 26 x 26 x 256 0.177 BFLOPs
54 conv 512 3 x 3 / 1 26 x 26 x 256 -> 26 x 26 x 512 1.595 BFLOPs
55 res 52 26 x 26 x 512 -> 26 x 26 x 512
56 conv 256 1 x 1 / 1 26 x 26 x 512 -> 26 x 26 x 256 0.177 BFLOPs
57 conv 512 3 x 3 / 1 26 x 26 x 256 -> 26 x 26 x 512 1.595 BFLOPs
58 res 55 26 x 26 x 512 -> 26 x 26 x 512
59 conv 256 1 x 1 / 1 26 x 26 x 512 -> 26 x 26 x 256 0.177 BFLOPs
60 conv 512 3 x 3 / 1 26 x 26 x 256 -> 26 x 26 x 512 1.595 BFLOPs
61 res 58 26 x 26 x 512 -> 26 x 26 x 5voc person12
62 conv 1024 3 x 3 / 2 26 x 26 x 512 -> 13 x 13 x1024 1.595 BFLOPs
63 conv 512 1 x 1 / 1 13 x 13 x1024 -> 13 x 13 x 512 0.177 BFLOPs
64 conv 1024 3 x 3 / 1 13 x 13 x 512 -> 13 x 13 x1024 1.595 BFLOPs
65 res 62 13 x 13 x1024 -> 13 x 13 x1024
66 conv 512 1 x 1 / 1 13 x 13 x1024 -> 13 x 13 x 512 0.177 BFLOPs
67 conv 1024 3 x 3 / 1 13 x 13 x 512 -> 13 x 13 x1024 1.595 BFLOPs
68 res 65 13 x 13 x1024 -> 13 x 13 x1024
69 conv 512 1 x 1 / 1 13 x 13 x1024 -> 13 x 13 x 512 0.177 BFLOPs
70 conv 1024 3 x 3 / 1 13 x 13 x 512 -> 13 x 13 x1024 1.595 BFLOPs
71 res 68 13 x 13 x1024 -> 13 x 13 x1024
72 conv 512 1 x 1 / 1 13 x 13 x1024 -> 13 x 13 x 512 0.177 BFLOPs
73 conv 1024 3 x 3 / 1 13 x 13 x 512 -> 13 x 13 x1024 1.595 BFLOPs
74 res 71 13 x 13 x1024 -> 13 x 13 x1024
75 conv 512 1 x 1 / 1 13 x 13 x1024 -> 13 x 13 x 512 0.177 BFLOPs
76 conv 1024 3 x 3 / 1 13 x 13 x 512 -> 13 x 13 x1024 1.595 BFLOPs
77 conv 512 1 x 1 / 1 13 x 13 x1024 -> 13 x 13 x 512 0.177 BFLOPs
78 conv 1024 3 x 3 / 1 13 x 13 x 512 -> 13 x 13 x1024 1.595 BFLOPs
79 conv 512 1 x 1 / 1 13 x 13 x1024 -> 13 x 13 x 512 0.177 BFLOPs
80 conv 1024 3 x 3 / 1 13 x 13 x 512 -> 13 x 13 x1024 1.595 BFLOPs
81 conv 75 1 x 1 / 1 13 x 13 x1024 -> 13 x 13 x 75 0.026 BFLOPs
82 yolo
83 route 79
84 conv 256 1 x 1 / 1 13 x 13 x 512 -> 13 x 13 x 256 0.044 BFLOPs
85 upsample 2x 13 x 13 x 256 -> 26 x 26 x 256
86 route 85 61
87 conv 256 1 x 1 / 1 26 x 26 x 768 -> 26 x 26 x 256 0.266 BFLOPs
88 conv 512 3 x 3 / 1 26 x 26 x 256 -> 26 x 26 x 512 1.595 BFLOPs
89 conv 256 1 x 1 / 1 26 x 26 x 512 -> 26 x 26 x 256 0.177 BFLOPs
90 conv 512 3 x 3 / 1 26 x 26 x 256 -> 26 x 26 x 512 1.595 BFLOPs
91 conv 256 1 x 1 / 1 26 x 26 x 512 -> 26 x 26 x 256 0.177 BFLOPs
92 conv 512 3 x 3 / 1 26 x 26 x 256 -> 26 x 26 x 512 1.595 BFLOPs
93 conv 75 1 x 1 / 1 26 x 26 x 512 -> 26 x 26 x 75 0.052 BFLOPs
94 yolo
95 route 91
96 conv 128 1 x 1 / 1 26 x 26 x 256 -> 26 x 26 x 128 0.044 BFLOPs
97 upsample 2x 26 x 26 x 128 -> 52 x 52 x 128
98 route 97 36
99 conv 128 1 x 1 / 1 52 x 52 x 384 -> 52 x 52 x 128 0.266 BFLOPs
100 conv 256 3 x 3 / 1 52 x 52 x 128 -> 52 x 52 x 256 1.595 BFLOPs
101 conv 128 1 x 1 / 1 52 x 52 x 256 -> 52 x 52 x 128 0.177 BFLOPs
102 conv 256 3 x 3 / 1 52 x 52 x 128 -> 52 x 52 x 256 1.595 BFLOPs
103 conv 128 1 x 1 / 1 52 x 52 x 256 -> 52 x 52 x 128 0.177 BFLOPs
104 conv 256 3 x 3 / 1 52 x 52 x 128 -> 52 x 52 x 256 1.595 BFLOPs
105 conv 75 1 x 1 / 1 52 x 52 x 256 -> 52 x 52 x 75 0.104 BFLOPs
106 yolo
```

#### COCO dataset
```python
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   608 x 608 x   3   ->   608 x 608 x  32  0.639 BFLOPs
    1 conv     64  3 x 3 / 2   608 x 608 x  32   ->   304 x 304 x  64  3.407 BFLOPs
    2 conv     32  1 x 1 / 1   304voc person x 304 x  64   ->   304 x 304 x  32  0.379 BFLOPs
    3 conv     64  3 x 3 / 1   304 x 304 x  32   ->   304 x 304 x  64  3.407 BFLOPs
    4 res    1                 304 x 304 x  64   ->   304 x 304 x  64
    5 conv    128  3 x 3 / 2   304 x 304 x  64   ->   152 x 152 x 128  3.407 BFLOPs
    6 conv     64  1 x 1 / 1   152 x 152 x 128   ->   152 x 152 x  64  0.379 BFLOPs
    7 conv    128  3 x 3 / 1   152 x 152 x  64   ->   152 x 152 x 128  3.407 BFLOPs
    8 res    5                 152 x 152 x 128   ->   152 x 152 x 128
    9 conv     64  1 x 1 / 1   152 x 152 x 128   ->   152 x 152 x  64  0.379 BFLOPs
   10 conv    128  3 x 3 / 1   152 x 152 x  64   ->   152 x 152 x 128  3.407 BFLOPs
   11 res    8                 152 x 152 x 128   ->   152 x 152 x 128
   12 conv    256  3 x 3 / 2   152 x 152 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   13 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   14 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   15 res   12                  76 x  76 x 256   ->    76 x  76 x 256
   16 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   17 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   18 res   15                  76 x  76 x 256   ->    76 x  76 x 256
   19 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   20 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   21 res   18                  76 x  76 x 256   ->    76 x  76 x 256
   22 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   23 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   24 res   21                  76 x  76 x 256   ->    76 x  76 x 256
   25 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   26 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   27 res   24                  76 x  76 x 256   ->    76 x  76 x 256
   28 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   29 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   30 res   27                  76 x  76 x 256   ->    76 x  76 x 256
   31 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   32 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   33 res   30                  76 x  76 x 256   ->    76 x  76 x 256
   34 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   35 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   36 res   33                  76 x  76 x 256   ->    76 x  76 x 256
   37 conv    512  3 x 3 / 2    76 x  76 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   38 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   39 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   40 res   37                  38 x  38 x 512   ->    38 x  38 x 512
   41 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   42 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   43 res   40                  38 x  38 x 512   ->    38 x  38 x 512
   44 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   45 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   46 res   43                  38 x  38 x 512   ->    38 x  38 x 512
   47 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   48 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   49 res   46                  38 x  38 x 512   ->    38 x  38 x 512
   50 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   51 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   52 res   49                  38 x  38 x 512   ->    38 x  38 x 512
   53 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   54 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   55 res   52                  38 x  38 x 512   ->    38 x  38 x 512
   56 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   57 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   58 res   55                  38 x  38 x 512   ->    38 x  38 x 512
   59 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   60 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   61 res   58                  38 x  38 x 512   ->    38 x  38 x 512
   62 conv   1024  3 x 3 / 2    38 x  38 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   63 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
   64 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   65 res   62                  19 x  19 x1024   ->    19 x  19 x1024
   66 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
   67 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   68 res   65                  19 x  19 x1024   ->    19 x  19 x1024
   69 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
   70 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   71 res   68                  19 x  19 x1024   ->    19 x  19 x1024
   72 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
   73 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   74 res   71                  19 x  19 x1024   ->    19 x  19 x1024
   75 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
   76 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   77 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
   78 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   79 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
   80 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   81 conv    255  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 255  0.189 BFLOPs
   82 yolo
   83 route  79
   84 conv    256  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 256  0.095 BFLOPs
   85 upsample            2x    19 x  19 x 256   ->    38 x  38 x 256
   86 route  85 61
   87 conv    256  1 x 1 / 1    38 x  38 x 768   ->    38 x  38 x 256  0.568 BFLOPs
   88 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   89 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   90 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   91 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   92 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   93 conv    255  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 255  0.377 BFLOPs
   94 yolo
   95 route  91
   96 conv    128  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 128  0.095 BFLOPs
   97 upsample            2x    38 x  38 x 128   ->    76 x  76 x 128
   98 route  97 36
   99 conv    128  1 x 1 / 1    76 x  76 x 384   ->    76 x  76 x 128  0.568 BFLOPs
  100 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
  101 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
  102 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
  103 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
  104 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
  105 conv    255  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 255  0.754 BFLOPs
  106 yolo
```


### 从75到105层我为yolo网络的特征交互层，分为`三个尺度`，每个尺度内，通过卷积核的方式实现局部的特征交互，作用类似于全连接层但是是通过卷积核（3x3和1x1）的方式实现feature map之间的局部特征（fc层实现的是全局的特征交互）交互。

1. 最小尺度yolo层：
> 输入：`13x13`的feature map(74 res) ，一共1024个通道。
> 操作：一系列的卷积操作，feature map的大小不变，但是通道数最后减少为75个。
> 输出；输出13*13大小的feature map，`75`个通道，在此基础上进行分类和位置回归。
> VOC dataset:[tx,ty,tw,th,Objectness Score(, Class Confidences(20)]x3 = [4,1,20]x3 = 25x3 = 75.

2. 中尺度yolo层：
> 输入：将`79层`的13*13,512通道的feature map进行卷积操作,生成13x13、256通道的feature map,然后进行上采样,生成26x26 256通道的feature map,同时于`61层`的26x26、512通道的中尺度的feature map合并。再进行一系列卷积操作，
> 操作：一系列的卷积操作，feature map的大小不变，但是通道数最后减少为75个。
> 输出：26x26大小的feature map，75个通道，然后在此进行分类和位置回归。

3. 大尺度的yolo层：
> 输入：将`91层`的26x26、256通道的feature map进行卷积操作，生成26x26、128通道的feature map，然后进行上采样生成52x52、128通道的feature map，同时于`36层`的52x52、256通道的中尺度的feature map合并。再进行一系列卷积操作，
> 操作：一系列的卷积操作，feature map的大小不变，但是通道数最后减少为75个。
> 输出：52x52大小的feature map，75个通道，然后在此进行分类和位置回归。
------------------------------------------ 

## YOLOv3
- [x] [yolo系列之yolo v3 深度解析](https://blog.csdn.net/leviopku/article/details/82660381)
* yolo_v3结构图(COCO dataset):

![](https://img-blog.csdn.net/2018100917221176?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xldmlvcGt1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
> `DBL`: 如图1左下角所示，也就是代码中的Darknetconv2d_BN_Leaky，是yolo_v3的基本组件。就是卷积+BN+Leaky relu。对于v3来说，BN和leaky relu已经是和卷积层不可分离的部分了(最后一层卷积除外)，共同构成了最小组件。
> `resn`：n代表数字，有res1，res2, … ,res8等等，表示这个res_block里含有多少个res_unit。这是yolo_v3的大组件，yolo_v3开始借鉴了ResNet的残差结构，使用这种结构可以让网络结构更深(从v2的darknet-19上升到v3的darknet-53，前者没有残差结构)。对于res_block的解释，可以在图1的右下角直观看到，其基本组件也是DBL。
> `concat`：张量拼接。将darknet中间层和后面的某一层的上采样进行拼接。拼接的操作和残差层add的操作是不一样的，拼接会扩充张量的维度，而add只是直接相加不会导致张量维度的改变。
> COCO dataset:[tx,ty,tw,th,Objectness Score(, Class Confidences(80)]*3 = [4,1,80]*3 = 85*3 = 255.

* [**模型结构可视化(vis)工具**](https://github.com/lutzroeder/Netron)
* [Netron browser version](https://lutzroeder.github.io/netron/)

* 轻量化网络: SqueezeNet.
* v3毫无疑问现在成为了工程界首选的检测算法之一了，结构清晰，实时性好。
* `疑问`:输出y1,y2,y3的预测是叠加一起成为最后的输出的吗?
2019.03.13
Ans: For an image of size `416 x 416`, YOLO predicts ((52 x 52) + (26 x 26) + 13 x 13)) x 3 = `10647 bounding boxes`. 
* loss function
```python
xy_loss = object_mask * box_loss_scale * K.`binary_crossentropy`(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
wh_loss = object_mask * box_loss_scale * 0.5 * K.`square`(raw_true_wh - raw_pred[..., 2:4])
confidence_loss = object_mask * K.`binary_crossentropy`(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
class_loss = object_mask * K.`binary_crossentropy`(true_class_probs, raw_pred[..., 5:], from_logits=True)

xy_loss = K.sum(xy_loss) / mf
wh_loss = K.sum(wh_loss) / mf
confidence_loss = K.sum(confidence_loss) / mf
class_loss = K.sum(class_loss) / mf
`loss += xy_loss + wh_loss + confidence_loss + class_loss`

```
> 除了w, h的损失函数依然采用`总方误差`之外，其他部分的损失函数用的是`二值交叉熵`.

-------------------------
## YOLOv3
- [x] [基于keras-yolov3，原理及代码细节的理解](https://blog.csdn.net/KKKSQJ/article/details/83587138)
### anchor box:
* yolov3 anchor box一共有9个，由k-means聚类得到。在COCO数据集上，9个聚类是：（10x13）;（16x30）;（33x23）;（30x61）;（62x45）; （59x119）; （116x90）; （156x198）; （373x326）。
> 不同尺寸特征图对应不同大小的先验框。

    13*13feature map对应[（116*90），（156*198），（373*326）]
    26*26feature map对应[（30*61），（62*45），（59*119）]
    52*52feature map对应[（10*13），（16*30），（33*23）]

原因: 
	特征图越大，感受野越小。对小目标越敏感，所以选用小的anchor box。

    特征图越小，感受野越大。对大目标越敏感，所以选用大的anchor box。

### 边框预测：
1. 预测tx ty tw th
* 对tx和ty进行`sigmoid`，并加上对应的offset（Cx, Cy）.
* 对th和tw进行exp，并乘以对应的锚点值(pw,ph).
* 对tx,ty,th,tw乘以对应的步幅，即：416/13, 416/26, 416/52.
* 最后，使用sigmoid对Objectness和Classes confidence进行sigmoid得到0~1的概率，`之所以用sigmoid取代之前版本的softmax，原因是softmax会扩大最大类别概率值而抑制其他类别概率值`.

![边框预测](https://img2018.cnblogs.com/blog/1505200/201810/1505200-20181030204835020-902505029.png)
> (tx,ty) :目标中心点相对于该点所在网格左上角的偏移量，经过sigmoid归一化。即值属于[0,1]。如图约（0.3 , 0.4）<br>
> (cx,cy):该点所在网格的左上角距离最左上角相差的格子数。如图（1,1）<br>
> `(pw,ph):anchor box 的边长` <br>
> (tw,th):预测边框的宽和高, `maybe > 1`	<br>
> PS：最终得到的边框坐标值是bx,by,bw,bh.而网络学习目标是tx,ty,tw,th

--------------------
## YOLOv3
- [x] [YOLO从零开始：基于YOLOv3的行人检测入门指南](https://zhuanlan.zhihu.com/p/47196727)
### 通过`voc_label.py`转化`voc数据`格式为`yolo支持`的格式.
### 10、性能检测(`AlexeyAB/darknet`)
* 计算mAp
> ./darknet detector map cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc_80172.weights
* 计算recall（2097张的结果）
> ./darknet detector recall cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc_final.weights
* VOC2007test
```shell
# (会在/results生成默认的comp4_det_test_person.txt，这是在VOC2007 test上的结果)
$ ./darknet detector valid cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc_final.weights -gpu 0,1
```

### VOC的图片格式
* 行列分布同pillow.Image，先行后列

### 训练过程
* 训练迭代数：`8w iters`
* [训练技巧](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)
* [**yolov3训练的集大成者**](https://blog.csdn.net/lilai619/article/details/79695109)
```python
Region xx: 		# cfg文件中yolo-layer的索引；

Avg IOU:  		# 当前迭代中，预测的box与标注的box的平均交并比，越大越好，期望数值为1；

Class:        	# 标注物体的分类准确率，越大越好，期望数值为1；

obj:            # 越大越好，期望数值为1；

No obj:      	# 越小越好；

.5R:            # 以IOU=0.5为阈值时候的recall; recall = 检出的正样本/实际的正样本

0.75R:         	# 以IOU=0.75为阈值时候的recall;

count:        	# 正样本数目。
```
![cfg](https://img-blog.csdn.net/20180430171058652)

### YOLOv3结构图(VOC dataset):

![网络模型](https://img-blog.csdn.net/20180608100212649)

### 模型什么时候保存？
* 迭代次数小于1000时，每100次保存一次，大于1000时，没10000次保存一次。

### 图片上添加置信值
* 代码比较熟悉的童鞋，使用opencv在画框的函数里面添加一下就行了。

---------------------------
## Paper Reading
- [x] [YOLOv3 全文翻译](https://zhuanlan.zhihu.com/p/34945787)

* `Darknet` is an open source neural network framework written in `C and CUDA`. It is fast, easy to install, and supports `CPU and GPU computation`. 

* 优点：速度快，精度提升，小目标检测有改善；
* 不足：中大目标有一定程度的削弱，遮挡漏检，速度稍慢于V2。

* 哥们论文写的太随意了.

-------------------------------
## YOLOv3 train voc only person
1. 通过`ubuntu/datasets/VOC/extract_person_2007/2012.py`提取含人数据,生成文件：
```shell
$ cd /media/jun/ubuntu/datasets/VOC/
$ python extract_person_2007.py
## 'ubuntu/datasets/VOC/VOCdevkit/VOC2007/ImageSets/Main/train_person.txt'
## 'ubuntu/datasets/VOC/VOCdevkit/VOC2007/ImageSets/Main/test_person.txt'
$ python extract_person_2012.py
## 'ubuntu/datasets/VOC/VOCdevkit/VOC2012/ImageSets/Main/train_person.txt'
```

2. 通过`voc_label_person`转化voc数据格式为yolo支持的格式
```shell
$ python voc_label_person.py
## 'ubuntu/datasets/VOC/2007_train_person.txt'
## 'ubuntu/datasets/VOC/2007_test_person.txt'
## 'ubuntu/datasets/VOC/2012_train_person.txt'
```
3. 整合下训练集、测试集：
```shell
$ cat 2007_train_person.txt 2012_train_person.txt > train_person.txt
```

4. 配置`data/voc_person.names`
```python
person
```

5. 配置`cfg/voc_person.data`
```python
classes= 1
train  = /media/jun/ubuntu/datasets/VOC/train_person.txt
valid  = /media/jun/ubuntu/datasets/VOC/2007_test_person.txt
names = data/voc_person.names
backup = backup_person
```

6. 配置`cfg/yolov3-voc-person.cfg`
```python
# 1. 一共三处
# filters=75
`filters=18`
activation=linear

[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
`classes=1`

# 2. 迭代次数
max_batches = 80200

# 3. uncomment the 'Training' parameters.
# Testing
# batch=1
# subdivisions=1
# Training
 batch=64
 subdivisions=16
```

7. 训练
```shell
## Train
$ ./darknet detector train cfg/voc-person.data cfg/yolov3-voc-person.cfg weights/darknet53.conv.74 -gpus 0,1 |tee -a backup_person/train_voc.txt

## Restart training from a checkpoint:
$ ./darknet detector train cfg/voc-person.data cfg/yolov3-voc-person.cfg backup_person/yolov3-voc-person.backup -gpus 0,1

## Test NetWork
# Modify 'cfg/yolov3-voc-person.cfg'
# Testing
 batch=1
 subdivisions=1
$ ./darknet detector test cfg/voc-person.data cfg/yolov3-voc-person.cfg backup_person/yolov3-voc-person.backup data/person6.jpg
```
## 训练效果很好！！！
8. 用脚本`analyse.py`对`训练日志train7-loss.txt`的训练过程可视化(`AlexeyAB/darknet`)。


