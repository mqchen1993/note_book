# labelme 

# 1. annotate
labelme runway/ --labels labels_kitti_road.txt --nodata

# 2. labelme2kitti
$ python labelme2kitti_road.py labels_kitti_road.txt runway/ kitti/
