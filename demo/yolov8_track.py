# 导入ultralytics库，这是一个开源库，用于对象检测和跟踪
from ultralytics import YOLO 

# 通过加载预训练模型'yolov8n.pt'创建一个YOLO对象检测模型实例，YOLO是一种深度学习的对象检测算法
model = YOLO('../weights/yolov8n.pt')

# 使用YOLO模型的track方法，对源视频'people_walking_1.mp4'进行对象跟踪
# conf参数为0.3，表示只有当对象检测的置信度高于0.3时，才会被认为是有效的检测结果
# iou参数为0.5，表示在对象跟踪过程中，只有当两个检测框的交并比（Intersection over Union）高于0.5时，才会被认为是同一个对象的检测结果
# show参数为True，表示在进行对象跟踪的过程中，将实时显示出视频中的对象检测和跟踪结果
model.track(source="../assets/people_walking_1.mp4", conf=0.3, iou=0.5, show=True) 
