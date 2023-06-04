# 引入cv2用于视频处理
import cv2
# 引入YOLO模型用于目标检测
from ultralytics import YOLO

# 加载YOLO模型，进行目标检测
model = YOLO('../weights/yolov8n-seg.pt')

# 打开视频文件，这里的路径需要自行替换
video_path = "../assets/people_walking_1.mp4"
cap = cv2.VideoCapture(video_path)

# 循环读取视频帧
while cap.isOpened():
    # 读取一帧图像
    success, frame = cap.read()

    if success:
        # 对读取的帧进行目标检测
        results = model(frame)

        # 可视化检测结果，这里设置置信度阈值为0.5，不显示框，显示掩码和概率
        annotated_frame = results[0].plot(conf=0.5, boxes=False, masks=True, probs=True)

        # 显示标注后的帧
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # 如果按下“q”键，跳出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束，跳出循环
        break

# 释放视频捕捉对象，并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
