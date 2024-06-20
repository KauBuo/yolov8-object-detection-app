import cv2
from ultralytics import YOLO

def init_model(model_path):
    model = YOLO(model_path)
    return model

def process_frame(model, frame, show_box=True, show_mask=False):
    """处理视频帧，进行目标检测、分割或姿势估计"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 确保颜色空间转换为 RGB
    results = model(frame_rgb, conf=0.25, iou=0.7)

    if "seg" in model.model.names:
        processed_frame = results[0].plot(polygon=True)
    elif "pose" in model.model.names:
        processed_frame = results[0].plot()
    else:
        processed_frame = results[0].plot()

    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)  # 转回 BGR 以便显示
    return processed_frame