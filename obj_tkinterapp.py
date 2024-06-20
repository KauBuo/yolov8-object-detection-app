import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
import logging
import utils
import os

# 设置日志
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# 创建应用类，用于创建界面
class Application:
    def __init__(self, window, window_title):
        self.detecting = False  # 默认检测标志为 False，表示不进行目标检测
        logging.info("初始化应用界面")

        self.window = window  # 设置界面窗口
        self.window.title(window_title)  # 设置界面标题

        self.vid = None  # 初始化摄像头或者视频对象为 None
        self.running = True  # 设置运行标志为 True
        self.is_camera = True  # 标识当前输入流是否为摄像头
        self.file_path = None  # 存储视频文件路径

        # 添加 show_box 和 show_mask 的默认值
        self.show_box = True  # 是否显示边框
        self.show_mask = False  # 是否显示遮罩

        logging.info("创建单选钮确定模型的类型['定位', '分割', '姿势']和大小['n', 's', 'm', 'l', 'x']")
        top_frame = tk.Frame(window)
        top_frame.grid(row=0, column=0)
        self.model_opts1 = ['定位', '分割', '姿势']
        self.model_opts2 = ['n', 's', 'm', 'l', 'x']

        self.model_var1 = tk.StringVar(value=self.model_opts1[0])
        self.model_var2 = tk.StringVar(value=self.model_opts2[0])

        for idx, opt in enumerate(self.model_opts1):
            tk.Radiobutton(top_frame, text=opt, variable=self.model_var1, value=opt,
                           command=self.change_model).grid(row=0, column=idx)

        for idx, opt in enumerate(self.model_opts2):
            tk.Radiobutton(top_frame, text=opt, variable=self.model_var2, value=opt,
                           command=self.change_model).grid(row=1, column=idx)

        logging.info("初始化 YOLOv8 模型")
        self.change_model()

        logging.info("创建 '读取像头' 和 '读取视频' 按钮")
        self.camera_button = tk.Button(
            top_frame, text="读取像头", command=self.open_camera)
        self.camera_button.grid(row=0, column=3)

        self.video_button = tk.Button(
            top_frame, text="读取视频", command=self.open_video)
        self.video_button.grid(row=0, column=4)

        logging.info("创建画布用于展示视频")
        self.canvas = tk.Canvas(window, width=800, height=600)
        self.canvas.grid(row=2, column=0)

        logging.info("创建底部按钮框架")
        bottom_frame = tk.Frame(window)
        bottom_frame.grid(row=3, column=0)

        self.scale_var = tk.DoubleVar()
        self.progress_bar = tk.Scale(bottom_frame, variable=self.scale_var,
                                     orient='horizontal', length=500, sliderlength=10, showvalue=False)
        self.progress_bar.pack(side=tk.RIGHT)
        self.progress_bar.bind("<ButtonRelease-1>", self.set_video_position)
        self.progress_bar.bind("<B1-Motion>", self.set_video_position)

        logging.info("创建 '暂停', '播放' 和 '识别切换' 按钮")
        self.pause_button = tk.Button(
            bottom_frame, text="暂停", command=self.pause)
        self.pause_button.pack(side=tk.LEFT)

        self.play_button = tk.Button(
            bottom_frame, text="播放", command=self.play)
        self.play_button.pack(side=tk.LEFT)

        self.replay_button = tk.Button(
            bottom_frame, text="重新播放", command=self.replay)
        self.replay_button.pack(side=tk.LEFT)

        self.detect_button = tk.Button(
            bottom_frame, text="识别切换", command=self.detect_objects)
        self.detect_button.pack(side=tk.LEFT)

        self.delay = 30  # 增加界面更新时间间隔（毫秒），避免频繁更新
        self.update()  # 开始更新界面

        self.window.mainloop()  # 进入界面主循环

    def change_model(self):
        # 获取选择的模型类型（定位、分割、姿势）和大小（n, s, m, l, x）
        model_opt1 = self.model_var1.get()
        model_opt2 = self.model_var2.get()

        # 根据选择的模型类型和大小，构建模型文件的完整路径
        model_type_prefix = {
            '定位': f'yolov8{model_opt2}',        # 检测模型如 yolov8n.pt
            '分割': f'yolov8{model_opt2}-seg',   # 分割模型如 yolov8n-seg.pt
            '姿势': f'yolov8{model_opt2}-pose'  # 姿势估计模型如 yolov8n-pose.pt
        }

        # 构建模型文件路径
        model_name = f"models/{model_type_prefix[model_opt1]}.pt"
        
        # 检查模型文件是否存在
        if not os.path.isfile(model_name):
            logging.error(f"模型文件 {model_name} 不存在")
            return

        # 初始化模型
        self.model = utils.init_model(model_name)

        # 记录当前更改的模型名称
        logging.info(f"更改模型为 {model_name}")

    def open_camera(self):
        logging.info("尝试打开摄像头")
        self.vid = cv2.VideoCapture(0)  # 初始化摄像头对象
        self.running = True  # 设置运行标志为 True
        self.is_camera = True  # 当打开摄像头时，将标识设置为 True

    def open_video(self):
        logging.info("尝试打开视频文件")
        self.file_path = filedialog.askopenfilename()  # 弹出文件对话框选择文件
        self.vid = cv2.VideoCapture(self.file_path)  # 初始化视频文件对象
        self.running = True  # 设置运行标志为 True
        self.is_camera = False  # 当打开视频文件时，将标识设置为 False

        # 获取视频总帧数，设置进度条的最大值
        self.total_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.progress_bar.config(to=self.total_frames)

    def pause(self):
        logging.info("暂停视频播放")
        self.running = False  # 设置运行标志为 False

    def play(self):
        logging.info("播放视频")
        self.running = True  # 设置运行标志为 True

    def set_video_position(self, event):
        self.running = False
        if not self.is_camera and self.file_path is not None:
            frame_pos = self.scale_var.get()
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = self.vid.read()
            self.display_frame(ret, frame)

    # 将当前帧显示在画布上，将输入的视频帧进行格式转换并根据检测标志进行目标检测，
    # 将处理后的视频帧显示在应用界面的画布上
    def display_frame(self, ret, frame):
        if self.vid is not None and ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = frame.shape
            new_width = 800
            new_height = int(new_width * (height / width))
            frame = cv2.resize(frame, (new_width, new_height))

            if self.detecting and self.model is not None:
                # 处理视频帧，根据模型类型处理框
                frame = utils.process_frame(self.model, frame, show_box=self.show_box, show_mask=self.show_mask)

                # 如果是分割或姿势模型，去掉类别为 "person" 的框
                if "seg" in self.model.model.names or "pose" in self.model.model.names:
                    frame = self.filter_person_boxes(frame)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def filter_person_boxes(self, frame):
        results = self.model(frame, conf=0.25, iou=0.7)

        if "seg" in self.model.model.names:
            processed_frame = results[0].plot(polygon=True)
        elif "pose" in self.model.model.names:
            processed_frame = results[0].plot()
        else:
            processed_frame = frame  # 对于普通检测模型，直接返回原始帧

        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)  # 转回 BGR 以便显示
        return processed_frame



    def replay(self):
        if not self.is_camera and self.file_path is not None:
            logging.info("重新播放视频文件")
            self.vid = cv2.VideoCapture(self.file_path)
            self.running = True

    def detect_objects(self):
        logging.info("点击 '识别切换'，开始/停止目标检测")
        self.detecting = not self.detecting  # 切换目标检测状态

    def update(self):
        if self.vid is not None and self.running:
            ret, frame = self.vid.read()
            if ret:
                frame_pos = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
                self.scale_var.set(frame_pos)
                self.display_frame(ret, frame)
            else:
                logging.info("视频播放完成")
                self.running = False
        self.window.after(self.delay, self.update)

# 创建一个窗口并将其传递给 Application 对象
App = Application(tk.Tk(), "Tkinter and OpenCV")