import cv2  # 导入 OpenCV 作为 cv2，主要用于访问摄像头和处理图像
import tkinter as tk  # 导入 tkinter 作为 tk，用于创建图形用户界面
from tkinter import filedialog  # 用于在界面上打开文件对话框
from PIL import Image, ImageTk  # 用于将 OpenCV 处理的图像转换为 tkinter 图像
from ultralytics import YOLO  # 导入 YOLO 对象侦测模型
import numpy as np  # 导入 numpy 作为 np，用于处理数组
import logging  # 导入日志模块，方便查看程序运行状态
import utils # 导入 utils.py 文件，用于初始化模型和处理图像
import os # 导入 os 模块，用于获取文件路径

# 设置日志
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# 创建应用类，用于创建界面


class Application:
    # 初始化应用类，设置界面窗口和标题，以及初始化相关参数、UI组件和事件处理函数
    def __init__(self, window, window_title):
        self.detecting = False  # 默认检测标志为 False，表示不进行目标检测
        logging.info("初始化应用界面")

        self.window = window  # 设置界面窗口
        self.window.title(window_title)  # 设置界面标题

        self.vid = None  # 初始化摄像头或者视频对象为 None
        self.running = True  # 设置运行标志为 True
        self.is_camera = True  # 标识当前输入流是否为摄像头
        self.file_path = None  # 存储视频文件路径

        logging.info(
            "创建单选钮确定模型的类型['定位', '分割', '姿势']和大小['n', 's', 'm', 'l', 'x']")
        # 在给定的窗口（window）上创建一个新的框架（Frame），并赋值给变量 top_frame=
        top_frame = tk.Frame(window)
        top_frame.grid(row=0, column=0)  # 将这个框架放在窗口的网格布局的第 0 行第 0 列
        # 定义一个列表，用于存储第一组单选按钮（RadioButton）的选项
        self.model_opts1 = ['定位', '分割', '姿势']
        self.model_opts2 = ['n', 's', 'm', 'l', 'x']  # 定义一个列表，用于存储第二组单选按钮的选项

        # 定义一个字符串变量，并赋予初值为第一组单选按钮的第一个选项
        self.model_var1 = tk.StringVar(value=self.model_opts1[0])
        # 定义一个字符串变量，并赋予初值为第二组单选按钮的第一个选项
        self.model_var2 = tk.StringVar(value=self.model_opts2[0])

        for idx, opt in enumerate(self.model_opts1):  # 对于第一组单选按钮的每个选项，获取它的索引和值
            tk.Radiobutton(top_frame, text=opt, variable=self.model_var1, value=opt,
                           command=self.change_model).grid(row=0, column=idx)
            # 创建一个单选按钮，并放置在 top_frame 上，显示文本为选项值，变量关联为 self.model_var1，按钮的值为选项值，
            # 当按钮被点击时，调用 self.change_model 方法，然后使用 grid 方法将按钮放在 top_frame 的第 0 行 idx 列

        for idx, opt in enumerate(self.model_opts2):  # 对于第二组单选按钮的每个选项，获取它的索引和值
            tk.Radiobutton(top_frame, text=opt, variable=self.model_var2, value=opt,
                           command=self.change_model).grid(row=1, column=idx)
            # 创建一个单选按钮，并放置在 top_frame 上，显示文本为选项值，变量关联为 self.model_var2，按钮的值为选项值，
            # 当按钮被点击时，调用 self.change_model 方法，然后使用 grid 方法将按钮放在 top_frame 的第 1 行 idx 列

        logging.info("初始化 YOLOv8 模型")
        self.change_model()

        # 创建按钮
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

        # 在底部创建一个用于按钮的框架
        logging.info("创建底部按钮框架")
        bottom_frame = tk.Frame(window)
        bottom_frame.grid(row=3, column=0)

        # 创建一个进度条，用于控制和显示视频播放进度
        self.scale_var = tk.DoubleVar()  # 创建一个双精度浮点数类型的变量，用于存储进度条的当前值
        self.progress_bar = tk.Scale(bottom_frame, variable=self.scale_var,
                                     orient='horizontal', length=500, sliderlength=10, showvalue=False)
        self.progress_bar.pack(side=tk.RIGHT)  # 按钮放入到底部框架的右侧
        # 绑定鼠标左键释放事件到 set_video_position 方法
        self.progress_bar.bind("<ButtonRelease-1>", self.set_video_position)
        # 绑定鼠标左键拖动事件到 set_video_position 方法
        self.progress_bar.bind("<B1-Motion>", self.set_video_position)

        # 控制按钮
        logging.info("创建 '暂停', '播放' 和 '识别切换' 按钮")
        self.pause_button = tk.Button(
            bottom_frame, text="暂停", command=self.pause)  # 创建暂停按钮并绑定 pause 方法
        self.pause_button.pack(side=tk.LEFT)  # 按钮放入到底部框架的左侧

        self.play_button = tk.Button(
            bottom_frame, text="播放", command=self.play)  # 创建播放按钮并绑定 play 方法
        self.play_button.pack(side=tk.LEFT)  # 按钮放入到底部框架的左侧

        # 创建重新播放按钮并绑定 replay 方法
        self.replay_button = tk.Button(
            bottom_frame, text="重新播放", command=self.replay)
        self.replay_button.pack(side=tk.LEFT)  # 按钮放入到底部框架的左侧

        # 创建目标检测按钮并绑定 detect_objects 方法
        self.detect_button = tk.Button(
            bottom_frame, text="识别切换", command=self.detect_objects)
        self.detect_button.pack(side=tk.LEFT)  # 按钮放入到底部框架的左侧

        self.delay = 15  # 设置界面更新时间间隔（毫秒）
        self.update()  # 开始更新界面

        self.window.mainloop()  # 进入界面主循环

    # 更改模型并初始化 YOLO 模型，根据界面上的单选按钮改变当前使用的目标检测模型类型和大小，
    # 并设置是否在视频上显示边界框和遮罩的标志
    def change_model(self):
        # 获取第一组单选按钮的当前值（模型类型）
        model_opt1 = self.model_var1.get()
        # 获取第二组单选按钮的当前值（模型大小）
        model_opt2 = self.model_var2.get()

        # TODO: 获取选择的模型内容构建模型的名称 model_name
        # TODO: 根据 model_name 构建模型完整路径
        
        self.model = # TODO: 利用utils函数获取模型
        # 用日志记录当前更改的模型名称
        logging.info(f"更改模型为 {model_name}")

    # 尝试打开摄像头，创建摄像头对象并将运行标志和摄像头标志设置为 True，
    # 表示更新画布时将显示摄像头提供的视频流
    def open_camera(self):  # 尝试打开摄像头的方法
        logging.info("尝试打开摄像头")
        self.vid = cv2.VideoCapture(0)  # 初始化摄像头对象
        self.running = True  # 设置运行标志为 True
        self.is_camera = True  # 当打开摄像头时，将标识设置为 True

    # 尝试打开视频文件，创建视频文件对象并将运行标志设置为 True，摄像头标志设置为 False，
    # 表示更新画布时将显示视频文件的内容
    def open_video(self):  # 尝试打开视频文件的方法
        logging.info("尝试打开视频文件")
        self.file_path = filedialog.askopenfilename()  # 弹出文件对话框选择文件
        self.vid = cv2.VideoCapture(self.file_path)  # 初始化视频文件对象
        self.running = True  # 设置运行标志为 True
        self.is_camera = False  # 当打开视频文件时，将标识设置为 False

        # 获取视频总帧数，设置进度条的最大值
        self.total_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.progress_bar.config(to=self.total_frames)

    # 暂停视频播放，将运行标志设置为 False，使得更新画布时不更新视频帧
    def pause(self):  # 暂停视频播放的方法
        logging.info("暂停视频播放")
        self.running = False  # 设置运行标志为 False

    # 播放视频或摄像头，将运行标志设置为 True，使得更新画布时更新视频帧
    def play(self):  # 播放视频或摄像头的方法
        logging.info("播放视频")
        self.running = True  # 设置运行标志为 True

    # 根据进度条设置视频播放位置，仅当当前输入流为视频文件时生效，
    # 通过获取进度条上显示的当前帧数来设置视频文件播放的位置
    def set_video_position(self, event):
        self.running = False  # 设置运行标志为 False，这样在拖动的时候就不会一直动了
        if not self.is_camera and self.file_path is not None:  # 只有当当前输入流不是摄像头且视频文件路径不为空时，才可以设置视频位置
            frame_pos = self.scale_var.get()  # 获取进度条当前值
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)  # 设置视频位置
            ret, frame = self.vid.read()  # 读取视频当前帧
            self.display_frame(ret, frame)

    # 将当前帧显示在画布上，将输入的视频帧进行格式转换并根据检测标志进行目标检测，
    # 将处理后的视频帧显示在应用界面的画布上
    def display_frame(self, ret, frame):
        if self.vid is not None:
            # ret, frame = self.vid.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = frame.shape
                new_width = 800
                new_height = int(new_width * (height / width))
                frame = cv2.resize(frame, (new_width, new_height))
                if self.detecting:
                    frame = # TODO: 仔细阅读本文件中的代码，利用 utils 中相关函数对视频帧进行处理
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    # 重新播放视频文件，仅当当前输入流为视频文件时生效，重新初始化视频文件对象，
    # 并将运行标志设置为 True，使得更新画布时更新视频帧
    def replay(self):
        if not self.is_camera and self.file_path is not None:  # 只有当当前输入流不是摄像头且视频文件路径不为空时，才可以重新播放
            logging.info("重新播放视频文件")
            self.vid = cv2.VideoCapture(self.file_path)  # 重新初始化视频文件对象
            self.running = True  # 设置运行标志为 True

    # 开始或停止目标检测，点击识别切换按钮时，改变检测目标的标志位，
    # 当检测标志为 True 时，在视频帧上进行目标检测并显示检测结果
    def detect_objects(self):  # 开始或停止目标检测的方法
        logging.info("点击 'Detect'，开始/停止目标检测")
        self.detecting = not self.detecting  # 点击 Detect 时，改变 detecting 的布尔值

    # 更新画布显示的视频帧，根据运行标志、摄像头/视频文件对象来获取和处理视频帧，
    # 将处理后的视频帧显示在画布上，并根据设置的时间间隔递归调用自身实现视频流的更新
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
