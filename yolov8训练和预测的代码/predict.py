import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO('./runs/detect/train7/weights/best.pt')

# 打开视频
video_path = "./深度学习任务二测试视频.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频帧的维度和速率
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# 创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./output.mp4', fourcc, fps, (frame_width, frame_height))

# 循环视频帧
while cap.isOpened():
    # 读取帧
    success, frame = cap.read()
    if success:
        # 预测
        results = model(frame)
         # 可视化结果
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
    else:
     # 最后结尾中断视频帧循环
     break
        
cap.release()
out.release()