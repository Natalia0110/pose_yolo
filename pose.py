import cv2
import csv
import os
import time
import torch
from pathlib import Path
from ultralytics import YOLO


# =========================
# 1. 配置区
# =========================

# 本地测试视频路径
VIDEO_SOURCE = r"d:\xwechat_files\wxid_vhzos171qdg622_83f0\temp\RWTemp\2026-05\d5fc77a82b9caaf3edb9a5706eff1443\ffb6b5151f87b504fd59e832795b4479.mp4"

# 海康 RTSP 摄像头，使用时把上面 VIDEO_SOURCE 注释掉，启用下面这一行
# VIDEO_SOURCE = "rtsp://admin:你的密码@192.168.0.64:554/Streaming/Channels/102"

# 训练完成后的模型路径
MODEL_PATH = r"D:/yolov8/runs/swim_pose/underwater_yolov8n_pose/weights/best.pt"

# 输出文件夹
OUTPUT_DIR = r"D:/yolov8/pose_output_trained"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输出视频和 CSV
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "trained_model_annotated_video.mp4")
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "trained_model_keypoints_timeseries.csv")

# 置信度阈值
CONF_THRES = 0.25

# 是否显示实时画面
SHOW_WINDOW = True

# 是否保存标注后的视频
SAVE_VIDEO = True

# 使用 GPU
# 你现在 CUDA 已经正常，所以这里用 0
DEVICE = 0

# COCO 17 个关键点名称
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


# =========================
# 2. 检查 GPU
# =========================

print("========== 运行环境检查 ==========")
print("PyTorch 版本：", torch.__version__)
print("CUDA 是否可用：", torch.cuda.is_available())
print("CUDA 设备数量：", torch.cuda.device_count())

if torch.cuda.is_available():
    print("当前 GPU：", torch.cuda.get_device_name(0))
else:
    print("警告：当前没有检测到 CUDA，将无法使用 GPU")
    DEVICE = "cpu"


# =========================
# 3. 检查模型文件
# =========================

if not os.path.exists(MODEL_PATH):
    print("没有找到 best.pt：")
    print(MODEL_PATH)

    print("\n正在 D:/yolov8 下自动搜索 best.pt ...")
    best_files = list(Path(r"D:/yolov8").rglob("best.pt"))

    if len(best_files) == 0:
        print("没有搜索到 best.pt，请确认你已经运行过 train_swim_pose.py")
        exit()

    print("搜索到以下 best.pt：")
    for i, p in enumerate(best_files):
        print(f"{i}: {p}")

    MODEL_PATH = str(best_files[0])
    print("\n默认使用第一个模型：")
    print(MODEL_PATH)


# =========================
# 4. 加载训练后的模型
# =========================

print("\n========== 加载模型 ==========")
print("使用模型：", MODEL_PATH)

model = YOLO(MODEL_PATH)

print("模型加载完成")


# =========================
# 5. 打开视频 / 摄像头
# =========================

print("\n========== 打开视频源 ==========")
print("视频源：", VIDEO_SOURCE)

cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("视频或摄像头打开失败")
    print("请检查：视频路径、RTSP地址、IP、用户名、密码、网络连接")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0 or fps > 120:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"视频打开成功：{width}x{height}, FPS={fps}, 总帧数={total_frames}")


# =========================
# 6. 创建视频保存器
# =========================

video_writer = None

if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        fourcc,
        fps,
        (width, height)
    )

    if not video_writer.isOpened():
        print("视频保存器创建失败")
        cap.release()
        exit()

    print("标注视频将保存到：", OUTPUT_VIDEO_PATH)


# =========================
# 7. 创建 CSV 文件
# =========================

csv_file = open(OUTPUT_CSV_PATH, mode="w", newline="", encoding="utf-8-sig")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame_id",
    "time_sec",
    "wall_time_sec",
    "person_id",
    "keypoint_id",
    "keypoint_name",
    "x",
    "y",
    "confidence",
    "image_width",
    "image_height",
    "model_path"
])

print("关节点 CSV 将保存到：", OUTPUT_CSV_PATH)


# =========================
# 8. 主循环：检测 + 显示 + 保存视频 + 保存 CSV
# =========================

start_time = time.time()
frame_id = 0

print("\n========== 开始检测 ==========")
print("按 q 退出")

while True:
    ret, frame = cap.read()

    if not ret:
        print("视频读取结束或读取失败")
        break

    time_sec = frame_id / fps
    wall_time_sec = time.time() - start_time

    # 使用训练后的 best.pt 进行 YOLOv8-Pose 推理
    results = model.predict(
        source=frame,
        conf=CONF_THRES,
        device=DEVICE,
        verbose=False
    )

    result = results[0]

    # 画出检测框、关节点和骨架
    annotated_frame = result.plot()

    # 保存标注视频
    if SAVE_VIDEO and video_writer is not None:
        video_writer.write(annotated_frame)

    # 保存关节点到 CSV
    if result.keypoints is not None and result.keypoints.xy is not None:
        keypoints_xy = result.keypoints.xy.cpu().numpy()

        if result.keypoints.conf is not None:
            keypoints_conf = result.keypoints.conf.cpu().numpy()
        else:
            keypoints_conf = None

        # 没检测到人时，keypoints_xy 可能长度为 0
        if len(keypoints_xy) == 0:
            csv_writer.writerow([
                frame_id,
                round(time_sec, 4),
                round(wall_time_sec, 4),
                -1,
                -1,
                "no_person_detected",
                -1,
                -1,
                -1,
                width,
                height,
                MODEL_PATH
            ])
        else:
            for person_id, person_kpts in enumerate(keypoints_xy):
                for keypoint_id, (x, y) in enumerate(person_kpts):
                    if keypoints_conf is not None:
                        conf = float(keypoints_conf[person_id][keypoint_id])
                    else:
                        conf = -1.0

                    keypoint_name = KEYPOINT_NAMES[keypoint_id]

                    csv_writer.writerow([
                        frame_id,
                        round(time_sec, 4),
                        round(wall_time_sec, 4),
                        person_id,
                        keypoint_id,
                        keypoint_name,
                        round(float(x), 2),
                        round(float(y), 2),
                        round(conf, 4),
                        width,
                        height,
                        MODEL_PATH
                    ])
    else:
        csv_writer.writerow([
            frame_id,
            round(time_sec, 4),
            round(wall_time_sec, 4),
            -1,
            -1,
            "no_person_detected",
            -1,
            -1,
            -1,
            width,
            height,
            MODEL_PATH
        ])

    # 实时写入 CSV，方便后续分析程序读取
    csv_file.flush()

    # 显示实时画面
    if SHOW_WINDOW:
        cv2.imshow("Trained YOLOv8 Pose Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("手动退出")
            break

    frame_id += 1


# =========================
# 9. 释放资源
# =========================

cap.release()

if video_writer is not None:
    video_writer.release()

csv_file.close()
cv2.destroyAllWindows()

print("\n========== 处理完成 ==========")
print(f"使用模型：{MODEL_PATH}")
print(f"标注视频保存位置：{OUTPUT_VIDEO_PATH}")
print(f"关节点 CSV 保存位置：{OUTPUT_CSV_PATH}")