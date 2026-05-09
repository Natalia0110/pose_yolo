from ultralytics import YOLO


# ============================================================
# 1. 模型路径
# ============================================================

# YOLOv8l-Pose 预训练模型
# 如果本地没有这个文件，Ultralytics 一般会自动下载
PRETRAINED_MODEL = r"D:/yolov8/yolov8l-pose.pt"

# 数据集配置文件
DATA_YAML = r"D:/yolov8/dataset/swim_pose/swim_pose.yaml"


# ============================================================
# 2. 加载预训练模型
# ============================================================

model = YOLO(PRETRAINED_MODEL)


# ============================================================
# 3. 开始训练
# ============================================================

model.train(
    data=DATA_YAML,

    # YOLOv8l 比 n 大很多，训练更慢
    epochs=5,

    imgsz=640,

    # 重点：YOLOv8l 显存占用更高
    # RTX 4060 Laptop 8GB 建议先用 batch=2
    # 如果爆显存，再改成 batch=1
    batch=2,

    # 使用 GPU
    device=0,

    # Windows 下建议 0
    workers=0,

    # 迁移学习学习率
    lr0=0.001,

    pretrained=True,

    # 自动早停
    # 连续 10 轮验证集指标没提升就停止
    patience=10,

    # 保存目录
    project=r"D:/yolov8/runs/swim_pose",

    # 本次训练名称，和 n 模型区分开
    name="underwater_yolov8l_pose",

    exist_ok=True,
    save=True,
    verbose=True,
    plots=True
)


print("YOLOv8l-Pose 训练完成！")
print("训练好的模型一般在：")
print(r"D:/yolov8/runs/swim_pose/underwater_yolov8l_pose/weights/best.pt")