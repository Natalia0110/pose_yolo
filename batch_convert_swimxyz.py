import cv2
import os
import random
import math
import shutil
from pathlib import Path


# ============================================================
# 1. 路径配置
# ============================================================

RAW_DIR = Path(r"D:/yolov8/raw_swimxyz")
DATASET_DIR = Path(r"D:/yolov8/dataset/swim_pose")


# ============================================================
# 2. 关键参数
# ============================================================

# 你已经核对过 offset = -1 是正确的
# 视频第 n 帧 → 标注第 n-1 行
LABEL_FRAME_OFFSET = -1

# 你的视频和标注是上下镜像，所以只翻转 y 坐标
MIRROR_X = False
MIRROR_Y = True

# 每隔多少帧抽一张
# 5 表示每隔 5 帧保存一张
FRAME_INTERVAL = 5

# 每个视频最多抽多少张
MAX_FRAMES_PER_VIDEO = 400

# JPG 图片质量
JPEG_QUALITY = 85

# train / val / test 划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 如果之前生成过错误数据，第一次重新跑建议改成 True
# 跑完后再改回 False，避免误删
CLEAR_OLD_OUTPUT = True

random.seed(42)


# ============================================================
# 3. YOLOv8-Pose 使用的 COCO17 关键点顺序
# ============================================================

COCO17_FROM_SWIMXYZ = [
    "Nose",
    "LEye",
    "REye",
    "LEar",
    "REar",
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "LWrist",
    "RWrist",
    "LHip",
    "RHip",
    "LKnee",
    "RKnee",
    "LAnkle",
    "RAnkle",
]

FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


# ============================================================
# 4. 工具函数
# ============================================================

def parse_number(value: str) -> float:
    """
    SwimXYZ 里小数可能是 386,05
    这里转换成 Python 能识别的 386.05
    """
    value = value.strip()

    if value == "":
        return float("nan")

    value = value.replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return float("nan")


def safe_name(name: str) -> str:
    """
    把文件夹名转换成安全的文件名前缀。
    """
    return "".join(
        c if c.isalnum() or c in ["_", "-"] else "_"
        for c in name
    )


def valid_point(x, y, img_w, img_h) -> bool:
    """
    判断关键点是否有效。
    """
    if x is None or y is None:
        return False

    if math.isnan(x) or math.isnan(y):
        return False

    if x <= 0 or y <= 0:
        return False

    if x >= img_w or y >= img_h:
        return False

    return True


def transform_keypoints(keypoints, img_w, img_h):
    """
    对标注点做镜像修正。

    MIRROR_X = True：左右镜像
    MIRROR_Y = True：上下镜像

    你现在的情况是上下镜像，所以：
    MIRROR_X = False
    MIRROR_Y = True
    """
    transformed = []

    for x, y in keypoints:
        if x is None or y is None:
            transformed.append([x, y])
            continue

        if math.isnan(x) or math.isnan(y):
            transformed.append([x, y])
            continue

        if MIRROR_X:
            x = img_w - 1 - x

        if MIRROR_Y:
            y = img_h - 1 - y

        transformed.append([x, y])

    return transformed


# ============================================================
# 5. 读取 SwimXYZ 的 2D_cam.txt
# ============================================================

def read_swimxyz_2d_cam(label_path: Path):
    """
    读取 2D_cam.txt。

    兼容你的数据情况：
    - 第一行表头是 75 个字段
    - 实际每一帧可能是 54 个字段
    - 54 个字段 = 18 个点 * x,y,z
    - 缺失的关键点会自动设为 nan
    - 后面生成 YOLO 标签时会把 nan 点设为 visibility=0
    """

    frames = []

    with open(label_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    if len(lines) < 2:
        print(f"警告：{label_path} 内容太少，跳过")
        return frames

    full_header = lines[0].strip().split(";")
    full_header = [h.strip() for h in full_header if h.strip() != ""]

    print(f"读取标注文件：{label_path}")
    print(f"表头字段数：{len(full_header)}")

    for line_idx, line in enumerate(lines[1:]):
        line = line.strip()

        if not line:
            frames.append(None)
            continue

        parts = line.split(";")
        parts = [p.strip() for p in parts if p.strip() != ""]

        if len(parts) < 30:
            print(f"警告：第 {line_idx} 帧字段太少，实际 {len(parts)}，跳过")
            frames.append(None)
            continue

        # 关键点：
        # 表头可能是75列，但实际数据只有54列
        # 所以只用实际存在的字段数量来对应表头
        usable_header = full_header[:len(parts)]

        point_data = {}

        for field_name, raw_value in zip(usable_header, parts):
            if "." not in field_name:
                continue

            point_name, axis_name = field_name.rsplit(".", 1)

            if point_name not in point_data:
                point_data[point_name] = {}

            point_data[point_name][axis_name] = parse_number(raw_value)

        coco17_keypoints = []

        for point_name in COCO17_FROM_SWIMXYZ:
            data = point_data.get(point_name, {})

            x = data.get("x", float("nan"))
            y = data.get("y", float("nan"))

            coco17_keypoints.append([x, y])

        frames.append(coco17_keypoints)

    return frames


# ============================================================
# 6. 根据关键点计算人体框 bbox
# ============================================================

def compute_bbox_from_keypoints(keypoints, img_w, img_h):
    """
    YOLOv8-Pose 的标签必须有 bbox。
    这里根据可见关键点自动计算人体框。
    """

    valid_points = []

    for x, y in keypoints:
        if valid_point(x, y, img_w, img_h):
            valid_points.append([x, y])

    if len(valid_points) < 4:
        return None

    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]

    x_min = max(0, min(xs))
    y_min = max(0, min(ys))
    x_max = min(img_w - 1, max(xs))
    y_max = min(img_h - 1, max(ys))

    # 加一点边距，避免框太紧
    pad_x = (x_max - x_min) * 0.15
    pad_y = (y_max - y_min) * 0.15

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(img_w - 1, x_max + pad_x)
    y_max = min(img_h - 1, y_max + pad_y)

    box_w = x_max - x_min
    box_h = y_max - y_min

    if box_w <= 1 or box_h <= 1:
        return None

    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    box_w = box_w / img_w
    box_h = box_h / img_h

    return x_center, y_center, box_w, box_h


# ============================================================
# 7. 生成 YOLOv8-Pose 标签
# ============================================================

def make_yolo_pose_label(coco17_keypoints, img_w, img_h):
    """
    YOLOv8-Pose 标签格式：

    class x_center y_center width height
    kpt1_x kpt1_y v1
    kpt2_x kpt2_y v2
    ...
    kpt17_x kpt17_y v17

    所有坐标都必须归一化到 0~1。
    """

    bbox = compute_bbox_from_keypoints(coco17_keypoints, img_w, img_h)

    if bbox is None:
        return None

    x_center, y_center, box_w, box_h = bbox

    values = [
        "0",
        f"{x_center:.6f}",
        f"{y_center:.6f}",
        f"{box_w:.6f}",
        f"{box_h:.6f}",
    ]

    for x, y in coco17_keypoints:
        if valid_point(x, y, img_w, img_h):
            x_norm = x / img_w
            y_norm = y / img_h
            visibility = 2
        else:
            x_norm = 0.0
            y_norm = 0.0
            visibility = 0

        values.extend([
            f"{x_norm:.6f}",
            f"{y_norm:.6f}",
            str(visibility)
        ])

    return " ".join(values)


# ============================================================
# 8. 查找所有 sample 中的 .webm + 2D_cam.txt
# ============================================================

def find_video_label_pairs(raw_dir: Path):
    """
    自动扫描 raw_swimxyz 目录。

    支持这种结构：

    D:/yolov8/raw_swimxyz/sample001/
        position_1,75.webm
        2D_cam.txt

    D:/yolov8/raw_swimxyz/sample002/
        position_3,75.webm
        2D_cam.txt
    """

    pairs = []

    for folder in sorted(raw_dir.rglob("*")):
        if not folder.is_dir():
            continue

        label_path = folder / "2D_cam.txt"

        if not label_path.exists():
            continue

        video_files = sorted(folder.glob("*.webm"))

        if len(video_files) == 0:
            print(f"跳过：{folder} 中找不到 .webm 视频")
            continue

        if len(video_files) > 1:
            print(f"提示：{folder} 中有多个 .webm，默认使用第一个：{video_files[0].name}")

        video_path = video_files[0]

        relative_name = folder.relative_to(raw_dir).as_posix()
        pair_name = safe_name(relative_name.replace("/", "_"))

        pairs.append({
            "name": pair_name,
            "video": video_path,
            "label": label_path
        })

    return pairs


# ============================================================
# 9. 创建输出目录
# ============================================================

def prepare_output_dirs():
    if CLEAR_OLD_OUTPUT and DATASET_DIR.exists():
        print("正在清空旧数据集输出目录...")
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val", "test"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


# ============================================================
# 10. 写入 swim_pose.yaml
# ============================================================

def write_dataset_yaml():
    yaml_path = DATASET_DIR / "swim_pose.yaml"

    yaml_text = f"""path: {str(DATASET_DIR).replace(os.sep, '/')}

train: images/train
val: images/val
test: images/test

names:
  0: person

kpt_shape: [17, 3]

flip_idx: {FLIP_IDX}
"""

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"YAML 文件已生成：{yaml_path}")


# ============================================================
# 11. 主程序
# ============================================================

def main():
    prepare_output_dirs()

    pairs = find_video_label_pairs(RAW_DIR)

    if len(pairs) == 0:
        print("没有找到任何 .webm + 2D_cam.txt 数据对")
        print("请检查目录结构是否类似：")
        print(r"D:/yolov8/raw_swimxyz/sample001/position_1,75.webm")
        print(r"D:/yolov8/raw_swimxyz/sample001/2D_cam.txt")
        return

    print(f"共找到 {len(pairs)} 组视频+标注")

    random.shuffle(pairs)

    total_videos = len(pairs)
    train_end = int(total_videos * TRAIN_RATIO)
    val_end = int(total_videos * (TRAIN_RATIO + VAL_RATIO))

    split_pairs = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:]
    }

    print(f"train 视频数：{len(split_pairs['train'])}")
    print(f"val 视频数：{len(split_pairs['val'])}")
    print(f"test 视频数：{len(split_pairs['test'])}")

    total_saved_images = 0

    for split, items in split_pairs.items():
        print(f"\n========== 开始处理 {split} ==========")

        for item in items:
            video_name = item["name"]
            video_path = item["video"]
            label_path = item["label"]

            print(f"\n处理样本：{video_name}")
            print(f"视频：{video_path}")
            print(f"标注：{label_path}")

            labels = read_swimxyz_2d_cam(label_path)

            if len(labels) == 0:
                print("标注为空，跳过")
                continue

            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print("视频打开失败，跳过")
                continue

            img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"视频尺寸：{img_w}x{img_h}")
            print(f"视频FPS：{fps}")
            print(f"视频帧数：{total_frames}")
            print(f"标注帧数：{len(labels)}")
            print(f"LABEL_FRAME_OFFSET = {LABEL_FRAME_OFFSET}")
            print(f"MIRROR_X = {MIRROR_X}")
            print(f"MIRROR_Y = {MIRROR_Y}")

            if abs(total_frames - len(labels)) > 2:
                print("警告：视频帧数和标注行数差异较大，请确认是否对应。")

            frame_id = 0
            saved_this_video = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                label_id = frame_id + LABEL_FRAME_OFFSET

                # offset = -1 时，视频第0帧对应标注第-1行，不存在，所以跳过第0帧
                if label_id < 0:
                    frame_id += 1
                    continue

                if label_id >= len(labels):
                    break

                if frame_id % FRAME_INTERVAL == 0:
                    coco17_keypoints = labels[label_id]

                    if coco17_keypoints is not None:
                        # 做上下镜像修正
                        coco17_keypoints = transform_keypoints(coco17_keypoints, img_w, img_h)

                        # 生成 YOLOv8-Pose 标签
                        yolo_label = make_yolo_pose_label(coco17_keypoints, img_w, img_h)

                        if yolo_label is not None:
                            file_stem = f"{video_name}_{frame_id:06d}"

                            image_out_path = DATASET_DIR / "images" / split / f"{file_stem}.jpg"
                            label_out_path = DATASET_DIR / "labels" / split / f"{file_stem}.txt"

                            cv2.imwrite(
                                str(image_out_path),
                                frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                            )

                            with open(label_out_path, "w", encoding="utf-8") as f:
                                f.write(yolo_label + "\n")

                            saved_this_video += 1
                            total_saved_images += 1

                    if MAX_FRAMES_PER_VIDEO is not None:
                        if saved_this_video >= MAX_FRAMES_PER_VIDEO:
                            break

                frame_id += 1

            cap.release()

            print(f"该视频保存图片/标签数量：{saved_this_video}")

    write_dataset_yaml()

    print("\n========== 全部转换完成 ==========")
    print(f"总共生成图片/标签数量：{total_saved_images}")
    print(f"数据集目录：{DATASET_DIR}")
    print(f"YAML文件：{DATASET_DIR / 'swim_pose.yaml'}")


if __name__ == "__main__":
    main()