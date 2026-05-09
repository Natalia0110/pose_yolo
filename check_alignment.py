import cv2
import math
from pathlib import Path


# ============================================================
# 1. 路径配置
# ============================================================

VIDEO_PATH = Path(r"D:/yolov8/raw_swimxyz/sample001/position_1,75.webm")
LABEL_PATH = Path(r"D:/yolov8/raw_swimxyz/sample001/2D_cam.txt")

OUT_DIR = Path(r"D:/yolov8/alignment_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 检查参数
# ============================================================

# 检查哪些视频帧
CHECK_FRAMES = [0, 1, 10, 50, 100, 150, 200, 250, 300]

# 测试标注偏移
# offset = 0 表示：视频第 n 帧 对应 标注第 n 行
# offset = 1 表示：视频第 n 帧 对应 标注第 n+1 行
# offset = -1 表示：视频第 n 帧 对应 标注第 n-1 行
OFFSETS = [0, 1, -1]

# 你的情况是上下镜像，所以这样设置
MIRROR_X = False
MIRROR_Y = True


# ============================================================
# 3. YOLOv8-Pose COCO17 点顺序
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

SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


# ============================================================
# 4. 工具函数
# ============================================================

def parse_number(value: str) -> float:
    value = value.strip().replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return float("nan")


def valid_point(x, y, img_w, img_h) -> bool:
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


def read_swimxyz_2d_cam(label_path: Path):
    """
    读取 2D_cam.txt。

    兼容：
    - 表头 75 字段
    - 实际每行 54 字段
    """
    frames = []

    with open(label_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    if len(lines) < 2:
        print("标注文件内容太少")
        return frames

    full_header = lines[0].strip().split(";")
    full_header = [h.strip() for h in full_header if h.strip() != ""]

    print(f"表头字段数：{len(full_header)}")

    for line in lines[1:]:
        line = line.strip()

        if not line:
            frames.append(None)
            continue

        parts = line.split(";")
        parts = [p.strip() for p in parts if p.strip() != ""]

        usable_header = full_header[:len(parts)]

        point_data = {}

        for field_name, raw_value in zip(usable_header, parts):
            if "." not in field_name:
                continue

            point_name, axis_name = field_name.rsplit(".", 1)

            if point_name not in point_data:
                point_data[point_name] = {}

            point_data[point_name][axis_name] = parse_number(raw_value)

        coco17 = []

        for point_name in COCO17_FROM_SWIMXYZ:
            data = point_data.get(point_name, {})

            x = data.get("x", float("nan"))
            y = data.get("y", float("nan"))

            coco17.append([x, y])

        frames.append(coco17)

    return frames


def draw_pose(frame, keypoints, text):
    img = frame.copy()
    img_h, img_w = img.shape[:2]

    # 画骨架线
    for a, b in SKELETON:
        xa, ya = keypoints[a]
        xb, yb = keypoints[b]

        if valid_point(xa, ya, img_w, img_h) and valid_point(xb, yb, img_w, img_h):
            cv2.line(
                img,
                (int(xa), int(ya)),
                (int(xb), int(yb)),
                (0, 255, 0),
                2
            )

    # 画关节点
    for idx, (x, y) in enumerate(keypoints):
        if valid_point(x, y, img_w, img_h):
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(
                img,
                str(idx),
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    cv2.putText(
        img,
        text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    return img


def read_specific_frame(cap, frame_id):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if not ret:
        return None

    return frame


# ============================================================
# 5. 主程序
# ============================================================

labels = read_swimxyz_2d_cam(LABEL_PATH)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("视频打开失败")
    exit()

video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"视频帧数：{video_frames}")
print(f"视频FPS：{fps}")
print(f"标注帧数：{len(labels)}")
print(f"标注帧数 - 视频帧数 = {len(labels) - video_frames}")
print(f"MIRROR_X = {MIRROR_X}")
print(f"MIRROR_Y = {MIRROR_Y}")

for frame_id in CHECK_FRAMES:
    if frame_id >= video_frames:
        continue

    frame = read_specific_frame(cap, frame_id)

    if frame is None:
        continue

    img_h, img_w = frame.shape[:2]

    compare_imgs = []

    for offset in OFFSETS:
        label_id = frame_id + offset

        if label_id < 0 or label_id >= len(labels):
            continue

        keypoints = labels[label_id]

        if keypoints is None:
            continue

        keypoints = transform_keypoints(keypoints, img_w, img_h)

        text = f"video={frame_id}, label={label_id}, offset={offset}, mirror_y={MIRROR_Y}"

        img = draw_pose(frame, keypoints, text)

        save_path = OUT_DIR / f"frame_{frame_id:06d}_label_{label_id:06d}_offset_{offset}.jpg"
        cv2.imwrite(str(save_path), img)

        # 缩小后用于横向对比
        new_w = 640
        new_h = int(img.shape[0] * new_w / img.shape[1])
        resized = cv2.resize(img, (new_w, new_h))
        compare_imgs.append(resized)

    if len(compare_imgs) > 0:
        compare_img = cv2.hconcat(compare_imgs)
        compare_path = OUT_DIR / f"compare_frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(compare_path), compare_img)

cap.release()

print("对齐检查完成")
print(f"检查图保存位置：{OUT_DIR}")