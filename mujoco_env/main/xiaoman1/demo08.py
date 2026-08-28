import json
from pathlib import Path
import cv2

DATASET = Path("/home/xiaoman/study/Arm_Project/rgbd_saved_external")
RGB_DIR = DATASET / "rgb"
LABEL_DIR = DATASET / "yolo_labels"
LABEL_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {
    "beaker1": 0,
    "graduated_cylinder": 1,
    "erlenmeyer_flask": 2,
}


def make_yolo_label(image_path, json_path, label_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    height, width = image.shape[:2]
    objects = json.loads(json_path.read_text())
    labels = []

    for obj in objects:
        name = obj["name"]
        print(f"\n请框选物体：{name}")
        print("拖动鼠标框选，按 Enter 确认，按 Esc 退出")

        window = f"标注 {name}"
        x, y, w, h = cv2.selectROI(window, image, showCrosshair=True)
        cv2.destroyWindow(window)

        if w <= 0 or h <= 0:
            print(f"跳过 {name}")
            continue

        # 像素坐标
        x_center = x + w / 2.0
        y_center = y + h / 2.0

        # 转换为 YOLO 归一化格式
        line = (
            f"{CLASS_NAMES[name]} "
            f"{x_center / width:.6f} "
            f"{y_center / height:.6f} "
            f"{w / width:.6f} "
            f"{h / height:.6f}"
        )

        labels.append(line)

        # 在图像上画框检查
        cv2.rectangle(
            image,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            image,
            name,
            (int(x), int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    label_path.write_text("\n".join(labels) + "\n")
    preview_path = label_path.with_name(label_path.stem + "_preview.jpg")
    cv2.imwrite(str(preview_path), image)

    print(f"标签已保存：{label_path}")
    print(f"预览图已保存：{preview_path}")


def main():
    json_files = sorted((DATASET / "lables").glob("*.json"))

    for json_path in json_files:
        sample_id = json_path.stem
        image_path = RGB_DIR / f"{sample_id}_rgb.png"
        label_path = LABEL_DIR / f"{sample_id}.txt"

        if not image_path.exists():
            print(f"找不到图像：{image_path}")
            continue

        print(f"\n开始标注样本：{sample_id}")
        make_yolo_label(image_path, json_path, label_path)


if __name__ == "__main__":
    main()