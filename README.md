智慧化农业无人机玉米穗检测数据集
776张
无人机航拍玉米田图片，标注清晰，适合玉米穗目标检测模型训练。YOLO格式
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8146db08188e448c879f09f928d4adb5.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fa13dd9575dd482788a272c98e3e3ccb.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2215e38109784f8cad62c946e2dba76f.png)




---

## 🌽 项目概述：智慧农业无人机玉米穗检测系统

| 项目 | 内容 |
|------|------|
| 数据集 | 776 张无人机航拍玉米田图片（含多角度、光照变化） |
| 标注类型 | 清晰标注的玉米穗边界框（Bounding Box） |
| 标注格式 | 支持 YOLO `.txt` 格式（归一化坐标） |
| 模型选择 | 使用 **YOLOv8n**（轻量级，适合边缘部署）或模拟未来风格的 `YOLOv9` 架构调用方式 |
| 应用场景 | 农业估产、生长监测、智能收割导航 |

---

## ✅ 一、数据集结构要求（标准 YOLO 格式）【代码示例，仅供参考学习】

```
corn_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

### 示例 label 文件（`.txt`）：
```text
0 0.5 0.6 0.05 0.1   # class_id x_center y_center width height (normalized)
```

> 假设只有一个类别：“corn_ear”（玉米穗），class_id = 0

---

## ✅ 二、`dataset.yaml` 配置文件【代码示例，仅供参考学习】

```yaml
# dataset.yaml
train: ./corn_dataset/images/train
val: ./corn_dataset/images/val
test: ./corn_dataset/images/test

nc: 1
names: ['corn_ear']
```

---

## ✅ 三、`train.py` —— 使用 YOLOv8 训练玉米穗检测模型（兼容未来语法风格）【代码示例，仅供参考学习】

```python
# train.py
from ultralytics import YOLO
import torch

def train_corn_model():
    # 可选模型：yolov8n.pt, yolov8s.pt 等
    model = YOLO('yolov8n.pt')  # 加载预训练权重

    results = model.train(
        data='dataset.yaml',
        epochs=150,
        imgsz=640,
        batch=16,
        name='corn_detection_exp',
        workers=4,
        device=0 if torch.cuda.is_available() else 'cpu',
        optimizer='AdamW',           # 更先进的优化器
        lr0=0.001,                   # 初始学习率
        augment=True,                # 启用数据增强（旋转、翻转、色彩扰动）
        hsv_h=0.015,                 # 色彩增强
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,                  # 随机旋转
        translate=0.2,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5
    )

    print("✅ 训练完成！模型保存在 runs/detect/corn_detection_exp/")
    return results

if __name__ == "__main__":
    train_corn_model()
```

> 🔧 提示：如果 GPU 显存不足，可将 `batch` 改为 8 或 4。

---

## ✅ 四、`detect.py` —— 单张图像推理与可视化

```python
# detect.py
import cv2
from ultralytics import YOLO

class CornEarDetector:
    def __init__(self, model_path='runs/detect/corn_detection_exp/weights/best.pt'):
        self.model = YOLO(model_path)
        self.class_names = ['corn_ear']

    def detect(self, image_path, conf_threshold=0.5):
        results = self.model(image_path, conf=conf_threshold)
        result = results[0]
        frame = result.plot()  # 自动绘制边界框和标签
        cv2.imwrite("result_detected.jpg", frame)
        print(f"✅ 检测完成，结果已保存为 result_detected.jpg")
        return frame, len(result.boxes)

# 测试
if __name__ == "__main__":
    detector = CornEarDetector()
    frame, count = detector.detect("test_image.jpg")
    cv2.imshow("Detected Corn Ears", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

---

## ✅ 五、评估与预测建议

### 1. **验证集评估**
```bash
yolo task=detect mode=val model=runs/detect/corn_detection_exp/weights/best.pt data=dataset.yaml
```

### 2. **测试集推理（批量）**
```bash
yolo task=detect mode=predict model=best.pt source=corn_dataset/images/test save=True
```

---

## ✅ 六、数据增强建议（提升小目标检测性能）【代码示例，仅供参考学习】

由于玉米穗在高空航拍中属于**小目标**，建议：

- 使用 **Mosaic 数据增强**（默认开启）
- 添加 **Copy-Paste 增强**：复制粘贴玉米穗到新位置，增加密度多样性
- 使用 **高分辨率输入**（如 `imgsz=800` 或 `1024`），但需权衡速度
- 启用 **FPN/PANet 结构**（YOLOv8 已内置，无需修改）

---

## ✅ 七、模型导出与部署

```python
# export.py
from ultralytics import YOLO

model = YOLO('runs/detect/corn_detection_exp/weights/best.pt')
model.export(format='onnx', imgsz=640)      # ONNX for Windows/Linux
model.export(format='engine', half=True)    # TensorRT for Jetson (加速10倍+)
model.export(format='tflite')               # TensorFlow Lite for mobile
```

> 可部署至：
> - 无人机机载计算单元（如 NVIDIA Jetson Nano/TX2）
> - 边缘服务器进行实时分析
> - 手机 App 查看检测结果

---

## ✅ 八、UI 界面原型（可选 PyQt6 实现）

```python
# ui_app.py（简化版）
import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from detect import CornEarDetector

class CornApp(QWidget):
    def __init__(self):
        super().__init__()
        self.detector = CornEarDetector()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.label = QLabel("上传玉米田图片")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn = QPushButton("选择图片")
        btn.clicked.connect(self.run_detection)

        layout.addWidget(self.label)
        layout.addWidget(btn)
        self.setLayout(layout)
        self.setWindowTitle("🌽 智慧农业 - 玉米穗检测")

    def run_detection(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Image Files (*.jpg *.png)")
        if path:
            frame, count = self.detector.detect(path)
            pixmap = QPixmap("result_detected.jpg").scaled(800, 600)
            self.label.setPixmap(pixmap)
            QMessageBox.information(self, "结果", f"共检测到 {count} 个玉米穗！")

app = QApplication(sys.argv)
win = CornApp()
win.show()
sys.exit(app.exec())
```

---

## ✅ 九、应用领域拓展【代码示例，仅供参考学习】

| 应用方向 | 说明 |
|---------|------|
| 📊 **产量预估** | 统计单位面积内玉米穗数量，预测总产量 |
| 🌱 **品种对比实验** | 分析不同品种结穗密度、分布均匀性 |
| 🤖 **自动驾驶收割机引导** | 提供实时玉米位置信息用于路径规划 |
| ☁️ **数字农田平台** | 接入云平台实现远程监控与历史数据分析 |

---


