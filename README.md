# Tiny Target Detection

本仓库整理并实现了多个面向 **小目标检测（Tiny Target Detection）** 的改进算法，结合博客解析、论文原文与开源代码，方便学习与复现。

## 🚀 使用方法

### ultralytics系列的改进

1. 安装最新版本的 [ultralytics](https://blog.csdn.net/m0_62919535/article/details/151312190)

2. 克隆本仓库：

   ```bash
   git clone git@github.com:Auorui/tiny-target-detection.git
   cd tiny-target-detection/ultralytics-tiny
   ```

3. 按照下面的要求放置或替换文件：

   - 将 `extra_modules` 文件夹放到 `./ultralytics/nn/` 下，与 `modules` 处于同目录下  
   - 将 `tiny_det` 文件夹放到 `./ultralytics/cfg/modules/` 下  
   - 将 `task.py` 文件替换掉 `./ultralytics/nn/task.py`（主要修改的是其中的 `parse_model` 函数）  
   - ⚠️ 注意：`tiny_det` 中的一些函数可能还需要在对应的 `__init__.py` 文件中进行导入  

---

### mmdet系列的改进

1. 安装最新版本的 [mmdet](https://github.com/open-mmlab/mmdetection)

2. 安装教程请看以下两篇：
   - [MMCV与MMDetection安装指南](https://blog.csdn.net/m0_62919535/article/details/132595519)
   - [MMDetection新手教程](https://blog.csdn.net/m0_62919535/article/details/151828419)

3. 按照下面的要求放置或替换文件：

   - 将 `extra_modules` 文件夹放到 `./mmdet` 根目录下  
   - 将 `models` 下的文件放到对应的同目录下；对于每个添加了新文件的目录，都需要在对应的 `__init__.py` 文件中注册  
   - 将 `configs_tiny` 放到项目文件根目录中，与 `configs` 文件夹同目录
   - 将 `get_map.py` 放到项目文件根目录中，与 `configs_tiny` 文件夹同目录
   - torch==2.0.0+cu118、mmcv==2.0.1、mmengine==0.10.7、mmdet==3.3.0

最终目录结构如下：

   ```bash
   mmdetection/
   ├── mmdet/
   │   ├── extra_modules/          ← 放在这里
   │   │   ├── __init__.py
   │   │   ├── fp16_utils.py
   │   ├── models/
   │   ├── structures/
   │   └── ...
   ├── configs/
   ├── configs_tiny/               ← 与configs同级
   ├── get_map.py                  ← 输出与ultralytics相同格式的指标
   └── 
   ```

---

## 📌 算法列表

> 算法列表的顺序仅跟学习顺序有关

|  方法  |  博客  |  论文地址  | 代码仓库  |  发表期刊/会议 & 时间  |
| --------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ---------------------------- |
|  **FFCA-YOLO**  | [CSDN](https://blog.csdn.net/m0_62919535/article/details/151312190) | [FFCA-YOLO for Small Object Detection in Remote Sensing Images](https://ieeexplore.ieee.org/document/10423050) | [GitHub](https://github.com/yemu1138178251/FFCA-YOLO) |*IEEE TGRS*, 2024|
|  **FBRT-YOLO**  | [CSDN](https://blog.csdn.net/m0_62919535/article/details/151573708) | [FBRT-YOLO: Faster and Better for Real-Time Aerial Image Detection](https://arxiv.org/pdf/2504.20670v1) | [GitHub](https://github.com/galaxy-oss/FCM)  |*AAAI Conference*, 2025|
|  **LUD-YOLO**   | [CSDN](https://blog.csdn.net/m0_62919535/article/details/152164243) | [LUD-YOLO: A novel lightweight object detection network for unmanned aerial vehicle](https://www.sciencedirect.com/science/article/pii/S0020025524012805) | 无 |*Information Sciences*, 2025|
|  **VRF-DETR**   | [CSDN](https://blog.csdn.net/m0_62919535/article/details/152615666) | [An Efficient Aerial Image Detection with Variable Receptive Fields](https://arxiv.org/pdf/2504.15165) | [GitHub](https://github.com/LiuWenbin-CV/VRF-DETR) |*Arxiv*, 2025| 
|  **IF-YOLO**  | [CSDN](https://blog.csdn.net/m0_62919535/article/details/153835570) | [Unmanned Aerial Vehicle Object Detection Based on Information-Preserving and Fine-Grained Feature Aggregation](https://www.mdpi.com/2072-4292/16/14/2590) | 无 | *Remote Sensing*, 2024 |
|  **PRNet**  | [CSDN](https://blog.csdn.net/m0_62919535/article/details/153969350) | [PRNet: Original Information Is All You Have](https://arxiv.org/abs/2510.09531) | [GitHub](https://github.com/hhao659/PRNet) | *Arxiv*, 2025 |
|  **HS-FPN**  | [CSDN](https://blog.csdn.net/m0_62919535/article/details/154121182) | [HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection](https://arxiv.org/pdf/2412.10116) | [GitHub](https://github.com/ShiZican/HS-FPN) | *AAAI Conference*, 2025 |
|  **PKINet**  | [CSDN] | [Poly Kernel Inception Network for Remote Sensing Detection](https://arxiv.org/pdf/2403.06258) | [GitHub](https://github.com/NUST-Machine-Intelligence-Laboratory/PKINet) | *CVPR Conference*, 2024 |

---

## 📊 消融实验
### AI-TOD 数据集消融实验结果
主要面向ultralytics系列改进的实验
| 模型                  | 精度(P)  |  召回率(R) |  mAP50 | mAP50-95 |
| --------------------- | -------- | ---------- | -----  | -------- |
| YOLOv5n 原版          |  0.551   |   0.128    | 0.106  |  0.0310  |
| YOLOv5n + 预训练      |  0.545   |   0.174    | 0.165  |  0.0569  |
| YOLOv5m 原版          |  0.753   |   0.171    | 0.166  |  0.0561  |
| YOLOv5m + NWD 损失    |  0.737   |   0.172    | 0.167  |  0.0546  |
| FFCA-YOLOv5m (改进版) |  0.783   |   0.224    | 0.224  |  0.0804  |
| YOLOv8n 原版          |  0.558   |   0.165    | 0.160  |  0.0593  |
| FBRT-YOLOv8n (改进版) |  0.534   |   0.164    | 0.162  |  0.0611  |
| LUD-YOLOv8n (改进版)  |  0.621   |   0.160    | 0.156  |  0.0577  |
| RT-DETR-resnet50 原版 |  0.645   |   0.133    | 0.0848 |  0.0274  |
| VRF-DETR (改进版)     |  0.632   |   0.0954   | 0.0594 |  0.0189  |
| YOLOv8s 原版          |  0.548   |   0.189    | 0.193  |  0.0769  |
| IF-YOLOv8s (改进版)   |  0.647   |   0.190    | 0.193  |  0.0771  |
| YOLO11n 原版          |  0.720   |   0.146    | 0.138  |  0.0518  |
| YOLO11n-PRNet (改进版)|  0.523   |   0.202    | 0.206  |  0.0851  |  
| YOLOv8n-hsfpn (改进版) |  0.540   |   0.173    | 0.171  |  0.0679  | 
> 提示：以上结果均为 **30 epoch 的小规模实验**，仅用于对比分析，不代表完整训练的最终性能。


### VisDrone2019 数据集消融实验结果
主要面向mmdet系列改进的实验
| 模型                    | 精度(P)  |  召回率(R) |  mAP50 | mAP50-95 |
| ----------------------- | -------- | ---------- | -----  | -------- |
| faster-rcnn(r50, FPN)   |  0.564   |   0.416    |  0.463 |   0.286  |
| faster-rcnn(r50, HS-FPN)|  0.581   |   0.421    |  0.473 |   0.293  |
> 提示：以上结果均为 **1x(12 epoch) 的小规模实验**，仅用于对比分析，不代表完整训练的最终性能。


## 🙌 致谢

感谢各位研究者开源他们的工作，也欢迎大家补充更多 **Tiny Target Detection** 相关方法！
