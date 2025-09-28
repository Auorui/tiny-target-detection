# Tiny Target Detection

本仓库整理并实现了多个面向 **小目标检测（Tiny Object Detection）** 的改进 YOLO 算法，结合博客解析、论文原文与开源代码，方便学习与复现。

## 🚀 使用方法

1. 安装最新版本的[ultralytics]([https://blog.csdn.net/m0_62919535/article/details/151312190])

2. 克隆本仓库：

   ```bash
   git clone git@github.com:Auorui/tiny-target-detection.git
   cd tiny-target-detection
   ```

3. 按照下面的要求放置或替换文件：

   - 将 `std_block` 文件夹放到 `./ultralytics/cfg/models/` 下  
   - 将 `tiny_det` 文件夹放到 `./ultralytics/nn/modules/` 下  
   - 将 `task.py` 文件替换掉 `./ultralytics/nn/task.py`  
   - ⚠️ 注意：`tiny_det` 中的一些函数可能还需要在对应的 `__init__.py` 文件中进行导入

---

## 📌 算法列表

### 1. FFCA-YOLO

* **博客详解**：[CSDN](https://blog.csdn.net/m0_62919535/article/details/151312190)
* **论文地址**：[IEEE Xplore](https://ieeexplore.ieee.org/document/10423050)
* **代码仓库**：[GitHub](https://github.com/yemu1138178251/FFCA-YOLO)

### 2. FBRT-YOLO

* **博客详解**：[CSDN](https://blog.csdn.net/m0_62919535/article/details/151573708)
* **论文地址**：[arXiv](https://arxiv.org/pdf/2504.20670v1)
* **代码仓库**：[GitHub](https://github.com/galaxy-oss/FCM)

### 3. LUD-YOLO

* **博客详解**：[CSDN](https://blog.csdn.net/m0_62919535/article/details/152164243)
* **论文地址**：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0020025524012805)
* **代码仓库**：无

---

## 📖 引用

如果你在研究中使用了这些算法，请引用原论文：

> ```bibtex
> @ARTICLE{10423050,
>   author={Zhang, Yin and Ye, Mu and Zhu, Guiyi and Liu, Yong and Guo, Pengyu and Yan, Junhua},
>   journal={IEEE Transactions on Geoscience and Remote Sensing}, 
>   title={FFCA-YOLO for Small Object Detection in Remote Sensing Images}, 
>   year={2024},
>   volume={62},
>   number={},
>   pages={1-15},
>   keywords={Feature extraction;Remote sensing;YOLO;Convolution;Context-aware services;Finite element analysis;Detectors;Context information;feature fusion;lightweight network;remote sensing image;small object detection},
>   doi={10.1109/TGRS.2024.3363057}}
> ```

> ```bibtex
> @inproceedings{xiao2025fbrt,
>   title={FBRT-YOLO: Faster and Better for Real-Time Aerial Image Detection},
>   author={Xiao, Yao and Xu, Tingfa and Xin, Yu and Li, Jianan},
>   booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
>   volume={39},
>   number={8},
>   pages={8673--8681},
>   year={2025}
> }
> ```

> ```bibtex
> @article{fan2025lud,
>   title={LUD-YOLO: A novel lightweight object detection network for unmanned aerial vehicle},
>   author={Fan, Qingsong and Li, Yiting and Deveci, Muhammet and Zhong, Kaiyang and Kadry, Seifedine},
>   journal={Information Sciences},
>   volume={686},
>   pages={121366},
>   year={2025},
>   publisher={Elsevier}
> }
> ```

---

## 📊 AI-TOD 数据集消融实验结果

| 模型                 | 精度 (P) | 召回率 (R) | mAP50 | mAP50-95 |
| ------------------ | ------ | ------- | ----- | -------- |
| YOLOv5n 原版         | 0.551  | 0.128   | 0.106 | 0.0310   |
| YOLOv5n + 预训练      | 0.545  | 0.174   | 0.165 | 0.0569   |
| YOLOv5m 原版         | 0.753  | 0.171   | 0.166 | 0.0561   |
| YOLOv5m + NWD 损失   | 0.737  | 0.172   | 0.167 | 0.0546   |
| FFCA-YOLOv5m (改进版) | 0.783  | 0.224   | 0.224 | 0.0804   |
| YOLOv8n 原版         | 0.558  | 0.165   | 0.160 | 0.0593   |
| FBRT-YOLOv8n (改进版) | 0.534  | 0.164   | 0.162 | 0.0611   |

> 提示：以上结果均为 **30 epoch 的小规模实验**，仅用于对比分析，不代表完整训练的最终性能。

## 🙌 致谢

感谢各位研究者开源他们的工作，也欢迎大家补充更多 **Tiny Object Detection** 相关方法！
