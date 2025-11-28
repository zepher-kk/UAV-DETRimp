# UAV-DETR+
# UAV-DETR+: More tiny pixels and frequency information help small object detection

Updates
- **[2025.6]**​ Release UAV-DETR+-R50, UAV-DETR+-R18.
paper : https://doi.org/10.1117/1.JEI.34.5.053032
  

- 🔥 ​**UAV-DETR+**



## Experimental Results on the VisDrone-2019-DET Val Dataset

**Note:** The hyperparameters of the UAV-YOLO+-S version are highly sensitive during training and require adjustments based on the hyperparameters of DETR.

| **Model**​            | **Backbone**​         | **Input Size**​ | **Params (M)**​ | **GFLOPs**​ | **AP**​  | **AP$_{50}$**​ |
|----------------------|---------------------|----------------|----------------|------------|---------|---------------|
| UAV-YOLO+-S (Ours) | YOLOv8s | 640×640        | 12.1      | 32.7  | 23.2 | 39.3      |
| UAV-DETR+-R18 (Ours) | ResNet18            | 640×640        | 23.7       | 78.8   | **30.6** | **50.1** |
| UAV-DETR+-R50 (Ours) | ResNet50            | 640×640        | 38.2       | 137   | 31.1 | 50.7 |

---

## Experimental Results on UAVVaste Val Dataset

| **Model**​            | **Params (M)**​ | **GFLOPs**​ | **AP**​ | **AP$_{50}$**​ |
| -------------------- | -------------- | ---------- | ------ | ------------- |
| UAV-DETR+-R18 (Ours) | 23.7           | 78.8       | 49.4   | **77.6**      |

---

## Ablation Study

| **Model Configuration**​ | **AP**​  | **AP$_{50}$**​ |
|-------------------------|---------|---------------|
| Baseline(UAV-DETR)      | 29.8 | 48.8      |
| Baseline + MF | 30.3 | 49.5      |
| Baseline + MF+WTDS | 30.5 | 49.8      |
| Baseline + MF+WTDS+(SE+cat)+FFC | 30.6 | 50        |
| **Full Model**​          | **30.6**​ | **50.1**​  |

---

## 📍 Environment
- torch 2.0.1+cu11.7 
- torchvision 0.15.2+cuda11.7 
- Ubuntu 20.04

---
Correspondence should be addressed to jzhang@hnust.edu.cn
