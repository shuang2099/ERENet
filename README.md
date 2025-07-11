# ERENet: Multi-Level Information Fusion for Refined and Enhanced Edge Detection

**Authors**: Shuang Li, Yicheng Chen, Changqing Li, Chang Feng, Changhai Zhai

---

## 🔍 Overview

Edge detection is one of the most fundamental yet challenging problems in computer vision, particularly in preserving low-contrast, fine-grained structural information. Most existing methods overly focus on high-contrast contours, missing subtle boundary details crucial for accurate scene understanding.

**ERENet** (Edge Refinement and Enhancement Network) introduces a novel two-stage architecture that extracts multi-level features and refines them with plug-and-play modules, improving both edge sharpness and structural integrity.

---

## 🎯 Key Features

-  **Two-stage architecture**:
  - Stage 1: Multi-level feature extraction.
  - Stage 2: Plug-in refinement modules.

-  **Edge Enhancement Module**:
  - Utilizes receptive field flow to sharpen edges progressively.

-  **Attention Fusion Module**:
  - Fuses local textures with global semantic context for richer edge representations.

-  **Strong performance** across BIPED, BSDS500, and UDED benchmarks (ODS, OIS, AP).

---

## ✅Train
python main.py 
  - Note: After training, make sure to save the model weights in the model directory
## ✅Test
python test.py 
  - Link to the trained model:https://pan.baidu.com/s/1xxkrmX_NeKptoPo6hU1vbg?pwd=n4xb  code：n4xb
---

## 📬 Contact
For questions or collaboration requests, please contact:

- Shuang Li：shuangli@hit.edu.cn
- Yicheng Chen：23b933075@stu.hit.edu.cn
- Changqing Li：23b933033@stu.hit.edu.cn
- Chang Feng：23S933033@stu.hit.edu.cn
- Changhai Zhai：zch-hit@hit.edu.cn

