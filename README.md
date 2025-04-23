# An Edge Refinement and Enhancement Network (ERENet) Using Multi-Level Information Fusion for Edge Detection

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
---
## 📬 Contact
For questions or collaboration requests, please contact:

- shuangli：shuangli@hit.edu.cn
- YichengChen：23b933075@stu.hit.edu.cn
- ChangqingLi：23b933033@stu.hit.edu.cn
- ChangFeng：23S933033@stu.hit.edu.cn
- ChanghaiZhai：zch-hit@hit.edu.cn

