# Image Forensics Using ELA and Lightweight CNN: Impact of ELA Image Post‑Processing on Forensic Performance
*Impact of Post‑ELA Enhancements on Lightweight CNN Performance*

> MSc Data Science — Liverpool John Moores University  
> **Surendra Dhondale Mohan • August 2025**  

---

## 1  Project Synopsis
Digital photographs circulate at unprecedented scale, yet sophisticated editing tools make visual misinformation easy to create and hard to detect.  
This repository accompanies the LJMU Master’s thesis **“Image Forensics Using ELA and Lightweight CNN: Impact of ELA Image Post‑Processing on Forensic Performance.”**  
We investigate whether simple post‑processing operations applied *after* Error Level Analysis (ELA) can improve lightweight CNN detectors without the computational overhead of very deep models.

---

## 2  Repository Layout

```
LJMU/
└── CodeBase/
    ├── StableVersions-EfficientNet/
    │   ├── NonProcessedImageForensics/
    │   │   ├── EffNet_Base.ipynb
    │   │   ├── EffNet_Zoom_Range_0.2.ipynb
    │   │   ├── EffNet_Zoom_Range_0.3.ipynb
    │   │   └── NP_model_results_log.csv
    │   └── ProcessedImageForensics/
    │       ├── EffNet_C*_B*_S*_CLAHE*_INVO*_BC*_E30*.ipynb
    │       └── P_model_results_log.csv
    │
    ├── StableVersions-MobileNet/
    │   ├── NonProcessedImageForensics/
    │   │   ├── MobileNetV2_Base.ipynb
    │   │   ├── MobileNetV2_Zoom_Range_0.2.ipynb
    │   │   ├── MobileNetV2_Zoom_Range_0.3.ipynb
    │   │   └── NP_model_results_log.csv
    │   └── ProcessedImageForensics/
    │       ├── MobileNetV2_C*_B*_S*_CLAHE*_INVO*_BC*_*.ipynb
    │       └── P_model_results_log.csv
    │
    ├── StableVersions-ResNet/
    │   ├── NonProcessedImageForensics/
    │   │   ├── ResNet50v2_Base.ipynb
    │   │   ├── ResNet50v2_Zoom_Range_0.2.ipynb
    │   │   ├── ResNet50v2_Zoom_Range_0.3.ipynb
    │   │   └── NP_model_results_log.csv
    │   └── ProcessedImageForensics/
    │       ├── ResNet50v2_C*_B*_S*_CLAHE*_INVO*_BC*_*.ipynb
    │       └── P_model_results_log.csv
    │
    └── Dataset/                      ← **Not included in the repo (≈5 GB)**
        ├── Casia-v2-Modified/
        └── Casia-v2-originalDS/      ← Place downloaded dataset here
```

> **Tip:** The `C* B* S*` patterns encode the optimiser’s channel, block, and stride hyper‑parameters. Full naming conventions are described in Thesis §4.2.

---

## 3  Quick‑start (5 steps)

| Step | Command | Notes |
|------|---------|-------|
| 1 | `git clone https://github.com/Surendra-Dhondale/Image-Forgery-Detection-EffectofImagePostProcessing.git` |  |
| 2 | `cd Image-Forgery-Detection-EffectofImagePostProcessing` |  |
| 3 | `conda env create -f environment.yml`<br>`conda activate ela_cnn` | Python 3.10, TensorFlow 2.15, OpenCV, etc. |
| 4 | **Download dataset** from Kaggle → <https://www.kaggle.com/datasets/dk9892/casia-v2><br>Unzip and place images under `LJMU/CodeBase/Dataset/Casia-v2-originalDS/` so that paths look like `.../Authentic/*` and `.../Tampered/*`. | Dataset (~5 GB) is intentionally **not** pushed to GitHub. |
| 5 | Open any notebook in the relevant *StableVersions‑*/**ModelName** folder and run all cells **or** launch CLI training:<br>`python src/train_eval.py --config configs/config.yaml` | Re‑trains MobileNetV2 baseline (≈6 min on RTX 3060). |

---

## 4  Reproducing the Thesis Results

| Experiment | Notebook | Expected F1 | Thesis Table |
|------------|----------|-------------|--------------|
| MobileNetV2 baseline | `StableVersions‑MobileNet/NonProcessedImageForensics/MobileNetV2_Base.ipynb` | 0.81 ± 0.01 | 5.3 |
| MobileNetV2 +Contrast | `.../ProcessedImageForensics/MobileNetV2_C1_B1_S1_CLAHE0_INVO_BC1_E30i.ipynb` | 0.89 ± 0.00 | 5.10 |
| EfficientNetB0 +Sharp | `StableVersions‑EfficientNet/ProcessedImageForensics/EffNet_C2_B1_S1_CLAHE0_INVO_BCO_E30i.ipynb` | 0.92 ± 0.01 | 5.6 |
| ResNet50v2 +Blue Ch. | `StableVersions‑ResNet/ProcessedImageForensics/ResNet50v2_C1_B2_S1_CLAHE0_INVO_BCO_E30i.ipynb` | 0.90 ± 0.01 | 5.8 |

Each run produces its own CSV in the same folder (`NP_model_results_log.csv` or `P_model_results_log.csv`).

---

## 5  Key Findings
* Post‑ELA local enhancements **raise F1‑score by up to 8 pp** without increasing model depth.  
* **Contrast amplification** is the most reliable single enhancer across all three networks.  
* **MobileNetV2** (3.5 M parameters) matches or exceeds deeper ResNet50v2 once enhancements are applied, making it suitable for resource‑constrained devices.

Details are discussed in Thesis §5.5–§5.7.

---

## 6  Extending the Work
* **Custom Enhancers** – add a function to `src/enhancements.py`, register its name in `configs/config.yaml`, rerun training.  
* **Other Datasets** – create a subclass of `BaseDataLoader` pointing to your directory structure.  
* **Real‑time Demo** – `streamlit run demo_app.py` (prototype, GPU recommended).

---

## 7  Requirements
| Package | Version tested |
|---------|----------------|
| Python | 3.10 |
| TensorFlow | 2.15 |
| Keras CV | 0.11 |
| OpenCV‑Python | 4.10 |
| Pillow | 10.4 |
| scikit‑learn | 1.5 |
| NumPy / Pandas / Matplotlib / *etc.* | latest |

Full specification in `environment.yml`.

---

## 8  Citation
If you use this code or draw upon the reported results, please cite:

```bibtex
@mastersthesis{dhondale2025ela,
  title  = {Image Forensics Using ELA and Lightweight CNN: Impact of ELA Image Post‑Processing on Forensic Performance},
  author = {Surendra Dhondale Mohan},
  school = {Liverpool John Moores University},
  year   = {2025}
}
```

---

---

