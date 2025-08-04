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

> **Tip:** The `C* B* S* Clahe Inv BC` patterns encode the Enhancements and the factor of enhancement used against Contrast, Brightness, Sharpness, CLAHE (true/ false), Inversion (true / false) and Blue Channel RGB (true / false) respectively.

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

## 4 Logging and reproduction of thesis: 

Each run produces its own entry in the CSV file in the same folder (`NP_model_results_log.csv` or `P_model_results_log.csv`).
Multiple runs have been performed with various image sets within the dataset, Thesis related runs that were performed have been taken into account in the "Experiment Results.xlsx" file, providing comprehensive proof for the thesis claim.

The results you would see on the ipynb files are not final results, but rather iterative of various trials that have been conducted on the datasets. Not to be interpretted otherwise. 

---

## 5  Key Findings
* Post‑ELA local enhancements **raise F1‑score by up to 12 %** without increasing model depth.  
* **Contrast amplification** is the most reliable single enhancer across all three networks.  
* **MobileNetV2** (3.5 M parameters) matches or exceeds deeper ResNet50v2 once enhancements are applied, making it suitable for resource‑constrained devices.



---

## 6  Extending the Work
* **Custom Enhancers** – add a function to `src/enhancements.py`, register its name in `configs/config.yaml`, rerun training.  
* **Other Datasets** – create a subclass of `BaseDataLoader` pointing to your directory structure.  
* **Real‑time Demo** – `streamlit run demo_app.py` (prototype, GPU recommended).

---

## 7 Hardware and Software Requirements

### Hardware

| Component | Specification | Purpose |
|-----------|---------------|---------|
| Processor | Intel Core i7‑12700H | Multi‑core CPU for data processing tasks |
| GPU | NVIDIA GeForce RTX 3060 (12 GB VRAM) | Accelerated training of CNN models |
| RAM | 16 GB DDR4 | Efficient memory handling during training |
| Storage | 1 TB SSD | Fast read and write operations |
| Operating system | Ubuntu 22.04 LTS | Stable Linux environment for machine learning |

### Software

| Library or Tool | Version | Role in Implementation |
|-----------------|---------|------------------------|
| TensorFlow / Keras | 2.12 | Define, train, and evaluate CNN architectures |
| OpenCV | 4.7 | ELA generation, CLAHE processing, blue‑channel separation |
| Pillow (PIL) | 9.5.0 | Image loading, resizing, and enhancements (contrast, brightness, sharpness) |
| NumPy | 1.24.2 | Numerical computation and image matrix manipulation |
| Pandas | 1.5.3 | Logging experiment metadata and results |
| Matplotlib | 3.7.1 | Visualisation of training and evaluation metrics |
| Seaborn | 0.12.3 | Comparative heatmaps and plots for model metrics |
| scikit‑learn | 1.2.2 | Accuracy, precision, recall, F1‑score, confusion matrix |
| tqdm | 4.65.0 | Progress bars for loops and training |
| os, shutil, pathlib | Built‑in | File and directory manipulation, dataset structuring |
| random, datetime | Built‑in | Reproducible shuffling, seeding, timestamping of logs |
| time | Built‑in | Timing of training cycles and monitoring |
| BytesIO | Built‑in | In‑memory ELA map handling |
| collections.defaultdict | Built‑in | Structured dictionaries for experiment tracking |
| tensorflow.keras.callbacks | Bundled | Early stopping and learning‑rate scheduling |
| ImageDataGenerator | Bundled | Data augmentation: zoom, flip, rotate, shear |
| Visual Studio Code | 1.101.1 | IDE used for code development |
| Git | 2.50.1 | Version control |
| Miniconda | 23.5.1 | Virtual‑environment management |

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

