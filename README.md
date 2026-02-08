# Satellite Imagery Segmentation & Deforestation Analysis

This repository contains a **full-stack solution** for forest cover analysis and deforestation detection from Landsat 8 satellite imagery.

---

## 🌲 Two ML Pipelines

### 1. Forest Segmentation (mU-Net)
Generates pixel-level forest masks from single-date imagery.

| Stage | Method |
|-------|--------|
| **Data Prep** | Random Forest weak supervision |
| **Model** | mU-Net (Keras/TensorFlow) |
| **Input** | 9-band GeoTIFF |
| **Output** | Binary forest mask |

```
GeoTIFF → [RF Weak Labels] → [mU-Net Training] → Forest Mask
```

---

### 2. Change Detection (Siamese U-Net)
Detects forest cover changes between two dates.

| Stage | Method |
|-------|--------|
| **Data Prep** | NDVI Difference + Otsu |
| **Model** | Siamese U-Net (PyTorch) |
| **Input** | T1 + T2 images (4-band) |
| **Output** | Change mask |

```
T1 + T2 → [NDVI Diff] → [Siamese U-Net] → Change Mask
```

---

## 📁 Model Pipeline Structure

| File | Description |
|------|-------------|
| **src/data/** | Data loading, preprocessing, labeling |
| **src/models/forest_segmentation.py** | mU-Net (Keras) |
| **src/models/change_detection.py** | Siamese U-Net (PyTorch) |
| **src/training/** | Training loops, metrics |
| **prepare_data.py** | Data preparation CLI |
| **train_forest.py** | Forest segmentation training |
| **train.py** | Change detection training |

---

## 🚀 Quick Setup

```powershell
cd "Model Pipeline"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 📋 Usage

### Pipeline 1: Forest Segmentation

```powershell
# Step 1: Generate dataset (RF weak labels + patches)
python prepare_data.py --mode forest_prep --input_t1 image.tif --output_dir forest_dataset

# Step 2: Train mU-Net
python train_forest.py --data_root forest_dataset --epochs 150
```

---

### Pipeline 2: Change Detection

```powershell
# Step 1: Generate change mask
python prepare_data.py --mode change_label --input_t1 T1.tif --input_t2 T2.tif --output_dir output

# Step 2: Build dataset
python prepare_data.py --mode build_dataset --input_t1 T1.tif --input_t2 T2.tif --input_mask output/calculated_mask.tif --aoi_name Region --output_dir dataset

# Step 3: Train Siamese U-Net
python train.py --data_root dataset --epochs 60
```

---

## 🧪 Testing

```powershell
python tests/test_models.py        # Model architectures
python tests/test_comprehensive.py  # Full pipeline
```

| Test | Status |
|------|--------|
| mU-Net (Keras) | ✅ |
| Siamese U-Net (PyTorch) | ✅ |
| Data Loader | ✅ |
| RF Labeler | ✅ |
| NDVI Labeler | ✅ |
| Training Loop | ✅ |

---

## 🔬 Technologies

| Component | Technology |
|-----------|------------|
| **Forest Segmentation** | TensorFlow/Keras (mU-Net) |
| **Change Detection** | PyTorch (Siamese U-Net) |
| **Data Processing** | Rasterio, Scikit-learn |
| **Frontend** | React, Vite, TailwindCSS |
