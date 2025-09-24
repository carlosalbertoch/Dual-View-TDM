# Dual-View Tomographic Diffraction Microscopy (TDM)

[![Python](https://img.shields.io/badge/python-3.12.2-blue.svg)](https://python.org) [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]() [![Paper](https://img.shields.io/badge/paper-2025-orange.svg)]()

> **Authors:** Carlos Alberto Chacón Ávila, Nicolas Verrier, Matthieu Debailleul, Bruno Colicchio, Olivier Haeberlé

This repository contains the complete computational pipeline for **Dual-View Tomographic Diffraction Microscopy**, providing an end-to-end solution for 3D microscopic reconstruction and visualization.

<div align="center">
  <img src="https://raw.githubusercontent.com/carlosalbertoch/Dual-View-TDM/main/assets/setup.jpg" alt="Pipeline Setup" width="600"/>
</div>
---

## Overview

The pipeline implements a comprehensive workflow to:

• **Align** two opposing 3D microscopy stacks 
• **Segment and fuse** volumetric data plane-by-plane using frequency/gradient analysis  
• **Convert** fused volumes into 3D meshes (GLB/FBX formats)  



---

## 🧪 Live Demo — *Navicula* Dataset

Experience our reference implementation with the **Navicula** diatom dataset:

**🔗 [Interactive 3D Viewer](https://navicula.carloschacon.fr/)**

This live preview showcases the complete reconstruction pipeline from raw microscopy data to interactive 3D visualization.

---

## 📥 Getting Started

### Prerequisites

**Python 3.12.2** with the following core packages:

```bash
numpy
tifffile  
matplotlib
tqdm
trimesh
SimpleITK
```

### Dataset Download

Download the required **TIFF stacks** from our data portal:

**📂 [Dataset Portal](https://www.your-dataset-portal.org)**

Required files:
- `diatom_T1_indice.tif` - T1 refractive index reconstruction  
- `diatom_T1_Absorption.tif` - T1 absorption reconstruction  
- `diatom_T2_indice.tif` - T2  refractive index reconstruction  
- `diatom_T2_Absorption.tif` - T2 absorption reconstruction  

---

## 📁 Project Structure

```
dual-view-tdm/
├── images/
│   ├── diatom_T1_indice.tif
│   ├── diatom_T1_Absorption.tif
│   ├── diatom_T2_indice.tif
│   └── diatom_T2_Absorption.tif
├── utils/
│   └── back_fusion_methods.py
├── alignment.py
├── fusion.py
├── offset_index.py
├── background.py
├── mesh_voxels.py
├── back_adaptive_sustraction.py
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 🚀 Pipeline Execution

Execute the following scripts in sequence:

### 1) Stack Alignment
```bash
python alignment.py
```
aligns T2 (and its absorption volume) to T1 by extracting a centered ROI, performing translation + affine registration with Elastix, propagating the transform to the full volumes, and saving both the aligned stacks and registration parameters

<div align="center">
  <img src="https://raw.githubusercontent.com/carlosalbertoch/Dual-View-TDM/main/assets/alignment.jpg" alt="Alignment Process" width="600"/>
</div>

### 2) Volume Fusion
```bash
python fusion.py
```
Merges aligned stacks 

### 3) Index Offset Correction
```bash
python offset_index.py
```
Applies refractive index calibration 

### 4) Background Processing
```bash
python background.py
```
Generates adaptive masks and removes reconstruction artifacts.

### 5) 3D Mesh Generation
```bash
python mesh_voxels.py
```
Creates meshes using marching cubes algorithm with topology optimization.

---

## 🔬 Visualization

**Quick Preview:** Drag & drop generated `.glb` files into the [Babylon.js Sandbox](https://sandbox.babylonjs.com/) for immediate 3D visualization.

---

## 📖 Citation

If this work contributes to your research, please cite:

```bibtex
@article{ChaconAvila2025DualViewTDM,
  title   = {Dual-View Tomographic Diffraction Microscopy},
  author  = {Chacón Ávila, Carlos Alberto and Verrier, Nicolas and 
             Debailleul, Matthieu and Colicchio, Bruno and 
             Haeberlé, Olivier},
  year    = {2025},
  journal = {Manuscript in preparation},
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

We extend our gratitude to the **Université de Haute-Alsace (UHA)** and our collaborators for valuable discussions and data contributions that made this research possible.

---

<div align="center">
  <strong>Learn more about tomographic diffraction microscopy</strong><br>
  <a href="https://your-documentation.com">paper</a> • 
  <a href="mailto:carlos-alberto.chacon-avila@uha.fr">Contact</a>
</div>
