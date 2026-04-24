# TholdStormDX v0.0.1

A Python-based bioinformatics engine for the **determination and optimization of diagnostic cut-off points** in individual biomarkers and **multimarker panels** using advanced stochastic optimization.

<p align="center">
  <img src="https://github.com/roberto117343/TholdStormDX/blob/main/Logo%20GitHub.png" 
       alt="TholdStormDX Logo" width="600"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"/>
  <img src="https://img.shields.io/badge/Python-3.x-yellow.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg" alt="Platform"/>
</p>

<p align="center">
  <a href = "https://github.com/roberto117343/TholdStormDX/releases/download/v0.0.1/TholdStormDX.win.exe">
    <img src="https://img.shields.io/badge/Download-Windows%20Executable-blue?style=for-the-badge&logo=windows"/>
  </a>
  &nbsp;
  <a href="https://github.com/roberto117343/TholdStormDX/releases/download/v0.0.1/TholdStormDX">
    <img src="https://img.shields.io/badge/Download-Linux%20Executable-green?style=for-the-badge&logo=linux" alt="Download Linux"/>
  </a>
</p>

---

## 💡 What is TholdStormDX?

**TholdStormDX** is an advanced computational framework for deriving clinically meaningful diagnostic thresholds. It is specifically designed to address the instability, overfitting, and local minima issues commonly encountered in biomarker threshold optimization.

The system integrates:

- **Dual Simulated Annealing (global optimization)**
- **Logistic modeling (2P and 4P)**
- **Massively vectorized Monte Carlo simulations**
- **Boolean OR combinatorial logic with Max–Min balancing**

It has been validated across multiple oncological datasets, demonstrating robustness, interpretability, and clinical applicability.

---

## 🎯 The Problem

Translating continuous biomarkers into binary clinical decisions requires defining optimal cut-off points. However:

- ROC-based approaches often fail to ensure **balanced sensitivity/specificity**
- Multimarker combinations introduce **combinatorial explosion**
- Classical optimization methods are prone to **local minima and overfitting**

A robust solution requires:

- Global optimization
- Explicit control of balance (SE vs SP)
- Transparent, interpretable logic

---

## 🔬 Our Solution: A Two-Stage Architecture

TholdStormDX implements a **two-stage optimization pipeline**:

---

### **Stage 1: Individual Marker Characterization**

Each biomarker is independently analyzed using four complementary approaches:

1. **Exact Empirical Interpolation**
   - Direct computation of the intersection where Sensitivity = Specificity

2. **Logistic Model (2 Parameters)**
   - Ideal symmetric transition model

3. **Logistic Model (4 Parameters)**
   - Handles biological asymmetry and baseline noise

4. **Stochastic ThresholdXpert Method**
   - Monte Carlo-based empirical optimization

All parametric models are optimized using **Dual Simulated Annealing**, ensuring convergence to global optima and avoiding local minima.

---

### **Stage 2: Multimarker Combinatorial Optimization**

The core engine performs large-scale stochastic optimization:

- Evaluates **millions of threshold combinations**
- Uses **Boolean OR logic** for panel construction
- Applies **Max–Min balancing (precision 0.001)** to equalize Sensitivity and Specificity
- Fully vectorized → high computational efficiency

This allows identification of:

- Minimal high-performance panels
- Cases where **a single biomarker outperforms complex panels**

---

## ✨ Key Features

### 1. Global Optimization Engine
- Dual Annealing avoids local minima
- Robust across heterogeneous biomarker distributions

### 2. Massive Monte Carlo Vectorization
- Up to **10 million simulations per run**
- Efficient exploration of high-dimensional threshold spaces

### 3. Algorithmic Stability Detection
- Flags variables with **>15% threshold fluctuation**
- Prevents inclusion of noisy or unstable predictors

### 4. Clinical Parsimony Enforcement
- Identifies when **simpler models outperform complex panels**
- Reduces overfitting risk

### 5. Automatic Reporting
- Generates:
  - **PDF report** (summary of results)
  - **CSV file** (full combinatorial panel space)

---

## 🖥️ Platform & Distribution

TholdStormDX is distributed as **standalone executables**:

- 🐧 Linux executable  
- 🪟 Windows executable  

⚠️ The software is **no longer cross-platform via Java**.  
Each operating system requires its corresponding binary.

No installation of Python is required for end users.

---

## 📦 Usage Workflow

### 1. Input Data
- CSV format  
- Separator: `;`  
- Decimal: `.`  
- Last column = class (`1` positive, `0` negative)

---

### 2. Dataset Strategy
Recommended split:

- Training: ~60%  
- Validation: ~20%  
- Test: ~20%  

Test set must remain **completely unseen** until final evaluation.

---

### 3. Execution

1. Load datasets:
   - Training (required)
   - Validation (optional but recommended)
   - Test (optional, strictly for final evaluation)

2. Configure:
   - Output directory
   - Fast-Track mode (optional)

3. Run:
   - Click **Start Combo Analysis**

---

### 4. Output

- **PDF Report**
  - Individual biomarker analysis
  - Top multimarker panels

- **CSV File**
  - Full panel space
  - Thresholds
  - Metrics (Train / Val / Test)

---

## ⚠️ Important Considerations

### Overfitting Control
- Selection must rely on **Train + Validation**
- Test set is strictly for **final generalization assessment**

### OR Logic Trade-off
- Increases sensitivity
- Can reduce specificity if noisy variables are included

### Dimensionality Limit
- Recommended: **≤ 10–15 variables**
- Beyond this → exponential computational cost

---

## 📊 Validated Use Cases

The framework has been tested on:

- Breast Cancer (diagnosis & prognosis)
- Hepatocellular carcinoma
- Pulmonary nodules
- Cervical cancer

Results show:

- Cases where **single biomarkers dominate**
- Cases where **multimarker panels add value**
- Built-in detection of **unstable variables and noisy predictors**

---

## 📜 How to Cite

> *Coming Soon*

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0**.

---

## 📬 Contact

**Roberto Reinosa Fernández**  
📧 roberto117343@gmail.com  
💻 https://github.com/roberto117343
