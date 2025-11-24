# Hierarchy-Model: Hierarchical Classification of Sleep Apnea Events

This repository contains a hierarchical image–based classifier for sleep-related breathing events using transfer learning with a fine-tuned ResNet50 backbone.

The system is organized into two stages:

- **Model H1 (Binary Classifier):**  
  Classifies each event as either **Apnea** or **Hypopnea**.

- **Model H2 (Multi-Class Classifier):**  
  Further classifies events into **Hypopnea**, **Central Sleep Apnea (CSA)**, **Mixed Sleep Apnea (MSA)**, or **Obstructive Sleep Apnea (OSA)**.

Both models are implemented using transfer learning on ResNet50, where the pretrained network is fine-tuned on a curated dataset of apnea-related images.

---

## Dataset

The dataset is split into three parts:

- **Training set (70%)**
- **Validation set (30%)**
- **Test set (30 people)**

Each split maintains class imbalance similar to the original distribution across the five classes: Apnea, Hypopnea, MSA, CSA, and OSA.

Example visualization of the dataset distribution:

![Dataset distribution](/mnt/data/Screenshot 2025-11-24 123335.png)

> Note: When using this README in your GitHub repo, replace the image path with the correct relative path to your saved figure (for example, `images/dataset_distribution.png`).

---

## Project Structure

```text
Hierarchy-Model/
├── app.py                 # Streamlit application entry point
├── models/
│   ├── H1_best.keras      # Fin_
```
