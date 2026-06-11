**SVHN Digit Recognition (ANN vs CNN) — Deep Learning Computer Vision**

Overview

This project builds and compares two deep learning approaches to recognize street-view house numbers (SVHN) from images:
* **Artificial Neural Networks (ANN / Fully Connected)** using flattened pixel inputs
* **Convolutional Neural Networks (CNN)** leveraging spatial patterns in images

The goal is to demonstrate why CNNs are the standard for image classification and to provide a clean, reproducible notebook 
that highlights data prep, modeling, evaluation, and insights.

Why this matters

Digit recognition is a core computer vision task with applications in:
* Automated address/house-number reading (maps, delivery logistics)
* Document processing and OCR pipelines
* Smart city / IoT visual analytics

SVHN is a realistic dataset because images contain natural scene variation (lighting, background clutter, blur), unlike simpler 
handwritten digits.

Dataset

SVHN (Street View House Numbers)
* Inputs: images of digits (0–9)
* Task: multi-class classification (10 classes)

In this project, the dataset is provided as an .h5 file with predefined train/validation/test splits.

**Project workflow**

1. **Data loading & sanity checks**
    * Load train/val/test arrays from .h5
    * Verify shapes, dtypes, and label ranges
    * Visualize samples with labels

2. **Preprocessing**
    * Normalize pixel values to [0, 1] by dividing by 255
    * ANN path: flatten images to 1D vectors
    * CNN path: add channel dimension if grayscale
    * One-hot encode labels for ANN/CNN (as needed)

3. **Modeling**

**ANN baselines (from scratch)**
* Dense layers with ReLU
* Dropout and BatchNorm used in the larger ANN to improve generalization

**CNN models (from scratch)**
* Convolution + pooling blocks to capture edges/shapes
* Deeper CNN improves feature learning
* Regularization (BatchNorm/Dropout) to stabilize training

4. **Evaluation**
    * Accuracy on test set
    * Confusion matrix
    * Classification report (precision/recall/F1 per digit)
    * Model comparison summary (ANN vs CNN)
  
**Results (high level)**
* **ANN (baseline)**: lower accuracy due to loss of spatial structure after flattening
* **CNN**: significantly higher accuracy (your best CNN achieved ~95%+ test accuracy) and stronger per-class precision/recall

**Key takeaway**: CNNs outperform ANNs on images because they preserve and learn from spatial patterns (local edges → parts → full digit shapes).

**Key insights**
* Flattening images for ANNs removes spatial relationships (where pixels are located), which hurts performance.
* CNNs learn hierarchical features automatically and generalize better to real-world image variation.
* BatchNorm + Dropout can improve stability and reduce overfitting in deeper models.

**How to run**

Option 1: Google Colab
1. Open the notebook in Colab
2. Upload or mount the .h5 dataset
3. Run all cells top-to-bottom

Option 2: Local environment
pip install -r requirements.txt
jupyter notebook

**Dependencies**
* Python 3.x
* Numpy
* Pandas (optional for reporting)
* Matplotlib/Seaborn (for plots)
* TensorFlow / Keras (modeling)
* scikit-learn (classification report, confusion matrix)
* h5py (read .h5 dataset)

**Author**

Jerome Jabson
