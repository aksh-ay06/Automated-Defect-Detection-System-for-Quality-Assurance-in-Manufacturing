# Quality Inspection Project ReadMe

This repository contains the work and documentation for a Quality Inspection Project developed as part of the IENG 493C module. The project aims to automate defect detection in cell phone manufacturing using machine learning (ML) and deep learning (DL) models, adhering to the CRISP-DM framework. Below are the details and instructions for the project.

---

## Project Overview

The project addresses a critical issue in the CP-Lab manufacturing process: the lack of a quality inspection system for detecting defects in cell phone main cases. To solve this, a hands-free, non-invasive image-based inspection system was designed and implemented. This system uses an overhead camera to capture images of parts on a conveyor belt and employs ML/DL models for defect classification.

---

## Features

- **Defect Detection**: Detects visible defects such as scratches and excess material on parts.
- **Automated Inspection**: Hands-free, image-based inspection using machine learning.
- **Data Augmentation**: Includes noise addition and feature standardization for improved model performance.
- **Model Evaluation**: Detailed performance evaluation using metrics like accuracy, precision, recall, and specificity.
- **Integration Ready**: Designed for seamless deployment in manufacturing environments.

---

## CRISP-DM Workflow

1. **Business Understanding**:
   - Addressed the lack of inspection processes in CP-Lab by proposing an automated ML-based solution.
   - Designed an overhead camera setup for non-invasive defect detection.

2. **Data Understanding**:
   - Dataset contains images of good and defective parts divided into separate folders.
   - Explored dataset quality, diversity, and image resolution.

3. **Data Preparation**:
   - Preprocessed images by cropping, labeling, and normalizing pixel values.
   - Augmented data to increase the dataset size and diversity.

4. **Modeling**:
   - Trained and tested multiple ML models (KNN, Random Forest, SVM) and a deep learning model.
5. **Evaluation**:
   - Evaluated models using confusion matrices and derived metrics like sensitivity, specificity, and precision.
   - Identified deep learning as the optimal solution based on performance consistency.

6. **Deployment**:
   - Provided deployment plans for engineering teams, corporate leaders, and shop-floor employees.

---

## Files and Directories

- **/Superviseddata/**: Contains processed image datasets for ML training and testing.
- **x.pkl & y.pkl**: Pickled files storing preprocessed image and label data.
- **Deep Learning Model Code**: Includes deep learning implementation for defect detection.
- **ML Model Code**: Scripts for KNN, Random Forest, and SVM implementations.
- **Data Augmentation Code**: Methods for applying noise addition and feature standardization.

---

## Prerequisites

- **Software Requirements**:
  - Python (Jupyter Notebook/Google Colab recommended)
  - Libraries: OpenCV, TensorFlow/Keras, scikit-learn, NumPy, Matplotlib, PIL

- **Hardware Requirements**:
  - PC with moderate computational capacity for training models.
  - Camera setup for capturing real-time data in manufacturing environments.

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/aksh-ay06/Automated-Defect-Detection-System-for-Quality-Assurance-in-Manufacturing
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset:
   - Place "good" and "bad" images in respective folders under `/Superviseddata/`.
4. Run preprocessing scripts to crop, label, and augment the data.

---

## Usage

1. Preprocess the dataset by running the cropping and labeling script.
2. Train and evaluate the ML/DL models using provided scripts.
3. Use the deployment guide to integrate the model into your production system.

---

## Results

- Achieved 100% accuracy on augmented datasets with the deep learning model.
- Demonstrated the importance of data augmentation in improving model performance.
- Ensured real-time defect detection capability with non-invasive inspection.

---

## Future Work

- **Expand Dataset**: Collect diverse and larger datasets for robust training.
- **Improve Deployment**: Optimize deployment pipelines for real-time operations.
- **Explore New Models**: Test additional ML/DL models for performance comparison.

---

## Contributors

- [Akshay Patel] - Industrial Engineer & Machine Learning Enthusiast
- Interns and team members for data collection and preprocessing

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

Special thanks to the IENG 493C module instructor Mackenzie Keepers and teammates for their guidance and support throughout the project. 

For questions or suggestions, please contact [www.linkedin.com/in/aksh-ay06].
