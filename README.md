# AutoInsight - Automated Data Cleaning & Dashboard Generator

## 📌 Project Overview
AutoInsight is an automated data analytics web application built using **Python** and **Streamlit**.  
It allows users to upload datasets and automatically perform:

- Data cleaning
- Missing value handling
- Duplicate removal
- Outlier detection and optional removal
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Principal Component Analysis (PCA)
- Feature selection
- Interactive dashboard generation
- Download of cleaned and processed datasets

The goal of this project is to reduce manual effort in preparing datasets and generating useful analytical insights.

---

## 🎯 Objectives
The main objectives of AutoInsight are:

- To automate the process of dataset cleaning and preprocessing
- To help users quickly understand uploaded datasets
- To generate visual insights through EDA and dashboards
- To support feature engineering and dimensionality reduction
- To provide processed datasets ready for machine learning tasks
- To create a user-friendly no-code analytics tool

---

## 🚀 Features

### 1. Dataset Upload
- Supports **CSV** and **Excel (.xlsx)** files
- Instant dataset preview after upload

### 2. Dataset Overview
- Shows:
  - Number of rows
  - Number of columns
  - Column names
  - Data types
  - Missing values summary

### 3. Automatic Data Cleaning
- Handles missing values intelligently:
  - Numeric columns → filled using **median**
  - Categorical columns → filled using **mode**
- Removes duplicate rows
- Detects outliers using **IQR (Interquartile Range)**
- Optional outlier removal
- Generates a cleaning report

### 4. Exploratory Data Analysis (EDA)
- Numerical summary statistics
- Correlation heatmap
- Distribution analysis
- Missing value analysis
- Basic visual exploration

### 5. Data Preprocessing
- Encoding options:
  - Label Encoding
  - One-Hot Encoding
- Scaling options:
  - StandardScaler
  - MinMaxScaler

### 6. Feature Engineering & PCA
- Performs dimensionality reduction using **PCA**
- Shows transformed principal components
- Displays explained variance information

### 7. Feature Selection
- Allows selection of the most relevant features
- Helps in reducing dimensionality for modeling

### 8. Interactive Dashboard Generation
- Builds dashboards on:
  - Cleaned dataset
  - Preprocessed dataset
  - PCA dataset
  - Selected-features dataset
- Supports multiple visualizations and dashboard exploration

### 9. Dataset Download
- Download cleaned dataset
- Download preprocessed dataset
- Download transformed datasets for further use

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Plotly**
- **Scikit-learn**
- **OpenPyXL**

---

## 📂 Project Structure

```bash
AutoInsight/
│── app.py
│── requirements.txt
│── README.md
│── modules/
│   │── upload.py
│   │── cleaning.py
│   │── export.py
│   │── eda.py
│   │── dashboard.py
│   │── preprocessing.py
│   │── feature_engineering.py
│   │── feature_selection.py
│── assets/
│── venv/