# 🎓 Student Performance Prediction & Exploratory Data Analysis

## 📌 Project Overview
This end-to-end Data Analytics and Machine Learning project aims to analyze student performance metrics and predict final academic outcomes using demographic, social, and school-related features. 

By applying exploratory data analysis (EDA) and predictive modeling, the project identifies key drivers of academic success—helping educators make data-informed decisions.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (`LinearRegression`, `train_test_split`, `r2_score`, `mean_squared_error`)

---

## 📊 Key Findings & Insights
* **Correlation Analysis:** Identified strong correlations between study time, past performance (G1, G2 grades), and final academic performance (G3).
* **Impact Factors:** Factors like study frequency, Internet access, and parental education showed a positive influence on grades, while alcohol consumption and high absences negatively impacted performance.

---

## 📈 Model Performance
* **Model Used:** Linear Regression
* **R² Score:** `0.78` (Explains ~78% of variance in final student grades)
* **Mean Squared Error (MSE):** `4.504`

---

## 📂 Project Structure
```text
├── student_analysis.py       # Main Python script containing EDA, data preprocessing & ML pipeline
├── student_data.csv          # Dataset containing student academic and social metrics
└── README.md                 # Project documentation
