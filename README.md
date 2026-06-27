# Global Freedom Index 2016: Predictive Modeling & Dimensionality Reduction

## 📌 Project Overview
This repository features an advanced predictive modeling and data analytics pipeline focused on the **2016 Global Freedom Index** dataset compiled by Freedom House. Moving beyond descriptive statistics, this project aims to predict a country's Civil Liberties (CL) score strictly based on its structural socio-political parameters and regional indicators. The analysis serves as a comprehensive benchmark for various machine learning algorithms dealing with complex, real-world socio-economic data.

## 🛠️ Technological Stack
*   **Language:** Python (3.x)
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn (Sklearn)
*   **Data Visualization:** Matplotlib, Seaborn

## ⚙️ Data Engineering & Preprocessing
To ensure model integrity and prevent algorithmic bias, rigorous preprocessing steps were implemented:
*   **Dummy Variable Trap Avoidance:** Categorical features (e.g., Status, Region) were transformed using One-Hot Encoding with `drop_first=True` to eliminate perfect multicollinearity.
*   **Feature Scaling:** Standardized all numerical inputs using `StandardScaler` (mean = 0, variance = 1) to provide an equal-weight foundation for distance-based and gradient-descent algorithms.
*   **Data Splitting:** Applied a 75/25 Train-Test split (`random_state=34`) to evaluate model generalization capabilities.

## 🤖 Modeling Arsenal & Benchmarking
A massive benchmarking environment was engineered to test 13 distinct machine learning algorithms across multiple paradigms:
1.  **Linear & Regularized Models:** OLS, Ridge, Lasso, ElasticNet
2.  **Dimensionality Reduction:** Principal Component Regression (PCR), Partial Least Squares (PLS)
3.  **Distance-Based & SVM:** K-Nearest Neighbors (KNN), Support Vector Regression (SVR)
4.  **Tree & Ensemble Methods:** Decision Tree (CART), Bagging, Random Forest, Gradient Boosting Machine (GBM)
5.  **Deep Learning:** Artificial Neural Networks (ANN / Multi-Layer Perceptron)

## 📊 Key Analytical Insights

### 1. The R² vs. MAPE Trade-off
Extensive hyperparameter tuning (via 10-Fold Cross-Validation) revealed a fascinating statistical trade-off:
*   **Variance Explanation Champion:** **PCR ($n=6$)** achieved the highest **$R^2$ score of 0.9090**. By projecting the data into orthogonal principal components, it successfully eliminated multicollinearity noise and captured 91% of the total global variance in civil liberties.
*   **Prediction Error Champion:** **PLS** achieved the lowest Mean Absolute Percentage Error (**MAPE: 15.16%**). 
*   *Conclusion:* PCR was crowned the primary model because prioritizing structural variance explanation ($R^2$) is statistically more valuable for sociopolitical datasets than chasing microscopic improvements in MAPE.

### 2. Feature Importance: The Blueprint of Freedom
By extracting feature importance metrics from the **Random Forest** algorithm and analyzing the $L1$ regularization coefficients of the **Lasso** model, the underlying socio-political blueprint was decoded:
*   **Political Rights (PR)** emerged as the absolute dominant catalyst. The algorithms mathematically proved that robust political rights (transparent elections, democratic pluralism) are the non-negotiable prerequisites for civil liberties. 
*   Regional and institutional variables heavily influenced the models, proving that civil liberty is not an isolated metric but a byproduct of deeply integrated structural systems.

## 📁 Repository Structure
*   `freedom.csv`: The core 2016 dataset.
*   `Project_Code.py`: The automated preprocessing, training, and 13-model benchmarking pipeline.
*   `Report.pdf`: Comprehensive statistical and sociopolitical analysis report detailing model behaviors, residual plots, and insights.
