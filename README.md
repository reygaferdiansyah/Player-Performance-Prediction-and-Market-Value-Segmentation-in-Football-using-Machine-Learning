# ⚽ Player Performance Prediction and Market Value Segmentation in Football using Machine Learning

This project leverages **machine learning** and **unsupervised learning (clustering)** to predict football player performance and segment them based on market value. The dataset includes 1,748 players from top European leagues in the 2024–2025 season.

---

## 📌 Project Objectives

1. **Predict key performance metrics** (goals, assists, defensive actions, goalkeeper saves) using supervised learning models.
2. **Segment players by market value** using clustering to support strategic decisions in scouting and transfers.

---

## 🧠 Features & Data Sources


- 🔗 [**SofaScore**](https://www.sofascore.com): Provides player match performance metrics (goals, assists, passes, tackles, saves, etc.)
- 🔗 [**Transfermarkt**](https://www.transfermarkt.com): Provides player demographic and financial data (age, position, market value, transfer history, etc.)
  
### 📊 Match Performance Data – *SofaScore* 
- **Offensive**: Goals, xG, Big Chances, Shots, Goal Conversion %
- **Playmaking**: Assists, Key Passes, Accurate Pass %, Dribbles
- **Defensive**: Tackles, Interceptions, Clearances, Errors
- **Goalkeeping**: Saves, Clean Sheets, Penalties Saved
- **Overall Rating**: Average SofaScore

### 💰 Market & Profile Data – *Transfermarkt*
- League, Team, Player Name, Age, Position, Market Value

> 📅 Data reflects the **2024–2025 season**, covering performance and transfer market trends across major European leagues.

---

## 🔍 Methodology

### 🛠️ Data Preprocessing
- Missing value handling
- Market value conversion
- Positional standardization
- Feature engineering: goal contribution, xG efficiency, etc.
- Mean encoding for categorical features
- Feature scaling (MinMaxScaler / StandardScaler)

### 🤖 Modeling (Performance Prediction)
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **MLPRegressor (Neural Network)**

### 📈 Clustering (Market Segmentation)
- **K-Means Clustering**
- **Agglomerative Clustering**
- Elbow & Silhouette Score for optimal k
- PCA and t-SNE for visualization

---

## 📈 Model Evaluation Summary

The performance of each model was evaluated based on regression metrics (**R², MAE, MSE, RMSE**) and runtime (**training time**, **prediction latency**) across four key prediction tasks:

---

### 🔹 Top Scorer Prediction

| Model           | R² (Test) | MAE | MSE | RMSE | Training Time (s) | Prediction Time (s) |
|-----------------|-----------|-----|-----|------|--------------------|----------------------|
| **MLPRegressor**| 0.9386    | 0.156 | 0.068 | 0.260 | 23.58              | 0.0020               |
| Ridge           | 0.8521    | 0.242 | 0.163 | 0.404 | 0.16               | 0.0019               |
| Lasso           | 0.8522    | 0.241 | 0.163 | 0.404 | 0.13               | 0.0019               |
| Linear          | 0.8507    | 0.244 | 0.165 | 0.406 | 0.00               | 0.0020               |

✅ **Recommendation**: Use **MLPRegressor** for top scorer prediction when accuracy is the top priority.

---

### 🔹 Top Assist Prediction

| Model           | R² (Test) | MAE | MSE | RMSE | Training Time (s) | Prediction Time (s) |
|-----------------|-----------|-----|-----|------|--------------------|----------------------|
| Linear          | 0.7095    | 0.343 | 0.302 | 0.549 | 0.03               | 0.0021               |
| Ridge           | 0.7070    | 0.346 | 0.304 | 0.552 | 0.29               | 0.0019               |
| Lasso           | 0.7071    | 0.341 | 0.304 | 0.552 | 0.20               | 0.0018               |
| MLPRegressor    | 0.7094    | 0.355 | 0.302 | 0.550 | 16.92              | 0.0020               |

✅ **Recommendation**: Use **Linear or Lasso Regression** for fast and reliable assist prediction.

---

### 🔹 Top Defensive Prediction

| Model           | R² (Test) | MAE | MSE | RMSE | Training Time (s) | Prediction Time (s) |
|-----------------|-----------|-----|-----|------|--------------------|----------------------|
| Lasso           | 0.9907    | 2.36 | 10.93 | 3.31 | 0.14               | 0.0018               |
| Linear          | 0.9905    | 2.41 | 11.06 | 3.33 | 0.00               | 0.0020               |
| Ridge           | 0.9905    | 2.42 | 11.09 | 3.33 | 0.17               | 0.0019               |
| MLPRegressor    | 0.9874    | 2.96 | 14.75 | 3.84 | 79.20              | 0.0021               |

✅ **Recommendation**: Use **Lasso or Linear Regression** for accurate and fast defensive performance prediction.

---

### 🔹 Top Saves (Goalkeeper)

| Model           | R² (Test) | MAE | MSE | RMSE | Training Time (s) | Prediction Time (s) |
|-----------------|-----------|-----|-----|------|--------------------|----------------------|
| Linear          | 0.9905    | 0.701 | 4.02 | 2.01 | 0.00               | 0.0018               |
| Lasso           | 0.9904    | 0.689 | 4.04 | 2.01 | 0.13               | 0.0016               |
| Ridge           | 0.9904    | 0.704 | 4.06 | 2.02 | 0.16               | 0.0016               |
| MLPRegressor    | 0.9853    | 0.914 | 6.22 | 2.49 | 38.85              | 0.0020               |

✅ **Recommendation**: Use **Linear or Lasso Regression** for efficient and stable goalkeeper performance modeling.

---

## 🧩 Market Value Segmentation (Clustering)

| **Visualization Aspect** | **K-Means Clustering**                                                                 | **Agglomerative Clustering**                                                                 |
|--------------------------|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| **PCA (Linear)**         | Clear cluster separation, symmetrical, and visually interpretable                     | Cluster overlap, especially between Low and High Market segments                             |
| **t-SNE (Nonlinear)**    | Dominance of Low Market color, with overlapping clusters and less organic structure   | Better spatial separation, clearer distinction of High and Mid Market segments              |
| **Pie Chart Distribution**| 69.1% Low, 20.9% High, 10% Mid → balanced and reflects market diversity               | 72.3% Low, rest skewed → more conservative segmentation                                      |

✅ **Clustering Recommendation**:  
Use **K-Means Clustering** for player market segmentation as it produces **balanced, interpretable**, and **system-ready clusters**.


## 🔧 Tech Stack

- **Language**: Python 3.10  
- **Libraries**:  
  `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`

---



