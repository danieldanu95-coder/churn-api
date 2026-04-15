#  Telecom Customer Churn Prediction


---

##  Project Highlights

*  Predicts **customer churn (Yes/No)** using ML models
* Built on real-world telecom dataset (7K+ customers)
*  Deployed as an interactive **Streamlit web app**
*  Focused on **high recall** to catch maximum churn cases
*  Compared multiple models & selected best-performing one

---
## 🌐 Live Demo

**Try the app here:**
🔗 https://churn-api-2fv75bkvkm5fzzvzt9y4rp.streamlit.app/#prediction-result

---

## Problem Statement

Customer churn is one of the biggest challenges in subscription-based businesses.

**Goal:**
Identify customers who are likely to leave the telecom service.

 **Why it matters:**

* Retaining customers is cheaper than acquiring new ones
* Early detection enables proactive retention strategies
* Direct impact on revenue and growth

---

##  Dataset

*  **Source:** Kaggle Telco Customer Churn Dataset
*  **Records:** 7043 customers
*  **Features:** 21
*  **Target:** `Churn (Yes/No)`

---

##  Tech Stack

<p>
Python • Pandas • NumPy • Scikit-learn • Matplotlib • Seaborn • Streamlit
</p>

---

##  Models & Evaluation

| Model                 | Accuracy | Recall (Churn) | Precision |
| --------------------- | -------- | -------------- | --------- |
| ✅ Logistic Regression | 0.73     | 0.79           | 0.50      |
| Random Forest         | 0.63     | 0.43           | 0.63      |
| Decision Tree         | 0.69     | 0.84           | 0.45      |

###  Final Model: Logistic Regression

✔ Best balance between performance and interpretability
✔ Strong recall → captures more churn customers

---

##  Application Features

*  Real-time churn prediction
*  Simple and intuitive UI
* Instant results based on user input
* Business-focused output

---

##  Project Structure

```
├── notebook/
├── app.py
├── streamlit_app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── src/
```

---

## ML Pipeline

1. Data Cleaning & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training & Evaluation
5. Model Selection
6. Deployment with Streamlit

---

##  Future Improvements


*  Interactive dashboards (EDA inside app)
*  Model explainability (SHAP / LIME)
* REST API deployment (Flask / FastAPI)

---

##  If you like this project

Give it a ⭐ on GitHub and feel free to connect!

---

Author

Daniel Arulvijayan (Aspiring Data Scientist | ML Enthusiast)

LinkedIn: https://www.linkedin.com/in/daniel-arulvijayan/

Email: danieldanu95@gmail.com



