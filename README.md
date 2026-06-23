# Customer Churn Prediction and Management System

## Overview

The Customer Churn Prediction and Management System is a Machine Learning-based web application developed using Python and Streamlit. The system helps organizations identify customers who are likely to leave (churn) and provides risk analysis for customer retention.

The application allows administrators to manage customer records, predict churn probability, and monitor customer risk levels through an interactive dashboard.

---

## Features

* Customer Churn Prediction using Machine Learning
* Customer Management (Add, Update, Delete)
* Duplicate Customer ID Detection
* Dynamic Churn Probability Calculation
* Automatic Risk Level Classification
* Customer Database Management
* Feature Importance Visualization
* Interactive Dashboard
* Real-time Prediction Updates

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Joblib

---

## Dataset Attributes

The model uses the following customer attributes:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract Type
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

---

## Risk Levels

| Churn Probability | Risk Level |
| ----------------- | ---------- |
| 0% - 29%          | LOW 🟢     |
| 30% - 69%         | MEDIUM 🟡  |
| 70% - 100%        | HIGH 🔴    |

---

## Project Workflow

1. Admin enters customer details.
2. Machine Learning model predicts churn probability.
3. Risk level is generated automatically.
4. Customer data is stored in the database.
5. Admin can update or delete customer records.
6. Predictions are recalculated whenever customer information changes.

---

## How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Future Enhancements

* AI-powered churn explanation
* Customer retention recommendations
* Email notification system
* Customer segmentation using clustering
* Advanced analytics dashboard
* Cloud database integration

---

## Author

Bindhu Katuri

B.Tech Computer Science and Engineering

Machine Learning Project

