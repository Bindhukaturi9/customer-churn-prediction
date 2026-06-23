import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------
# PAGE CONFIG
# --------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Management System")
show_high = st.checkbox(
    "Show High Risk Customers Only"
)

# --------------------------
# DATABASE
# --------------------------

DATABASE = "customers.csv"

if not os.path.exists(DATABASE):

    columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Prediction",
        "Probability"
    ]
    

    pd.DataFrame(columns=columns).to_csv(
        DATABASE,
        index=False
    )

# --------------------------
# LOAD MODEL
# --------------------------

model, feature_list = joblib.load(
    "churn_model.pkl"
)

# --------------------------
# ENCODING MAPS
# --------------------------

gender_map = {
    "Female":0,
    "Male":1
}

partner_map = {
    "No":0,
    "Yes":1
}

dependents_map = {
    "No":0,
    "Yes":1
}

phone_map = {
    "No":0,
    "Yes":1
}

multiple_map = {
    "No":0,
    "No phone service":1,
    "Yes":2
}

internet_map = {
    "DSL":0,
    "Fiber optic":1,
    "No":2
}

service_map = {
    "No":0,
    "No internet service":1,
    "Yes":2
}

contract_map = {
    "Month-to-month":0,
    "One year":1,
    "Two year":2
}

paperless_map = {
    "No":0,
    "Yes":1
}

payment_map = {
    "Bank transfer (automatic)":0,
    "Credit card (automatic)":1,
    "Electronic check":2,
    "Mailed check":3
}

# --------------------------
# DASHBOARD
# --------------------------

df_dashboard = pd.read_csv(
    DATABASE
)

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "Total Customers",
        len(df_dashboard)
    )

with c2:

    if len(df_dashboard)>0:

        churn_count = len(
            df_dashboard[
                df_dashboard["Prediction"]==1
            ]
        )

    else:
        churn_count = 0

    st.metric(
        "Churn Customers",
        churn_count
    )

with c3:

    stay_count = len(df_dashboard)-churn_count

    st.metric(
        "Stay Customers",
        stay_count
    )

with c4:

    if len(df_dashboard)>0:

        avg_prob = round(
            df_dashboard[
                "Probability"
            ].mean(),
            2
        )

    else:
        avg_prob = 0

    st.metric(
        "Avg Churn %",
        avg_prob
    )

st.divider()

# --------------------------
# CUSTOMER FORM
# --------------------------

st.subheader("➕ Add Customer")

customer_id = st.text_input(
    "Customer ID"
)

col1,col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female","Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0,1]
    )

    partner = st.selectbox(
        "Partner",
        ["No","Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No","Yes"]
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0
    )

    phone = st.selectbox(
        "Phone Service",
        ["No","Yes"]
    )

    multiple = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "No phone service",
            "Yes"
        ]
    )

    internet = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

with col2:

    security = st.selectbox(
        "Online Security",
        [
            "No",
            "No internet service",
            "Yes"
        ]
    )

    backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "No internet service",
            "Yes"
        ]
    )

    protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "No internet service",
            "Yes"
        ]
    )

    support = st.selectbox(
        "Tech Support",
        [
            "No",
            "No internet service",
            "Yes"
        ]
    )

    tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "No internet service",
            "Yes"
        ]
    )

    movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "No internet service",
            "Yes"
        ]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        [
            "No",
            "Yes"
        ]
    )

payment = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0
)

total = tenure * monthly

st.write(
    f"Total Charges : ₹ {total:.2f}"
)

# --------------------------
# PREDICTION
# --------------------------

if st.button("Predict & Save"):
    
    # Customer ID Validation

    if customer_id.strip() == "":

        st.error(
            "Please enter Customer ID"
        )

        st.stop()

    # Load Database

    df = pd.read_csv(DATABASE)
    st.write("DATABASE FILE =", DATABASE)
    st.write("ROWS BEFORE SAVE =", len(df))
    st.write(df["customerID"].tolist())

    current_id = (
        str(customer_id)
        .strip()
        .lower()
    )

    customer_ids = set(

        df["customerID"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()

    )

    # Duplicate Check

    if current_id in customer_ids:

        st.error(
            "Customer ID already exists!"
        )

        st.stop()

    # Prepare Data For Prediction

    customer = [[

        gender_map[gender],
        senior,
        partner_map[partner],
        dependents_map[dependents],
        tenure,
        phone_map[phone],
        multiple_map[multiple],
        internet_map[internet],
        service_map[security],
        service_map[backup],
        service_map[protection],
        service_map[support],
        service_map[tv],
        service_map[movies],
        contract_map[contract],
        paperless_map[paperless],
        payment_map[payment],
        monthly,
        total

    ]]

    # Prediction

    prediction = model.predict(
        customer
    )

    probability = model.predict_proba(
        customer
    )

    churn_prob = round(
        probability[0][1] * 100,
        2
    )
    #_________________________________
    if churn_prob >= 70:
            
        risk = "HIGH 🔴"

    elif churn_prob >= 30:

        risk = "MEDIUM 🟡"

    else:

        risk = "LOW 🟢"

        st.info(
            f"Risk Level : {risk}"
        )

    # Result

    if prediction[0] == 1:

        st.error(
            f"Customer will Churn ({churn_prob}%)"
        )

    else:

        st.success(
            f"Customer will Stay ({100 - churn_prob}%)"
        )

    # Save Customer

    new_row = {

        "customerID": current_id,

        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,

        "Prediction": prediction[0],
        "Probability": churn_prob,
        "RiskLevel": risk

    }

    df = pd.concat(
        [df, pd.DataFrame([new_row])],
        ignore_index=True
    )

    df.to_csv(
        DATABASE,
        index=False
    )

    st.success(
        "Customer Saved Successfully"
    )
# --------------------------
# VIEW DATABASE
# --------------------------

st.divider()

st.subheader("📁 Customer Database")
try:

    df = pd.read_csv(DATABASE)

    update_mode = st.checkbox(
        "Enable Update Mode"
    )

    edited_df = st.data_editor(
        df,
        disabled=not update_mode,
        use_container_width=True,
        num_rows="fixed",
        column_config={

            "gender": st.column_config.SelectboxColumn(
                "gender",
                options=["Female", "Male"]
            ),

            "Partner": st.column_config.SelectboxColumn(
                "Partner",
                options=["No", "Yes"]
            ),

            "Dependents": st.column_config.SelectboxColumn(
                "Dependents",
                options=["No", "Yes"]
            ),

            "PhoneService": st.column_config.SelectboxColumn(
                "PhoneService",
                options=["No", "Yes"]
            ),

            "MultipleLines": st.column_config.SelectboxColumn(
                "MultipleLines",
                options=[
                    "No",
                    "No phone service",
                    "Yes"
                ]
            ),

            "InternetService": st.column_config.SelectboxColumn(
                "InternetService",
                options=[
                    "DSL",
                    "Fiber optic",
                    "No"
                ]
            ),

            "Contract": st.column_config.SelectboxColumn(
                "Contract",
                options=[
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            ),

            "PaperlessBilling": st.column_config.SelectboxColumn(
                "PaperlessBilling",
                options=[
                    "No",
                    "Yes"
                ]
            ),

            "PaymentMethod": st.column_config.SelectboxColumn(
                "PaymentMethod",
                options=[
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check"
                ]
            ),
            "SeniorCitizen": st.column_config.SelectboxColumn(
                    "SeniorCitizen",
                    options=[0, 1]
            ),

            "OnlineSecurity": st.column_config.SelectboxColumn(
                    "OnlineSecurity",
                    options=[
                        "No",
                        "No internet service",
                        "Yes"
                    ]
            ),

            "OnlineBackup": st.column_config.SelectboxColumn(
                    "OnlineBackup",
                    options=[
                        "No",
                        "No internet service",
                        "Yes"
                    ]
            ),

            "DeviceProtection": st.column_config.SelectboxColumn(
                    "DeviceProtection",
                    options=[
                        "No",
                        "No internet service",
                        "Yes"
                    ]
            ),

            "TechSupport": st.column_config.SelectboxColumn(
                    "TechSupport",
                    options=[
                        "No",
                        "No internet service",
                        "Yes"
                    ]
            ),

            "StreamingTV": st.column_config.SelectboxColumn(
                    "StreamingTV",
                    options=[
                        "No",
                        "No internet service",
                        "Yes"
                    ]
            ),

            "StreamingMovies": st.column_config.SelectboxColumn(
                    "StreamingMovies",
                    options=[
                        "No",
                        "No internet service",
                        "Yes"
                    ]
            ),
        }
    )

    if update_mode:

        if st.button("Update Database"):

            edited_df["TotalCharges"] = (
                edited_df["tenure"].astype(float)
                * edited_df["MonthlyCharges"].astype(float)
            )

            predictions = []
            probabilities = []
            risk_levels = []

            for _, row in edited_df.iterrows():

                customer = [[
                    gender_map[str(row["gender"]).strip()],
                    int(row["SeniorCitizen"]),
                    partner_map[str(row["Partner"]).strip()],
                    dependents_map[str(row["Dependents"]).strip()],
                    float(row["tenure"]),
                    phone_map[str(row["PhoneService"]).strip()],
                    multiple_map[str(row["MultipleLines"]).strip()],
                    internet_map[str(row["InternetService"]).strip()],
                    service_map[str(row["OnlineSecurity"]).strip()],
                    service_map[str(row["OnlineBackup"]).strip()],
                    service_map[str(row["DeviceProtection"]).strip()],
                    service_map[str(row["TechSupport"]).strip()],
                    service_map[str(row["StreamingTV"]).strip()],
                    service_map[str(row["StreamingMovies"]).strip()],
                    contract_map[str(row["Contract"]).strip()],
                    paperless_map[str(row["PaperlessBilling"]).strip()],
                    payment_map[str(row["PaymentMethod"]).strip()],
                    float(row["MonthlyCharges"]),
                    float(row["TotalCharges"])
                ]]

                prediction = model.predict(customer)[0]

                probability = round(
                    model.predict_proba(customer)[0][1] * 100,
                    2
                )

                if probability >= 70:
                    risk = "HIGH 🔴"
                elif probability >= 30:
                    risk = "MEDIUM 🟡"
                else:
                    risk = "LOW 🟢"

                predictions.append(prediction)
                probabilities.append(probability)
                risk_levels.append(risk)

            edited_df["Prediction"] = predictions
            edited_df["Probability"] = probabilities
            edited_df["RiskLevel"] = risk_levels

            edited_df.to_csv(
                DATABASE,
                index=False
            )

            st.success(
                " Database Updated Successfully!👍  please refresh the page 🔁 "
            )

except Exception as e:

    st.warning(
        f"Error: {e}"
    )




st.divider()

st.subheader(
    "Feature Importance"
)

try:

    imp = pd.read_csv(
        "feature_importance.csv"
    )

    st.bar_chart(
        imp.set_index(
            "Feature"
        )
    )

except:

    st.warning(
        "feature_importance.csv not found. Run churn3.py first."
    )

# --------------------------
# SEARCH CUSTOMER
# --------------------------
st.divider()

search_id = st.text_input(
    "Enter Customer ID"
)

if st.button("Search"):
    
    df = pd.read_csv(DATABASE)

    df["customerID"] = (
        df["customerID"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    search_value = (
        str(search_id)
        .strip()
        .lower()
    )

    result = df[
        df["customerID"] == search_value
    ]

    if len(result) > 0:

        st.success(
            "Customer Found"
        )

        st.dataframe(
            result,
            use_container_width=True
        )

    else:

        st.error(
            "Customer Not Found"
        )

# --------------------------
# DELETE CUSTOMER
# --------------------------

st.divider()

delete_id = st.text_input(
    "Customer ID To Delete"
)

if st.button("Delete"):
    
    df = pd.read_csv(DATABASE)

    df["customerID"] = (
        df["customerID"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    delete_value = (
        str(delete_id)
        .strip()
        .lower()
    )

    if delete_value in df["customerID"].values:

        df = df[
            df["customerID"] != delete_value
        ]

        df.to_csv(
            DATABASE,
            index=False
        )

        st.success(
            "Customer Deleted Successfully"
        )

    else:

        st.error(
            "Customer Not Found"
        )

# --------------------------
# DOWNLOAD DATABASE
# --------------------------

st.divider()

with open(
    DATABASE,
    "rb"
) as file:

    st.download_button(
        label="⬇ Download Customer Database",
        data=file,
        file_name="customers.csv",
        mime="text/csv"
    )