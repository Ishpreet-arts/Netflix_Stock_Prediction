#python -m streamlit run "C:/Users/HP/OneDrive/Desktop/ML project/ML_Project.py"

#python -m streamlit run "C:\Users\HP\OneDrive\Desktop\Netflix_Stock_Prediction\ML_Project.py"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Load and clean data
df = pd.read_csv("Netflix_stock_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.dropna(inplace=True)

# Create tabbed layout
tab1, tab2 = st.tabs([" ML Project", " LLaMA Chatbot"])

# --- ML PROJECT TAB ---
with tab1:
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Visualize", "Train Model", "Predict"])

    # Home Page
    if page == "Home":
        st.title("Netflix Stock Price Explorer")
        st.write("""
        This is a beginner-friendly machine learning app that uses **Linear Regression** to predict Netflix's closing stock prices.

        **What you can do here:**
        - View stock data
        - See simple visualizations
        - Train a basic ML model
        - Make your own predictions
        """)
        st.subheader("Sample Data")
        st.write(df.head())
        st.info(f"Data Range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    # Visualize Tab
    elif page == "Visualize":
        st.title("Stock Data Visualizations")

        st.subheader("Closing Price Over Time")
        fig, ax = plt.subplots()
        ax.plot(df['Date'], df['Close'], color='green')
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        st.pyplot(fig)

        st.subheader("Volume Traded Over Time")
        fig2, ax2 = plt.subplots()
        ax2.plot(df['Date'], df['Volume'], color='blue')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volume")
        st.pyplot(fig2)

    # Model Training Tab
    elif page == "Train Model":
        st.title("Train Your Model")

        # Use simple features
        features = ['Open', 'High', 'Low', 'Volume']
        target = 'Close'

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Model Performance")
        st.write("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
        st.write("R² Score:", round(r2_score(y_test, y_pred), 2))

        st.subheader("Actual vs Predicted (First 10 Samples)")
        results = pd.DataFrame({"Actual": y_test[:10].values, "Predicted": y_pred[:10]})
        st.write(results)

    # Prediction Tab
    elif page == "Predict":
        st.title("Predict Closing Price")
        st.write("Enter values to predict Netflix's closing stock price:")

        open_val = st.number_input("Opening Price", min_value=0.0, value=300.0)
        high_val = st.number_input("High Price", min_value=0.0, value=310.0)
        low_val = st.number_input("Low Price", min_value=0.0, value=295.0)
        volume_val = st.number_input("Volume", min_value=0.0, value=10000000.0)

        if st.button("Predict Now"):
            input_data = np.array([[open_val, high_val, low_val, volume_val]])
            model = LinearRegression()
            model.fit(df[['Open', 'High', 'Low', 'Volume']], df['Close'])
            prediction = model.predict(input_data)
            st.success(f"Predicted Closing Price: ${prediction[0]:.2f}")


# --- LLAMA CHATBOT TAB ---
with tab2:
    st.title("LLaMA Chatbot with Groq")

    api_key = st.text_input("Enter your Groq API Key", type="password")

    if api_key:
        llm = ChatOpenAI(
            model="llama-3.1-8b-instant",
            temperature=1,
            openai_api_key=api_key,
            openai_api_base="https://api.groq.com/openai/v1"
        )

        prompt = st.text_input("Enter your message:")

        if prompt:
            with st.spinner("Thinking..."):
                response = llm.predict_messages([HumanMessage(content=prompt)])
                st.success("Response:")
                st.write(response.content)
