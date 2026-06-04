import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import google.generativeai as genai

# ------------------------
# Gemini Setup
# ------------------------

import streamlit as st
import google.generativeai as genai

api_key = st.secrets["GEMINI_API_KEY"]

genai.configure(
    api_key=api_key
)

llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ------------------------
# Load Data
# ------------------------

df = pd.read_csv(
    "classroom_data.csv"
)

encoder = LabelEncoder()

df["Comfort"] = encoder.fit_transform(
    df["Comfort"]
)

X = df[
    ["Temperature",
     "Humidity",
     "Attendance"]
]

y = df["Comfort"]

model = RandomForestClassifier()

model.fit(X,y)

# ------------------------
# UI
# ------------------------

st.title(
    "AI Powered Classroom Digital Twin"
)

temperature = st.slider(
    "Temperature",
    20,
    40,
    25
)

humidity = st.slider(
    "Humidity",
    30,
    90,
    50
)

attendance = st.slider(
    "Attendance",
    0,
    50,
    35
)

# ------------------------
# Digital Twin
# ------------------------

st.subheader(
    "Digital Twin"
)

c1,c2,c3 = st.columns(3)

c1.metric(
    "Temperature",
    f"{temperature}°C"
)

c2.metric(
    "Humidity",
    f"{humidity}%"
)

c3.metric(
    "Attendance",
    attendance
)

# ------------------------
# ML Prediction
# ------------------------

pred = model.predict(
    [[temperature,
      humidity,
      attendance]]
)

status = encoder.inverse_transform(
    pred
)[0]

st.subheader(
    "AI Prediction"
)

st.success(status)

# ------------------------
# GenAI Assistant
# ------------------------

st.subheader(
    "Chat with Digital Twin"
)

user_question = st.text_input(
    "Ask anything about the classroom"
)

if user_question:

    prompt = f"""
You are an AI Digital Twin Assistant.

Classroom Status:

Temperature: {temperature}
Humidity: {humidity}
Attendance: {attendance}

Prediction: {status}

Question:
{user_question}

Give a simple explanation and recommendations.
"""

    response = llm.generate_content(
        prompt
    )

    st.info(
        response.text
    )

# ------------------------
# Agentic AI
# ------------------------

st.subheader(
    "Autonomous Agent Actions"
)

actions = []

if attendance < 20:
    actions.append(
        "Low Attendance Alert"
    )

if temperature > 30:
    actions.append(
        "Cooling Recommendation"
    )

if humidity > 70:
    actions.append(
        "Ventilation Recommendation"
    )

for action in actions:
    st.warning(action)

if len(actions)==0:
    st.success(
        "No action required"
    )

# ------------------------
# Auto Report
# ------------------------

if st.button(
    "Generate Smart Report"
):

    report_prompt = f"""
Generate a professional classroom report.

Temperature: {temperature}
Humidity: {humidity}
Attendance: {attendance}
Status: {status}

Actions:
{actions}
"""

    report = llm.generate_content(
        report_prompt
    )

    st.download_button(
        "Download Report",
        report.text,
        file_name="classroom_report.txt"
    )
