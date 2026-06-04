import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------
# Load Dataset
# ---------------------

df = pd.read_csv("classroom_data.csv")

encoder = LabelEncoder()

df["Comfort"] = encoder.fit_transform(df["Comfort"])
print(df.columns.tolist())
X = df[["Temperature","Humidity","Attendance"]]
y = df["Comfort"]

model = RandomForestClassifier()

model.fit(X,y)

# ---------------------
# UI
# ---------------------

st.title("AI Powered Digital Twin")

st.header("Smart Classroom")

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

# ---------------------
# Digital Twin
# ---------------------

st.subheader("Digital Twin")

col1,col2,col3 = st.columns(3)

col1.metric(
    "Temperature",
    f"{temperature} °C"
)

col2.metric(
    "Humidity",
    f"{humidity}%"
)

col3.metric(
    "Attendance",
    attendance
)

# ---------------------
# AI Prediction
# ---------------------

prediction = model.predict(
    [[temperature,
      humidity,
      attendance]]
)

status = encoder.inverse_transform(prediction)[0]

st.subheader("AI Prediction")

st.success(status)

# ---------------------
# GenAI Style Assistant
# ---------------------

st.subheader("AI Assistant")

question = st.text_input(
    "Ask a Question"
)

if question:

    if status == "Not Comfortable":

        answer = f"""
Classroom is uncomfortable because:

Temperature = {temperature}

Humidity = {humidity}

Attendance = {attendance}

Recommendation:

Switch on cooling system.

Improve ventilation.
"""

    else:

        answer = """
Classroom conditions are normal.
"""

    st.info(answer)

# ---------------------
# Agentic AI
# ---------------------

st.subheader("Agent Actions")

actions = []

if attendance < 20:
    actions.append(
        "Attendance Alert Generated"
    )

if temperature > 30:
    actions.append(
        "Cooling Recommendation Created"
    )

if humidity > 70:
    actions.append(
        "Ventilation Recommendation Created"
    )

if actions:

    for a in actions:
        st.warning(a)

else:
    st.success(
        "No Action Required"
    )

# ---------------------
# Auto Report
# ---------------------

if st.button(
    "Generate Report"
):

    report = f"""
SMART CLASSROOM REPORT

Temperature: {temperature}

Humidity: {humidity}

Attendance: {attendance}

Status: {status}

Actions:
{actions}
"""

    st.download_button(
        "Download Report",
        report,
        file_name="report.txt"
    )
