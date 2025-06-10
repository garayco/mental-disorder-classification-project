import streamlit as st
import numpy as np
import pickle
import pandas as pd

@st.cache_resource
def load_model_and_scaler():
    with open("clf_svc_model.pkl", "rb") as f:
        model, scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# List of features
best_features = [
    "mood_swing_YES", "sadness", "sexual_activity", "euphoric", "optimisim",
    "suicidal_thoughts_YES", "exhausted", "concentration",
    "sleep_dissorder", "anorxia_YES", "nervous_break_down_YES"
]

# Categorize features
yes_no_features = {"mood_swing_YES", "suicidal_thoughts_YES", "anorxia_YES", "nervous_break_down_YES"}
range_0_3 = {"sadness", "euphoric", "exhausted", "sleep_dissorder"}

# Display title
st.title("🧠 Clinical Symptoms Evaluation")

# Collect user input
st.subheader("📝 Answer the following questions:")

user_input = {}

# Group 1: Yes/No questions
st.markdown("### Binary Symptoms (Yes/No)")
cols = st.columns(2)
for i, feature in enumerate(yes_no_features):
    with cols[i % 2]:
        label = feature.replace('_YES', '').replace('_', ' ').capitalize()
        user_input[feature] = st.selectbox(f"{label}?", ["No", "Yes"]) == "Yes"

st.markdown("### Severity (0 to 3)")
cols = st.columns(2)
for i, feature in enumerate(range_0_3):
    with cols[i % 2]:
        label = feature.replace('_', ' ').capitalize()
        user_input[feature] = st.slider(f"{label}", 0, 3, 1)

st.markdown("### Intensity (1 to 10)")
cols = st.columns(2)
for i, feature in enumerate(f for f in best_features if f not in yes_no_features and f not in range_0_3):
    with cols[i % 2]:
        label = feature.replace('_', ' ').capitalize()
        user_input[feature] = st.slider(f"{label}", 1, 10, 5)

# Scale input
def scaling_features(input_data):
    return scaler.transform(input_data)

import pandas as pd

if st.button("🔮 Predict Diagnosis"):
    mapped_input = []
    for f in best_features:
        val = int(user_input[f])
        if f in range_0_3:
            mapped_input.append(val + 1)
        else:
            mapped_input.append(val)
            
    input_df = pd.DataFrame([mapped_input], columns=best_features)
    
    scaled_df = pd.DataFrame(scaling_features(input_df), columns=best_features) 

    prediction = model.predict(scaled_df)[0]
    st.success(f"Estimated diagnosis: **{prediction}**")

