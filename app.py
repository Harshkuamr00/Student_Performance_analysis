import streamlit as st
import joblib
import pandas as pd

model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_order = joblib.load('models/feature_order.pkl')

ordinal_maps = {
    'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
    'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
    'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
    'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
    'Parental_Education_Level': {'High School': 0, 'College': 1, 'Postgraduate': 2},
    'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2}
}
numeric_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']

st.title("Student Performance Prediction")

def user_input():
    data = {}
    data['Hours_Studied'] = st.sidebar.slider('Hours Studied', 0, 40, 20)
    data['Attendance'] = st.sidebar.slider('Attendance (%)', 0, 100, 80)
    data['Parental_Involvement'] = st.sidebar.selectbox('Parental Involvement', ['Low', 'Medium', 'High'])
    data['Access_to_Resources'] = st.sidebar.selectbox('Access to Resources', ['Low', 'Medium', 'High'])
    data['Extracurricular_Activities_Yes'] = st.sidebar.selectbox('Extracurricular Activities', ['No', 'Yes']) == 'Yes'
    data['Sleep_Hours'] = st.sidebar.slider('Sleep Hours', 0, 12, 7)
    data['Previous_Scores'] = st.sidebar.slider('Previous Scores', 0, 100, 60)
    data['Motivation_Level'] = st.sidebar.selectbox('Motivation Level', ['Low', 'Medium', 'High'])
    data['Internet_Access_Yes'] = st.sidebar.selectbox('Internet Access', ['No', 'Yes']) == 'Yes'
    data['Tutoring_Sessions'] = st.sidebar.slider('Tutoring Sessions', 0, 10, 2)
    data['Family_Income'] = st.sidebar.selectbox('Family Income', ['Low', 'Medium', 'High'])
    data['Teacher_Quality'] = st.sidebar.selectbox('Teacher Quality', ['Low', 'Medium', 'High'])
    data['School_Type_Public'] = st.sidebar.selectbox('School Type', ['Private', 'Public']) == 'Public'
    # Peer Influence (create both one-hot columns)
    peer_influence = st.sidebar.selectbox('Peer Influence', ['Positive', 'Negative', 'Neutral'])
    data['Peer_Influence_Neutral'] = 1 if peer_influence == 'Neutral' else 0
    data['Peer_Influence_Positive'] = 1 if peer_influence == 'Positive' else 0
    data['Physical_Activity'] = st.sidebar.slider('Physical Activity', 0, 10, 3)
    data['Learning_Disabilities_Yes'] = st.sidebar.selectbox('Learning Disabilities', ['No', 'Yes']) == 'Yes'
    data['Parental_Education_Level'] = st.sidebar.selectbox('Parental Education Level', ['High School', 'College', 'Postgraduate'])
    data['Distance_from_Home'] = st.sidebar.selectbox('Distance from Home', ['Near', 'Moderate', 'Far'])
    data['Gender_Male'] = st.sidebar.selectbox('Gender', ['Female', 'Male']) == 'Male'
    return data

def preprocess(data):
    # Ordinal encode
    data['Parental_Involvement'] = ordinal_maps['Parental_Involvement'][data['Parental_Involvement']]
    data['Access_to_Resources'] = ordinal_maps['Access_to_Resources'][data['Access_to_Resources']]
    data['Motivation_Level'] = ordinal_maps['Motivation_Level'][data['Motivation_Level']]
    data['Family_Income'] = ordinal_maps['Family_Income'][data['Family_Income']]
    data['Teacher_Quality'] = ordinal_maps['Teacher_Quality'][data['Teacher_Quality']]
    data['Parental_Education_Level'] = ordinal_maps['Parental_Education_Level'][data['Parental_Education_Level']]
    data['Distance_from_Home'] = ordinal_maps['Distance_from_Home'][data['Distance_from_Home']]
    # Convert booleans to int
    for col in ['Extracurricular_Activities_Yes', 'Internet_Access_Yes', 'School_Type_Public',
                'Peer_Influence_Neutral', 'Peer_Influence_Positive', 'Learning_Disabilities_Yes', 'Gender_Male']:
        data[col] = int(data[col])
    # Scale numeric features
    num_array = scaler.transform([[data[col] for col in numeric_cols]])[0]
    for idx, col in enumerate(numeric_cols):
        data[col] = num_array[idx]
    # Arrange in correct order
    X_input = pd.DataFrame([data])
    X_input = X_input.reindex(columns=feature_order, fill_value=0)
    return X_input

user_data = user_input()
X_input = preprocess(user_data)

if st.button("Predict"):
    pred = model.predict(X_input)[0]
    probs = model.predict_proba(X_input)[0]
    class_idx = list(model.classes_).index(pred)
    prob = probs[class_idx]
    st.subheader("Prediction Result")
    st.write("Prediction:", "**Pass**" if pred == 1 else "**Fail**")
    st.write(f"Confidence: {prob:.2%}")

st.write("---")
