import streamlit as st
import pandas as pd
import joblib

ml_pipeline = joblib.load("ml_pipeline_cb.pkl")

st.title("Прогноз сердечно-сосудистого заболевания")

st.header("Введите данные пациента:")

# форма для ввода
age = st.number_input("Возраст", min_value=0, max_value=120, value=50)
sex = st.selectbox("Пол", options=["M", "F"])
chest_pain = st.selectbox("Тип боли в груди: TA - типичная стенокардия, ATA - атипичная стенокардия, NAP - боль не связана с сердцем, ASY - бессимптомно", options=["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Давление в покое (мм рт. ст.)", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Уровень холестерина в крови (мг/дл)", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Сахар натощак (1 если >120 mg/dl иначе 0)", options=[0, 1])
rest_ecg = st.selectbox("Результаты ЭКГ в покое: Normal — нормальные, ST — отклонения ST-T сегмента, LVH — признаки гипертрофии левого желудочка", options=["Normal", "ST", "LVH"])
max_hr = st.number_input("Максимальная достигнутая частота пульса", min_value=60, max_value=202, value=150)
exercise_angina = st.selectbox("Стенокардия при нагрузке: есть/нет", options=["Y", "N"])
oldpeak = st.number_input("Депрессия ST сегмента (числовое значение)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st_slope = st.selectbox("Наклон ST сегмента при нагрузке: Up — восходящий, Flat — плоский, Down — нисходящий", options=["Up", "Flat", "Down"])

if st.button("Предсказать"):
    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [rest_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    # предсказание
    prediction = ml_pipeline.predict(input_data)
    prediction_proba = ml_pipeline.predict_proba(input_data)

    st.subheader("Результат предсказания:")
    st.write(f"Наличие болезни сердца: {'Да' if prediction[0] == 1 else 'Нет'}")
    st.write(f"Вероятность: {prediction_proba[0][1]:.2f}")