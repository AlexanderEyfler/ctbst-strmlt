# app.py
import streamlit as st
import joblib
import pandas as pd
import sklearn

from catboost import CatBoostClassifier
from sklearn import set_config


# Устанавливаем глобальную настройку для sklearn
set_config(transform_output="pandas")

# Загружаем конвейер предварительной обработки
preprocessor = joblib.load('preprocessor.joblib')

# Загружаем модель CatBoost
model = CatBoostClassifier()
model.load_model('catboost_model.cbm')

# Заголовок приложения
st.title('Предсказание сердечных заболеваний')

# Функция для ввода данных пользователем (остается прежней)
def user_input_features():
    Age = st.number_input('Возраст', min_value=0, max_value=120, value=30)
    Sex = st.selectbox('Пол', ('M', 'F'))
    ChestPainType = st.selectbox('Тип боли в груди', ('TA', 'ATA', 'NAP', 'ASY'))
    RestingBP = st.number_input('Артериальное давление в покое', min_value=0, max_value=200, value=120)
    Cholesterol = st.number_input('Холестерин', min_value=0, max_value=600, value=200)
    FastingBS = st.selectbox('Уровень сахара натощак > 120 mg/dl', (0, 1))
    RestingECG = st.selectbox('Результаты ЭКГ в покое', ('Normal', 'ST', 'LVH'))
    MaxHR = st.number_input('Максимальный пульс', min_value=0, max_value=220, value=150)
    ExerciseAngina = st.selectbox('Ангина при нагрузке', ('Y', 'N'))
    Oldpeak = st.number_input('Снижение ST', min_value=0.0, max_value=10.0, value=1.0)
    ST_Slope = st.selectbox('Наклон ST сегмента', ('Up', 'Flat', 'Down'))

    data = {
        'Age': Age,
        'Sex': Sex,
        'ChestPainType': ChestPainType,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'RestingECG': RestingECG,
        'MaxHR': MaxHR,
        'ExerciseAngina': ExerciseAngina,
        'Oldpeak': Oldpeak,
        'ST_Slope': ST_Slope
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Получаем данные от пользователя
input_df = user_input_features()

# Кнопка для предсказания
if st.button('Предсказать'):
    # Применяем предварительную обработку
    X_processed = preprocessor.transform(input_df)
    
    # Прогнозируем
    prediction = model.predict(X_processed)
    prediction_proba = model.predict_proba(X_processed)

    # Отображаем результат
    st.subheader('Результат предсказания')
    heart_disease = 'Наличие сердечных заболеваний' if prediction[0] == 1 else 'Отсутствие сердечных заболеваний'
    st.write(heart_disease)
    st.subheader('Вероятность')
    st.write(f'Вероятность наличия заболевания: {prediction_proba[0][1]*100:.2f}%')
