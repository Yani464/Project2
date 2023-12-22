import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
import sklearn
sklearn.set_config(transform_output="pandas")



st.title("Предсказание цены недвижимости по модели")

uploaded_file = st.file_uploader("Загрузите файл CSV", type=["csv"])

# Проверьте, был ли файл загружен
if uploaded_file is not None:
    # Прочитайте файл CSV в датафрейм Pandas
    df_test = pd.read_csv(uploaded_file)

    # Отобразите загруженные данные в Streamlit
    # st.write("Загруженный датафрейм:")
    # st.write(df_test)
#читаем дата сет в переменную df_test

ml_pipeline = joblib.load('ml_pipeline_no_scaler.pkl')

model = joblib.load('final_model_log_no_scaler.pkl')


st.title('Применение пайплайна и обученной модели к вашим данным')
st.write('Расчет стоимости дома находится в последней колонке под названием PredictedSalePrice')


# Отобразите загруженные тестовые данные
# st.subheader('Тестовые Данные:')
# st.write(df_test)

# Примените пайплайн к тестовым данным
test_pipeline = ml_pipeline.transform(df_test)

# Отобразите результаты преобразования



test_pypiline = ml_pipeline.transform(df_test)

predictions = model.predict(test_pypiline)

predictions = np.exp(predictions)

preds = pd.DataFrame(predictions,columns=['PredictedSalePrice'])

result = pd.concat([df_test,preds],axis = 1)
st.write(result)