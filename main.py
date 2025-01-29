import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Заголовок додатку
st.title("Аналіз даних з CSV")

# Завантаження файлу
uploaded_file = st.file_uploader("Завантажте CSV-файл", type=["csv"])
if uploaded_file is not None:
    # Читання даних
    df = pd.read_csv(uploaded_file)

    # Відображення даних
    st.write("Перші 5 рядків даних:")
    st.write(df.head())

    # Вибір стовпця для графіку
    column = st.selectbox("Виберіть стовпець для графіку", df.columns)

    # Побудова графіку
    st.write(f"Графік для стовпця '{column}':")
    fig, ax = plt.subplots()
    ax.plot(df[column])
    st.pyplot(fig)