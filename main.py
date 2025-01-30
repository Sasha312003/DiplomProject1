import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("Аналіз даних з CSV та лінійна регресія")

uploaded_file = st.file_uploader("Завантажте CSV-файл", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_cleaned = df.dropna()
    st.write("Перші 5 рядків даних (після видалення порожніх рядків):")
    st.write(df_cleaned.head())

    st.write("### Виберіть стовпці для лінійної регресії")
    feature = st.selectbox("Виберіть незалежну змінну (X)", df_cleaned.columns)
    target = st.selectbox("Виберіть залежну змінну (Y)", df_cleaned.columns)

    if pd.api.types.is_numeric_dtype(df_cleaned[feature]) and pd.api.types.is_numeric_dtype(df_cleaned[target]):
        Q1 = df_cleaned[target].quantile(0.25)
        Q3 = df_cleaned[target].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_filtered = df_cleaned[(df_cleaned[target] >= lower_bound) & (df_cleaned[target] <= upper_bound)]
        st.write("### Дані після видалення викидів")
        st.write(f"Видалено {len(df_cleaned) - len(df_filtered)} рядків (викидів).")
        st.write(df_filtered.head())

        if st.button("Запустити лінійну регресію"):
            X = df_filtered[[feature]]
            y = df_filtered[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write("### Результати лінійної регресії")
            st.write(f"Коефіцієнт регресії (нахил): {model.coef_[0]:.2f}")
            st.write(f"Вільний член (intercept): {model.intercept_:.2f}")
            st.write(f"Середньоквадратична помилка (MSE): {mse:.2f}")
            st.write(f"Коефіцієнт детермінації (R²): {r2:.2f}")

            st.write("### Графік регресії")
            fig, ax = plt.subplots()
            ax.scatter(X_test, y_test, color='blue', label="Фактичні значення")
            ax.plot(X_test, y_pred, color='red', label="Прогнозовані значення")
            ax.set_xlabel(feature)
            ax.set_ylabel(target)
            ax.legend()
            st.pyplot(fig)
    else:
        st.error("Помилка: Вибрані стовпці повинні містити числові дані.")