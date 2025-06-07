import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import time


# Функція очищення даних
def clean_data(df):
    log = []
    initial_shape = df.shape

    # Видалення колонок з усіма пропущеними значеннями
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        log.append(f"Видалено колонки з усіма пропущеними значеннями: {null_columns}")
        df = df.drop(columns=null_columns)

    # Видалення рядків з пропущеними значеннями
    null_rows = df[df.isnull().any(axis=1)].shape[0]
    if null_rows > 0:
        log.append(f"Видалено {null_rows} рядків з пропущеними значеннями")
        df = df.dropna()

    # Видалення нечислових колонок
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_columns:
        log.append(f"Видалено нечислові колонки: {non_numeric_columns}")
        df = df.select_dtypes(include=['number'])

    final_shape = df.shape
    log.append(f"Розмір даних до очищення: {initial_shape}, після: {final_shape}")

    return df, log


# Візуалізація
def plot_correlation_matrix(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Кореляційна матриця")
    return fig


def plot_histograms(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    df.hist(ax=ax)
    plt.tight_layout()
    return fig


# Функції для поліноміальної регресії
def find_optimal_degree(X_train, y_train, X_test, y_test, max_degree=5):
    degrees = list(range(1, max_degree + 1))
    train_scores = []
    test_scores = []

    for degree in degrees:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])

        # Крос-валідація для тренувальних даних
        cv_scores = cross_val_score(model, X_train, y_train,
                                    scoring='neg_mean_squared_error', cv=5)
        train_rmse = np.sqrt(-cv_scores.mean())
        train_scores.append(train_rmse)

        # Оцінка на тестових даних
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_scores.append(test_rmse)

    return degrees, train_scores, test_scores


class CustomStackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)

        # Адаптивний вибір кількості фолдів
        n_samples = X.shape[0]
        n_folds = min(self.n_folds, n_samples) if n_samples > 1 else 1

        meta_features = np.zeros((X.shape[0], len(self.base_models)))

        if n_folds > 1:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = list(kf.split(X, y))
        else:
            # Використовуємо один фолд (просте навчання)
            splits = [(np.arange(len(X)), np.arange(len(X)))]

        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in splits:
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_idx], y[train_idx])
                preds = instance.predict(X[val_idx])
                # Переконуємося, що прогнози мають правильну форму
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                meta_features[val_idx, i] = preds.ravel()

        # Додаємо стандартне відхилення прогнозів
        std_dev = np.std(meta_features, axis=1).reshape(-1, 1)
        meta_features = np.hstack([meta_features, std_dev])

        self.meta_model_.fit(meta_features, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        # Отримуємо прогнози від усіх базових моделей
        predictions = []
        for models in self.base_models_:
            model_preds = [model.predict(X) for model in models]
            # Переконуємося, що всі прогнози мають однакову форму
            model_preds = [p.reshape(-1, 1) if p.ndim == 1 else p for p in model_preds]
            avg_pred = np.mean(np.hstack(model_preds), axis=1)
            predictions.append(avg_pred.reshape(-1, 1))

        # Об'єднуємо прогнози
        meta_features = np.hstack(predictions)

        # Додаємо стандартне відхилення
        std_dev = np.std(meta_features, axis=1).reshape(-1, 1)
        meta_features = np.hstack([meta_features, std_dev])

        return self.meta_model_.predict(meta_features)


class ClusteredStackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model, n_clusters=3, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_clusters = n_clusters
        self.n_folds = n_folds

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_labels_ = self.kmeans.fit_predict(X)
        self.models_by_cluster_ = {}

        for cluster_id in range(self.n_clusters):
            idx = self.cluster_labels_ == cluster_id
            X_cluster = X[idx, :]
            y_cluster = y[idx]

            # Адаптивний вибір кількості фолдів
            n_samples = X_cluster.shape[0]
            n_folds = min(self.n_folds, n_samples) if n_samples > 1 else 1

            if n_samples < 2:
                # Якщо в кластері менше 2 зразків, просто навчаємо модель
                model = CustomStackingRegressor(self.base_models, self.meta_model, n_folds=1)
                model.fit(X_cluster, y_cluster)
            else:
                model = CustomStackingRegressor(self.base_models, self.meta_model, n_folds=n_folds)
                model.fit(X_cluster, y_cluster)

            self.models_by_cluster_[cluster_id] = model

        # PCA для візуалізації кластерів
        self.pca_ = PCA(n_components=2)
        self.X_pca_ = self.pca_.fit_transform(X)
        return self

    def predict(self, X):
        X = np.asarray(X)
        cluster_labels = self.kmeans.predict(X)
        preds = np.zeros(len(X))
        for cluster_id in range(self.n_clusters):
            idx = cluster_labels == cluster_id
            if np.any(idx):
                preds[idx] = self.models_by_cluster_[cluster_id].predict(X[idx, :])
        return preds

    def get_cluster_plot(self):
        df_pca = pd.DataFrame(self.X_pca_, columns=['PC1', 'PC2'])
        df_pca['Кластер'] = self.cluster_labels_
        fig = px.scatter(df_pca, x='PC1', y='PC2', color='Кластер', title='Візуалізація кластерів (PCA)')
        fig.update_layout(plot_bgcolor='white')
        return fig


# Візуалізація порівняння ступенів полінома
def plot_degree_comparison(degrees, train_scores, test_scores, optimal_degree):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=degrees,
        y=train_scores,
        mode='lines+markers',
        name='Тренувальна помилка (RMSE)',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=degrees,
        y=test_scores,
        mode='lines+markers',
        name='Тестова помилка (RMSE)',
        line=dict(color='red')
    ))

    fig.add_vline(
        x=optimal_degree,
        line_width=2,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Оптимальний ступінь: {optimal_degree}",
        annotation_position="top right"
    )

    fig.update_layout(
        title='Помилки моделі для різних ступенів полінома',
        xaxis_title='Ступінь полінома',
        yaxis_title='RMSE',
        hovermode="x",
        plot_bgcolor='white'
    )

    return fig


# Криві навчання
def plot_learning_curves(models, X, y, cv=5, scoring='neg_mean_squared_error'):
    plt.figure(figsize=(10, 6))

    for name, model in models.items():
        try:
            # Адаптуємо кількість фолдів до розміру вибірки
            n_samples = X.shape[0]
            actual_cv = min(cv, n_samples) if n_samples > 1 else 1

            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=actual_cv, scoring=scoring,
                train_sizes=np.linspace(0.1, 1.0, 10),
                error_score='raise'
            )

            train_scores_mean = np.sqrt(-train_scores.mean(axis=1))
            test_scores_mean = np.sqrt(-test_scores.mean(axis=1))

            plt.plot(train_sizes, train_scores_mean, 'o-', label=f"{name} (тренування)")
            plt.plot(train_sizes, test_scores_mean, 'o--', label=f"{name} (валідація)")
        except Exception as e:
            st.warning(f"Не вдалося побудувати криву навчання для {name}: {str(e)}")
            continue

    plt.xlabel("Розмір тренувального набору")
    plt.ylabel("RMSE")
    plt.title("Криві навчання моделей")
    plt.legend()
    plt.grid(True)
    return plt



# Аналіз важливості ознак
def get_feature_importance(model, feature_names):
    """Аналіз важливості ознак для різних типів моделей"""
    if hasattr(model, 'coef_'):
        # Для лінійних моделей
        importance = np.abs(model.coef_)
    elif hasattr(model, 'feature_importances_'):
        # Для деревових моделей
        importance = model.feature_importances_
    elif hasattr(model, 'steps') and isinstance(model.steps[-1][1], LinearRegression):
        # Для поліноміальних моделей
        importance = np.abs(model.steps[-1][1].coef_)[:len(feature_names)]
    else:
        return None

    return pd.DataFrame({
        'Ознака': feature_names,
        'Важливість': importance / importance.sum()  # Нормалізуємо до 0-1
    }).sort_values('Важливість', ascending=False)


# Генерація бізнес-інсайтів
def generate_business_insights(df, target, top_features):
    """Генерація прикладних інсайтів на основі даних"""
    insights = []

    # Аналіз цільової змінної
    target_stats = df[target].describe()
    insights.append(f"🔍 Цільова змінна '{target}' має середнє значення {target_stats['mean']:.2f} "
                    f"з коливаннями від {target_stats['min']:.2f} до {target_stats['max']:.2f}")

    # Аналіз топ-ознак
    for i, (feat, imp) in enumerate(top_features.items(), 1):
        corr = df[[target, feat]].corr().iloc[0, 1]
        direction = "позитивно" if corr > 0 else "негативно"

        feat_stats = df[feat].describe()
        insight = (
            f"{i}. Ознака '{feat}' (важливість: {imp:.0%}) впливає {direction} на результат (кореляція: {corr:.2f}). "
            f"Її значення коливаються від {feat_stats['min']:.2f} до {feat_stats['max']:.2f}")

        if abs(corr) > 0.7:
            insight += " - дуже сильний зв'язок!"
        elif abs(corr) > 0.5:
            insight += " - помітний вплив."
        elif abs(corr) > 0.3:
            insight += " - слабкий, але істотний вплив."

        insights.append(insight)

    # Додаткові рекомендації
    if len(top_features) > 0:
        main_feature = list(top_features.keys())[0]
        insights.append("\n💡 Рекомендації:")
        insights.append(f"- Зверніть увагу на '{main_feature}' - це найважливіший фактор")
        insights.append("- Для покращення результатів спробуйте збільшити значення ознак з позитивною кореляцією")
        insights.append("- Ознаки з негативною кореляцією можуть бути 'вузькими місцями' вашого бізнесу")

    return "\n\n".join(insights)


# Візуалізація залишків
def plot_residuals(y_true, y_pred):
    """Візуалізація залишків моделі"""
    residuals = y_true - y_pred
    fig = px.scatter(x=y_pred, y=residuals,
                     labels={'x': 'Прогнозовані значення', 'y': 'Залишки'},
                     title='Аналіз залишків моделі')
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(plot_bgcolor='white')
    return fig


# Розрахунок загального показника ефективності моделі
def calculate_model_score(model, y_true, y_pred, X_train, y_train):
    """
    Розраховує узагальнений показник ефективності моделі (MPS)
    MPS = 0.4*R² + 0.3*(1 - норм_RMSE) + 0.2*(1 - норм_MAE) + 0.1*(1 - час_навчання)
    """
    # Базові метрики
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Нормалізація метрик (приводимо до шкали 0-1)
    norm_rmse = rmse / (y_true.max() - y_true.min())
    norm_mae = mae / (y_true.max() - y_true.min())

    # Складність моделі (проста евристика)
    complexity = 0
    if hasattr(model, 'n_estimators'):  # Для ансамблевих моделей
        complexity += model.n_estimators * 0.001
    if hasattr(model, 'max_depth') and model.max_depth is not None:  # Для деревових
        complexity += model.max_depth * 0.01
    if hasattr(model, 'steps'):  # Для пайплайнів
        complexity += len(model.steps) * 0.1

    # Час навчання (імітація для демонстрації)
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = min(0.5, complexity * 0.1 + 0.01)  # Обмежуємо максимальний час

    # Розрахунок узагальненого показника
    mps = (0.4 * r2 +
           0.3 * (1 - norm_rmse) +
           0.2 * (1 - norm_mae) +
           0.1 * (1 - min(1, train_time * 2)))

    return {
        'MPS': max(0, min(1, mps)),  # Гарантуємо межі 0-1
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Норм. RMSE': norm_rmse,
        'Норм. MAE': norm_mae,
        'Складність': complexity,
        'Час навчання': train_time
    }


# Головний інтерфейс
def main():
    st.set_page_config(page_title="Аналіз і прогноз продажів", layout="wide")
    st.title("🔍 Аналіз та порівняння моделей машинного навчання")

    # Завантаження даних
    st.header("1. Завантаження даних")
    uploaded_file = st.file_uploader("Завантажте CSV або Excel-файл з даними", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📊 Первинні дані")
        st.write(df.head())

        # Очищення даних
        st.header("2. Очищення даних")
        df_cleaned, log = clean_data(df)
        st.write("### 🔧 Очищені дані")
        st.write(df_cleaned.head())

        st.write("### 📝 Лог очищення")
        for entry in log:
            st.write("-", entry)

        # Візуалізація залежностей
        st.header("3. Візуалізація залежностей")
        st.subheader("📉 Гістограми ознак")
        fig_hist = plot_histograms(df_cleaned)
        st.pyplot(fig_hist)

        st.subheader("📈 Кореляційна матриця")
        fig_corr = plot_correlation_matrix(df_cleaned)
        st.pyplot(fig_corr)

        # Вибір змінних
        st.header("4. Вибір змінних")
        columns = df_cleaned.columns.tolist()
        target = st.selectbox("Оберіть залежну (прогнозовану) змінну:", columns)
        features = st.multiselect("Оберіть незалежні змінні:", [col for col in columns if col != target])

        if target and features:
            st.success(f"Цільова змінна: {target}, незалежні змінні: {features}")

            # Налаштування розділення даних
            st.header("5. Налаштування навчання")
            test_size = st.slider("Відсоток тестових даних:", 10, 50, 20) / 100

            X = df_cleaned[features]
            y = df_cleaned[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Обрати модель
            st.header("6. Вибір моделі")
            model_name = st.selectbox("Оберіть модель для прогнозу:", [
                "Лінійна регресія",
                "Поліноміальна регресія",
                "Дерево рішень",
                "Випадковий ліс",
                "Комбінована модель (Stacking)"
            ])

            model = None
            predictions = None
            model_type = None  # Додано для ідентифікації типу моделі

            if model_name == "Лінійна регресія":
                model = LinearRegression()
                model_type = "linear"
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

            elif model_name == "Поліноміальна регресія":
                model_type = "polynomial"
                st.subheader("Автоматичний підбір ступеня полінома")
                max_degree = st.slider("Максимальний ступінь для перевірки:", 1, 10, 5)

                degrees, train_scores, test_scores = find_optimal_degree(
                    X_train, y_train, X_test, y_test, max_degree
                )

                # Знаходимо оптимальний ступінь
                optimal_degree = degrees[np.argmin(test_scores)]
                st.success(f"**Оптимальний ступінь полінома:** {optimal_degree}")

                # Візуалізація
                fig_degrees = plot_degree_comparison(degrees, train_scores, test_scores, optimal_degree)
                st.plotly_chart(fig_degrees, use_container_width=True)

                # Тренуємо модель з оптимальним ступенем
                model = Pipeline([
                    ('poly', PolynomialFeatures(degree=optimal_degree)),
                    ('linear', LinearRegression())
                ])
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Додаємо пояснення
                st.markdown("""
                **Як інтерпретувати графік:**
                - **Синій графік:** Помилка на тренувальних даних (RMSE) - зменшується зі збільшенням ступеня
                - **Червоний графік:** Помилка на тестових даних (RMSE) - спочатку зменшується, потім може зростати (перетренування)
                - **Оптимальний ступінь:** Точка, де тестова помилка мінімальна
                """)

                # Порівняння метрик для різних ступенів
                st.subheader("Порівняння метрик для різних ступенів")
                degrees_df = pd.DataFrame({
                    'Ступінь': degrees,
                    'Тренувальна RMSE': train_scores,
                    'Тестова RMSE': test_scores,
                    'Різниця': np.array(train_scores) - np.array(test_scores)
                })
                st.dataframe(degrees_df.style.highlight_min(axis=0, subset=['Тестова RMSE'], color='lightgreen'))

            elif model_name == "Дерево рішень":
                model = DecisionTreeRegressor(random_state=42)
                model_type = "tree"
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

            elif model_name == "Випадковий ліс":
                model = RandomForestRegressor(random_state=42)
                model_type = "forest"
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

            elif model_name == "Комбінована модель (Stacking)":
                model_type = "stacking"
                model = ClusteredStackingRegressor(
                    base_models=[LinearRegression(), DecisionTreeRegressor(),
                                 Pipeline([('poly', PolynomialFeatures(2)), ('linear', LinearRegression())])],
                    meta_model=LinearRegression(), n_clusters=3
                )
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Додаємо візуалізацію кластерів
                st.subheader("Візуалізація кластерів")
                cluster_fig = model.get_cluster_plot()
                st.plotly_chart(cluster_fig, use_container_width=True)

                st.markdown("""
                **Пояснення до кластерів:**
                - Кластеризація допомагає моделі адаптуватися до різних підгруп даних
                - Кожен кластер має свою комбіновану модель (stacking)
                - Візуалізація показує розподіл даних у просторі PCA (2 головні компоненти)
                """)

                # Аналіз важливості ознак для кожної базової моделі в кластерах
                st.subheader("Важливість ознак (агрегована по базовим моделям)")

                base_models_info = {
                    "Лінійна регресія": LinearRegression(),
                    "Дерево рішень": DecisionTreeRegressor(),
                    "Поліноміальна": Pipeline([('poly', PolynomialFeatures(2)), ('linear', LinearRegression())])
                }

                for cluster_id, cluster_model in model.models_by_cluster_.items():
                    st.write(f"### Кластер {cluster_id}")

                    for model_name, base_model in base_models_info.items():
                        try:
                            # Отримуємо першу базову модель з кластера
                            base_model_instance = cluster_model.base_models_[0][0]
                            if hasattr(base_model_instance, 'feature_importances_') or hasattr(base_model_instance,
                                                                                               'coef_'):
                                importance_df = get_feature_importance(base_model_instance, features)
                                st.write(f"**{model_name}**:")
                                st.dataframe(importance_df)
                        except Exception as e:
                            st.write(f"Помилка для {model_name}: {str(e)}")

            # Оцінка моделі
            if model is not None and predictions is not None:
                st.header(f"7. Результати моделі: {model_type}")

                # Розрахунок показника ефективності
                score = calculate_model_score(model, y_test, predictions, X_train, y_train)

                # Візуалізація у вигляді градусника
                st.subheader("📌 Загальний показник ефективності моделі (MPS)")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score['MPS'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Оцінка моделі: {model_type}"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'steps': [
                            {'range': [0, 0.4], 'color': "red"},
                            {'range': [0.4, 0.7], 'color': "yellow"},
                            {'range': [0.7, 1], 'color': "green"}],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': score['MPS']}
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

                # Деталізація показника
                with st.expander("🔍 Деталі розрахунку MPS"):
                    st.write("""
                    **Формула:**  
                    MPS = 0.4×R² + 0.3×(1 - норм.RMSE) + 0.2×(1 - норм.MAE) + 0.1×(1 - час навчання)

                    **Ваги:**
                    - Якість прогнозу (R²): 40%
                    - Точність (RMSE): 30%
                    - Стабільність (MAE): 20%
                    - Ефективність (час навчання): 10%
                    """)

                    metrics_df = pd.DataFrame({
                        'Метрика': ['MPS', 'R²', 'RMSE', 'MAE', 'Норм. RMSE', 'Норм. MAE', 'Складність'],
                        'Значення': [score['MPS'], score['R2'], score['RMSE'], score['MAE'],
                                     score['Норм. RMSE'], score['Норм. MAE'], score['Складність']],
                        'Ідеальне значення': [1, 1, 0, 0, 0, 0, '-']
                    })

                    # Виправлене форматування для DataFrame
                    def format_value(x):
                        if isinstance(x, (int, float)):
                            return f"{x:.3f}"
                        return str(x)

                    formatted_df = metrics_df.copy()
                    for col in ['Значення', 'Ідеальне значення']:
                        formatted_df[col] = formatted_df[col].apply(format_value)

                    st.dataframe(formatted_df.style.highlight_max(
                        subset=['Значення'],
                        color='lightgreen',
                        axis=0
                    ))

                # Інтерпретація результату
                st.markdown(f"""
                **Інтерпретація оцінки {score['MPS']:.2f}/1.00:**
                - {"🔴 Погана якість" if score['MPS'] < 0.4 else
                "🟡 Середня якість" if score['MPS'] < 0.7 else
                "🟢 Висока якість"}  
                - {"Модель потребує серйозного вдосконалення" if score['MPS'] < 0.4 else
                "Модель прийнятна, але може бути покращена" if score['MPS'] < 0.7 else
                "Модель демонструє відмінну якість прогнозу"}
                """)

                # Аналіз важливості ознак (для моделей, які підтримують)
                if model_type in ["linear", "tree", "forest", "polynomial"]:
                    st.subheader("🔎 Аналіз впливу ознак")
                    try:
                        importance_df = get_feature_importance(model, features)

                        if importance_df is not None:
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                fig = px.bar(
                                    importance_df,
                                    x='Важливість',
                                    y='Ознака',
                                    orientation='h',
                                    title='Важливість кожної ознаки',
                                    color='Важливість',
                                    color_continuous_scale='Blues'
                                )
                                fig.update_layout(showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                top_features = dict(zip(importance_df['Ознака'], importance_df['Важливість']))
                                insights = generate_business_insights(df_cleaned, target, top_features)
                                with st.expander("💡 Практичні інсайти та рекомендації"):
                                    st.info(insights)
                        else:
                            st.warning("Ця модель не підтримує аналіз важливості ознак")
                    except Exception as e:
                        st.error(f"Помилка при аналізі важливості ознак: {str(e)}")

                # Показники якості моделі
                st.subheader("📊 Показники якості")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, predictions)), 3),
                              help="Середньоквадратична помилка - чим менше, тим краще")
                with col2:
                    st.metric("MAE", round(mean_absolute_error(y_test, predictions), 3),
                              help="Середня абсолютна помилка - чим менше, тим краще")
                with col3:
                    st.metric("R²", round(r2_score(y_test, predictions), 3),
                              help="Коефіцієнт детермінації (0-1) - чим ближче до 1, тим краще")

                # Аналіз залишків
                st.subheader("📉 Аналіз залишків")
                residuals_fig = plot_residuals(y_test, predictions)
                st.plotly_chart(residuals_fig, use_container_width=True)

                st.markdown("""
                **Як інтерпретувати графік залишків:**
                - **Ідеальна модель:** Точки розподілені випадково навколо нульової лінії
                - **Проблеми моделі:** 
                    - Залишки формують певний шаблон (нелінійність)
                    - Розкид зростає/зменшується (гетероскедастичність)
                    - Викиди (точки далеко від нульової лінії)
                """)

                # Візуалізація результатів
                st.subheader("📈 Графік прогнозу")
                chart_type = st.selectbox("Оберіть тип графіка:", ["Лінійний", "Точковий", "Стовпчиковий", "Змішаний"])

                compare_df = pd.DataFrame({
                    "Реальні значення": y_test.values,
                    "Прогнозовані значення": predictions
                })

                if chart_type == "Лінійний":
                    st.line_chart(compare_df, use_container_width=True)
                elif chart_type == "Точковий":
                    fig = px.scatter(compare_df, x='Реальні значення', y='Прогнозовані значення',
                                     title='Реальні vs Прогнозовані значення')
                    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                             y=[y_test.min(), y_test.max()],
                                             mode='lines', name='Ідеальний прогноз'))
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == "Стовпчиковий":
                    melted = compare_df.reset_index().melt(id_vars='index', var_name='Тип', value_name='Значення')
                    fig = px.bar(melted, x='index', y='Значення', color='Тип',
                                 barmode='group', title='Порівняння реальних та прогнозованих значень')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=compare_df.index,
                        y=compare_df['Реальні значення'],
                        mode='lines+markers',
                        name='Реальні значення',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=compare_df.index,
                        y=compare_df['Прогнозовані значення'],
                        mode='lines',
                        name='Прогнозовані значення',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(title='Змішаний графік прогнозу',
                                      xaxis_title='Індекс',
                                      yaxis_title='Значення')
                    st.plotly_chart(fig, use_container_width=True)

                # Поглиблене порівняння моделей
                st.header("8. Поглиблене порівняння моделей")
                st.subheader("Вибір моделей для порівняння")
                available_models = {
                    "Лінійна регресія": LinearRegression(),
                    "Поліноміальна (ступінь 2)": Pipeline([
                        ('poly', PolynomialFeatures(degree=2)),
                        ('linear', LinearRegression())
                    ]),
                    "Дерево рішень": DecisionTreeRegressor(random_state=42),
                    "Випадковий ліс": RandomForestRegressor(random_state=42),
                    "Комбінована (кластерна)": ClusteredStackingRegressor(
                        base_models=[
                            LinearRegression(),
                            DecisionTreeRegressor(random_state=42),
                            Pipeline([
                                ('poly', PolynomialFeatures(degree=2)),
                                ('linear', LinearRegression())
                            ])
                        ],
                        meta_model=LinearRegression(),
                        n_clusters=3
                    )
                }

                selected_models = st.multiselect(
                    "Оберіть моделі для порівняльного аналізу:",
                    list(available_models.keys()),
                    default=["Лінійна регресія", "Випадковий ліс"]
                )

                if selected_models:
                    st.subheader("Криві навчання моделей")
                    st.markdown("""
                    **Як інтерпретувати графік:**
                    - **Тренувальна крива (суцільна лінія):** Показує, як модель навчається на тренувальних даних
                    - **Валідаційна крива (пунктирна лінія):** Показує, як модель працює на нових даних
                    - **Ідеальна ситуація:** Обидві криві збігаються на низькому рівні помилки
                    - **Перетренування:** Великий розрив між кривими (валідаційна значно вище)
                    """)

                    models_to_compare = {name: available_models[name] for name in selected_models}

                    try:
                        fig = plot_learning_curves(models_to_compare, X_train, y_train)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Помилка при побудові кривих навчання: {str(e)}")

                        # Додаткові метрики порівняння
                        st.subheader("Детальні показники продуктивності")

                        results = []
                        for name, model in models_to_compare.items():
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            score = calculate_model_score(model, y_test, y_pred, X_train, y_train)

                            results.append({
                                "Модель": name,
                                "MPS": score['MPS'],
                                "R²": score['R2'],
                                "RMSE": score['RMSE'],
                                "MAE": score['MAE'],
                                "Час навчання (сек)": score['Час навчання']
                            })

                        results_df = pd.DataFrame(results).set_index("Модель")

                        # Функція для форматування значень
                        def format_value(x):
                            if isinstance(x, (int, float)):
                                return f"{x:.3f}"
                            return str(x)

                        # Застосовуємо форматування до DataFrame
                        formatted_results = results_df.copy()
                        for col in results_df.columns:
                            formatted_results[col] = results_df[col].apply(format_value)

                        st.dataframe(formatted_results.style
                                     .highlight_max(subset=['MPS', 'R²'], color='lightgreen')
                                     .highlight_min(subset=['RMSE', 'MAE', 'Час навчання (сек)'], color='lightgreen'))

                        # Візуалізація прогнозів
                        st.subheader("Графік прогнозів моделей")

                        compare_df = pd.DataFrame({
                            "Реальні значення": y_test.values
                        })

                        for model_type in selected_models:
                            model = available_models[model_type]
                            model.fit(X_train, y_train)
                            compare_df[model_type] = model.predict(X_test)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=compare_df.index,
                            y=compare_df['Реальні значення'],
                            mode='markers',
                            name='Реальні значення',
                            marker=dict(color='black', size=8)
                        ))

                        colors = px.colors.qualitative.Plotly
                        for i, name in enumerate(selected_models):
                            fig.add_trace(go.Scatter(
                                x=compare_df.index,
                                y=compare_df[name],
                                mode='lines',
                                name=name,
                                line=dict(color=colors[i % len(colors)], width=2)
                            ))

                        fig.update_layout(
                            title='Порівняння прогнозів моделей',
                            xaxis_title='Індекс спостереження',
                            yaxis_title='Значення',
                            plot_bgcolor='white',
                            hovermode='x'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Помилка при побудові кривих навчання: {str(e)}")
                    # Додаємо цей блок після візуалізації результатів обраної моделі
                    # (перед if __name__ == "__main__":)

                    st.header("📊 Фінальний порівняльний аналіз моделей")

                    # Визначаємо моделі для порівняння
                    comparison_models = {
                        "Лінійна регресія": LinearRegression(),
                        "Поліноміальна (2 ступінь)": Pipeline([
                            ('poly', PolynomialFeatures(degree=2)),
                            ('linear', LinearRegression())
                        ]),
                        "Дерево рішень": DecisionTreeRegressor(random_state=42),
                        "Випадковий ліс": RandomForestRegressor(random_state=42),
                        "Комбінована модель": ClusteredStackingRegressor(
                            base_models=[
                                LinearRegression(),
                                DecisionTreeRegressor(random_state=42),
                                Pipeline([('poly', PolynomialFeatures(2)), ('linear', LinearRegression())])
                            ],
                            meta_model=LinearRegression(),
                            n_clusters=3
                        )
                    }

                    # Створюємо DataFrame для результатів
                    results = []

                    for model_type, model in comparison_models.items():
                        try:
                            start_time = time.time()
                            model.fit(X_train, y_train)
                            train_time = time.time() - start_time
                            y_pred = model.predict(X_test)

                            # Розраховуємо метрики
                            mse = mean_squared_error(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)

                            # Додаємо результати
                            results.append({
                                "Модель": model_type,
                                "MSE": mse,
                                "RMSE": np.sqrt(mse),
                                "MAE": mae,
                                "R²": r2,
                                "Час навчання (с)": train_time
                            })
                        except Exception as e:
                            st.error(f"Помилка при оцінці моделі {model_type}: {str(e)}")

                    if results:
                        results_df = pd.DataFrame(results).set_index('Модель')

                        # Візуалізація результатів
                        st.subheader("Таблиця порівняння моделей")
                        st.dataframe(
                            results_df.style
                            .highlight_min(subset=['MSE', 'RMSE', 'MAE'], color='lightgreen')
                            .highlight_max(subset=['R²'], color='lightgreen')
                            .format({
                                'MSE': '{:.3f}',
                                'RMSE': '{:.3f}',
                                'MAE': '{:.3f}',
                                'R²': '{:.3f}',
                                'Час навчання (с)': '{:.4f}'
                            })
                        )

                        # Визначаємо найкращу модель
                        best_model = results_df['R²'].idxmax()
                        st.success(
                            f"**Найкраща модель за R²:** {best_model} (R² = {results_df.loc[best_model, 'R²']:.3f})")

                        # Візуалізація у вигляді графіків
                        st.subheader("Візуальне порівняння моделей")

                        fig = go.Figure()
                        metrics_to_plot = ['RMSE', 'MAE', 'R²']
                        colors = px.colors.qualitative.Plotly

                        for i, metric in enumerate(metrics_to_plot):
                            fig.add_trace(go.Bar(
                                x=results_df.index,
                                y=results_df[metric],
                                name=metric,
                                marker_color=colors[i],
                                text=results_df[metric].round(3),
                                textposition='auto'
                            ))

                        fig.update_layout(
                            barmode='group',
                            title='Порівняння основних метрик моделей',
                            xaxis_title='Моделі',
                            yaxis_title='Значення метрики',
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Рекомендації
                        st.subheader("Рекомендації щодо вибору моделі")

                        if results_df.loc[best_model, 'Час навчання (с)'] > 1.0:
                            st.warning(
                                f"Увага: {best_model} має відносно високий час навчання ({results_df.loc[best_model, 'Час навчання (с)']:.2f} сек)")

                        st.markdown("""
                        **Критерії вибору:**
                        - Для максимальної точності: оберіть модель з найвищим R²
                        - Для швидкості: оберіть модель з найменшим часом навчання
                        - Для балансу: оберіть модель з гарним співвідношенням точності та швидкості
                        """)

                        # Показуємо найкращі моделі за різними критеріями
                        st.markdown("""
                        **Найкращі моделі за критеріями:**
                        - Точність (R²): {}
                        - Швидкість: {}
                        - Баланс (точність + швидкість): {}
                        """.format(
                            results_df['R²'].idxmax(),
                            results_df['Час навчання (с)'].idxmin(),
                            results_df['R²'].sub(results_df['Час навчання (с)']).idxmax()
                        ))
                    else:
                        st.error("Не вдалося отримати результати для жодної моделі")

if __name__ == "__main__":
    main()