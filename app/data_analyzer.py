import time

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


def sample_nodes(self, n=20):
    """Вибір випадкових точок з графа для аналізу"""
    if self.nodes is None:
        print("Спочатку завантажте граф міста за допомогою get_city_graph()")
        return None

    # Вибираємо випадкові вузли з графа
    sample_nodes = self.nodes.sample(min(n, len(self.nodes)))
    return sample_nodes


def calculate_traffic_intensity(self):
    """Розрахунок інтенсивності руху та збереження/завантаження з БД"""
    if self.graph is None or self.pollution_data.empty:
        print("Спочатку завантажте граф міста та зберіть дані про забруднення")
        return

    table_name = f"traffic_intensity_{self.city_name.lower().replace(' ', '_')}"

    try:
        # Спроба завантажити з БД
        query = f"SELECT node_id, traffic_intensity FROM {table_name}"
        df_db = pd.read_sql(query, con=self.engine)

        print(f"Інтенсивність трафіку завантажено з бази даних для {len(df_db)} вузлів.")

        # Зливаємо з pollution_data
        self.pollution_data = self.pollution_data.merge(
            df_db, on='node_id', how='left'
        )

        return self.pollution_data['traffic_intensity']

    except Exception as e:
        print(f"Не вдалося завантажити з БД: {e}")
        print("Обчислення інтенсивності трафіку на основі OSM...")

        # Простий евристичний підхід: road_count + ваги типу дороги
        node_scores = {}
        for node in self.graph.nodes:
            degree = len(list(self.graph.out_edges(node))) + len(list(self.graph.in_edges(node)))
            types = [
                self.graph.edges[u, v, k].get('highway', 'unclassified')
                for u, v, k in self.graph.out_edges(node, keys=True)
            ]
            # Оцінюємо тип дороги (вага)
            type_score = sum(_road_type_weight(t) for t in types) / max(1, len(types))
            node_scores[node] = degree * type_score

        # Нормалізація
        max_score = max(node_scores.values())
        for node_id in node_scores:
            node_scores[node_id] /= max_score

        # Запис у pollution_data
        self.pollution_data['traffic_intensity'] = self.pollution_data['node_id'].map(node_scores)

        # Формуємо DataFrame для БД
        traffic_df = self.pollution_data[['node_id', 'traffic_intensity']].copy()
        traffic_df.to_sql(table_name, con=self.engine, if_exists='replace', index=False)
        print(f"Інтенсивність трафіку збережено в БД у таблицю {table_name}.")

        return self.pollution_data['traffic_intensity']

def linear_regression_analysis(self):
    """Лінійна регресія між трафіком і AQI"""
    if 'traffic_intensity' not in self.pollution_data.columns:
        self.calculate_traffic_intensity()

    aqi_col = None
    if 'ow_aqi' in self.pollution_data.columns:
        aqi_col = 'ow_aqi'
    elif 'waqi_aqi' in self.pollution_data.columns:
        aqi_col = 'waqi_aqi'
    else:
        return None  # Немає даних для аналізу

    df = self.pollution_data[[aqi_col, 'traffic_intensity']].dropna()
    if df.empty or len(df) < 3:
        return None

    X = df[['traffic_intensity']].values
    y = df[aqi_col].values

    model = LinearRegression()
    model.fit(X, y)

    return {
        "equation": f"{aqi_col} = {model.intercept_:.2f} + {model.coef_[0]:.2f} * traffic_intensity",
        "r_squared": round(model.score(X, y), 3),
        "target": aqi_col
    }


def analyze_correlations(self):
    """
    Обчислює кореляцію між трафіком та всіма показниками забруднення.
    """
    if 'traffic_intensity' not in self.pollution_data.columns:
        self.calculate_traffic_intensity()

    pollution_cols = [col for col in self.pollution_data.columns if col.startswith("ow_") or col.startswith("waqi_")]
    results = []

    for col in pollution_cols:
        df = self.pollution_data[['traffic_intensity', col]].dropna()
        if len(df) >= 3:
            corr, p_value = pearsonr(df['traffic_intensity'], df[col])
            results.append({
                "parameter": col,
                "correlation": round(corr, 3),
                "p_value": round(p_value, 4),
                "significant": bool(p_value < 0.05)
            })

    return results


def analyze_regressions(self):
    """
    Лінійна регресія для всіх показників забруднення відносно traffic_intensity.
    """
    if 'traffic_intensity' not in self.pollution_data.columns:
        self.calculate_traffic_intensity()

    pollution_cols = [col for col in self.pollution_data.columns if col.startswith("ow_") or col.startswith("waqi_")]
    results = []

    for col in pollution_cols:
        df = self.pollution_data[['traffic_intensity', col]].dropna()
        if len(df) >= 3:
            X = df[['traffic_intensity']]
            y = df[col]

            model = LinearRegression()
            model.fit(X, y)

            result = {
                "parameter": col,
                "intercept": round(model.intercept_, 3),
                "coefficient": round(model.coef_[0], 3),
                "r2_score": round(model.score(X, y), 3),
                "equation": f"{col} = {model.intercept_:.2f} + {model.coef_[0]:.2f} * traffic_intensity"
            }
            results.append(result)
    print(results)
    return results


def aqi_distribution(self):
    """Підрахунок кількості точок з кожним рівнем AQI (1-5)"""
    counts = None
    aqi_col = None
    if 'ow_aqi' in self.pollution_data.columns:
        aqi_col = 'ow_aqi'
    elif 'waqi_aqi' in self.pollution_data.columns:
        aqi_col = 'waqi_aqi'

    if aqi_col:
        counts = self.pollution_data[aqi_col].value_counts().sort_index()
        for category, count in counts.items():
            print(f"Категорія {category}: {count} точок")
    return {int(k): int(v) for k, v in counts.items()}


def param_distribution(self, column: str, bins: int = 10):
    """
    Розподіл значень певного параметра (ow_co, ow_no2, ow_pm2_5, ow_pm10) по біннах.
    :param column: назва колонки у pollution_data
    :param bins: кількість бінів для гістограми
    :return: словник {"<interval>": count}
    """
    print(self.pollution_data.columns)
    if column not in self.pollution_data.columns:
        return {}

    values = self.pollution_data[column].dropna()
    hist, bin_edges = np.histogram(values, bins=bins)

    result = {}
    for i in range(len(hist)):
        interval = f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}"
        result[interval] = int(hist[i])
    return result


def analyze_clusters(self, n_clusters=3):
    """Кластеризація точок забруднення + статистика по кластерах"""
    pollution_cols = [col for col in self.pollution_data.columns if 'aqi' in col or 'co' in col]
    if not pollution_cols:
        return {
            "points": [],
            "cluster_summary": []
        }

    features = self.pollution_data[['latitude', 'longitude'] + pollution_cols].dropna()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features[pollution_cols])
    features['cluster'] = kmeans.labels_

    # Зведена статистика по кожному кластеру
    cluster_summary = (
        features
        .groupby("cluster")[pollution_cols]
        .mean()
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    # Список точок з координатами та кластером
    points = features[['latitude', 'longitude', 'cluster']].to_dict(orient='records')

    return {
        "points": points,
        "cluster_summary": cluster_summary
    }


def generate_report(self):
    """Генерація звіту про вплив транспорту на екологію"""
    if self.pollution_data.empty:
        print("Спочатку зберіть дані про забруднення")
        return

    print("\n====== ЗВІТ ПРО ВПЛИВ ТРАНСПОРТУ НА ЕКОЛОГІЮ ======")
    print(f"Місто: {self.city_name}")
    print(f"Кількість проаналізованих точок: {len(self.pollution_data)}")

    # Базова статистика щодо забруднення
    print("\n=== Статистика забруднення ===")
    pollutants = [col for col in self.pollution_data.columns
                  if any(col.startswith(p) for p in ['ow_', 'waqi_'])]

    for pollutant in pollutants:
        if self.pollution_data[pollutant].notnull().sum() > 0:
            print(f"{pollutant}:")
            print(f"  Середнє: {self.pollution_data[pollutant].mean():.2f}")
            print(f"  Мінімум: {self.pollution_data[pollutant].min():.2f}")
            print(f"  Максимум: {self.pollution_data[pollutant].max():.2f}")

    # Аналіз розподілу AQI по категоріях (1-5)
    print("\n=== Розподіл точок за категоріями AQI (1 — добре, 5 — дуже погано) ===")
    aqi_column = None
    if 'ow_aqi' in self.pollution_data.columns:
        aqi_column = 'ow_aqi'
    elif 'waqi_aqi' in self.pollution_data.columns:
        aqi_column = 'waqi_aqi'

    if aqi_column:
        counts = self.pollution_data[aqi_column].value_counts().sort_index()
        for category, count in counts.items():
            print(f"Категорія {category}: {count} точок")

    # Аналіз кореляції
    analyze_correlations(self)

    # Виявлення "гарячих точок" забруднення
    print("\n=== Гарячі точки забруднення ===")
    if aqi_column:
        sorted_by_pollution = self.pollution_data.sort_values(aqi_column, ascending=False).head(3)
        for _, row in sorted_by_pollution.iterrows():
            print(f"Координати: {row['latitude']:.4f}, {row['longitude']:.4f}, AQI: {row[aqi_column]}")

    # Рекомендації
    print("\n=== Рекомендації ===")
    print("1. Моніторинг забруднення у виявлених гарячих точках.")
    print("2. Регулювання транспортних потоків у районах з високим рівнем забруднення.")
    print("3. Розгляд можливості створення зон з низьким рівнем викидів.")

    correlation_results = analyze_correlations(self)
    regression_result = linear_regression_analysis(self)

    return {
        "correlations": correlation_results,
        "regression": regression_result
    }

def _road_type_weight(road_type):
    """Вага типу дороги — чим більша, тим вища очікувана інтенсивність"""
    weights = {
        'motorway': 1.0,
        'trunk': 0.9,
        'primary': 0.8,
        'secondary': 0.6,
        'tertiary': 0.4,
        'residential': 0.2,
        'service': 0.1,
        'unclassified': 0.05
    }
    if isinstance(road_type, list):
        road_type = road_type[0]
    return weights.get(road_type, 0.05)