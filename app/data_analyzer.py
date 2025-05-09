import time
import pandas as pd
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

def collect_pollution_data(self, num_points=20):
    """Збір даних про забруднення для вибраних точок"""
    print("Збір даних про забруднення повітря...")
    if self.nodes is None:
        print("Спочатку завантажте граф міста за допомогою get_city_graph()")
        return

    sample_nodes = self.sample_nodes(num_points)
    pollution_data = []

    for idx, node in sample_nodes.iterrows():
        lat, lon = node.geometry.y, node.geometry.x
        print(f"Обробка точки {lat}, {lon}")

        # Отримуємо дані про забруднення з двох джерел
        ow_data = self.get_openweather_pollution(lat, lon)
        waqi_data = self.get_waqi_pollution(lat, lon)

        # Об'єднуємо дані
        node_data = {
            'node_id': idx,
            'latitude': lat,
            'longitude': lon,
            'road_count': len(self.graph.out_edges(idx))
        }

        if ow_data:
            node_data.update({
                'ow_aqi': ow_data['aqi'],
                'ow_co': ow_data['co'],
                'ow_no2': ow_data['no2'],
                'ow_pm2_5': ow_data['pm2_5'],
                'ow_pm10': ow_data['pm10']
            })

        if waqi_data:
            node_data.update({
                'waqi_aqi': waqi_data['aqi'],
                'waqi_pm25': waqi_data['pm25'],
                'waqi_pm10': waqi_data['pm10']
            })

        pollution_data.append(node_data)
        print(node_data)
        time.sleep(0.5)  # Пауза між запитами до API

    self.pollution_data = pd.DataFrame(pollution_data)
    print(f"Зібрано дані про забруднення для {len(self.pollution_data)} точок.")
    return self.pollution_data

def calculate_traffic_intensity(self):
    """Розрахунок інтенсивності руху на основі структури дорожньої мережі"""
    if self.graph is None or self.pollution_data.empty:
        print("Спочатку завантажте граф міста та зберіть дані про забруднення")
        return

    # Простий підхід: використовуємо кількість доріг, підключених до вузла,
    # як проксі для інтенсивності руху
    # У реальній системі тут можна інтегрувати дані про трафік

    # Нормалізуємо показник дорожнього навантаження
    max_road_count = self.pollution_data['road_count'].max()
    self.pollution_data['traffic_intensity'] = self.pollution_data['road_count'] / max_road_count

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

def analyze_correlation(self):
    """Аналіз кореляції між інтенсивністю руху та забрудненням"""
    if 'traffic_intensity' not in self.pollution_data.columns:
        self.calculate_traffic_intensity()

    print("\n=== Аналіз кореляції між рухом та забрудненням ===")

    # Перевіряємо кореляцію з різними показниками забруднення
    pollution_metrics = [col for col in self.pollution_data.columns
                         if any(col.startswith(p) for p in ['ow_', 'waqi_'])]

    correlation_results = {}

    for metric in pollution_metrics:
        if self.pollution_data[metric].notnull().sum() > 2:  # Потрібно мінімум 3 точки для кореляції
            corr, p_value = pearsonr(
                self.pollution_data['traffic_intensity'],
                self.pollution_data[metric]
            )
            correlation_results[metric] = {'correlation': corr, 'p_value': p_value}

            significance = "статистично значуща" if p_value < 0.05 else "статистично не значуща"
            print(f"{metric}: кореляція = {corr:.2f}, p-value = {p_value:.4f} ({significance})")

    return correlation_results

def analyze_clusters(self, n_clusters=3):
    """Кластеризація точок забруднення"""
    pollution_cols = [col for col in self.pollution_data.columns if 'aqi' in col]
    if not pollution_cols:
        return []

    features = self.pollution_data[['latitude', 'longitude'] + pollution_cols].dropna()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features[pollution_cols])
    features['cluster'] = kmeans.labels_

    # Повертаємо список координат з їхніми кластерами
    return features[['latitude', 'longitude', 'cluster']].to_dict(orient='records')

def analyze_regression(self):
    """Лінійна регресія для оцінки впливу трафіку на забруднення"""
    if 'traffic_intensity' not in self.pollution_data.columns:
        self.calculate_traffic_intensity()

    results = {}
    for col in self.pollution_data.columns:
        if 'aqi' in col:
            df = self.pollution_data[['traffic_intensity', col]].dropna()
            if len(df) >= 3:
                model = LinearRegression()
                model.fit(df[['traffic_intensity']], df[col])
                score = model.score(df[['traffic_intensity']], df[col])
                results[col] = {
                    'coefficient': float(model.coef_[0]),
                    'intercept': float(model.intercept_),
                    'r2_score': float(score)
                }
    return results

def aqi_distribution(self):
    """Підрахунок кількості точок з кожним рівнем AQI (1-5)"""
    distribution = {}
    aqi_col = None
    if 'ow_aqi' in self.pollution_data.columns:
        aqi_col = 'ow_aqi'
    elif 'waqi_aqi' in self.pollution_data.columns:
        aqi_col = 'waqi_aqi'

    if aqi_col:
        bins = [0, 50, 100, 150, 200, 500]
        labels = [1, 2, 3, 4, 5]
        self.pollution_data['aqi_level'] = pd.cut(self.pollution_data[aqi_col], bins=bins, labels=labels)
        distribution = self.pollution_data['aqi_level'].value_counts().sort_index().to_dict()
    return {int(k): int(v) for k, v in distribution.items()}

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
    analyze_correlation(self)

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

    correlation_results = analyze_correlation(self)
    regression_result = linear_regression_analysis(self)

    return {
        "correlations": correlation_results,
        "regression": regression_result
    }
