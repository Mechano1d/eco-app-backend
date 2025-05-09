import os
import random
import requests
import pandas as pd
import geopandas as gpd
from datetime import datetime
import time
import logging
import osmnx as ox


def get_city_graph(self, engine):
    """Отримання графа доріг міста з бази даних або API"""
    print(f"Завантаження графа доріг для міста {self.city_name}...")

    try:
        # Спробуємо завантажити з бази даних
        nodes = gpd.read_postgis(
            f"SELECT * FROM nodes_{self.city_name.lower().replace(' ', '_')}",
            con=engine, geom_col='geometry'
        )
        edges = gpd.read_postgis(
            f"SELECT * FROM edges_{self.city_name.lower().replace(' ', '_')}",
            con=engine, geom_col='geometry'
        )

        # Перевірка та відновлення структури індексів
        if 'osmid' in nodes.columns:
            nodes = nodes.set_index('osmid')

        # Перевіряємо, чи є потрібні колонки для мультиіндексу
        required_cols = ['u', 'v', 'key']
        if all(col in edges.columns for col in required_cols):
            # Конвертуємо типи даних, якщо потрібно
            for col in ['u', 'v']:
                if edges[col].dtype != 'int64':
                    edges[col] = edges[col].astype('int64')

            # Конвертуємо key до int, якщо це можливо
            if edges['key'].dtype != 'int64':
                try:
                    edges['key'] = edges['key'].astype('int64')
                except:
                    # Якщо конвертація неможлива, залишаємо як є
                    pass

            # Встановлюємо мультиіндекс
            edges = edges.set_index(required_cols)
        else:
            raise ValueError(f"Необхідні колонки {required_cols} не знайдені в edges DataFrame")

        self.nodes = nodes
        self.edges = edges

        # Перевіряємо наявність обов'язкових колонок для graph_from_gdfs
        print(f"Форма індексу edges: {edges.index.names}")
        print(f"Форма індексу nodes: {nodes.index.names}")

        self.graph = ox.graph_from_gdfs(nodes, edges)
        print(f"Граф успішно завантажено з бази даних. {len(self.nodes)} вузлів та {len(self.edges)} доріг.")
        return True

    except Exception as e:
        print(f"Не вдалося завантажити з бази: {e}")
        print("Спроба завантажити граф з OSM API...")

        try:
            self.graph = ox.graph_from_place(self.city_name, network_type='drive')
            self.nodes, self.edges = ox.graph_to_gdfs(self.graph)

            # Збереження в базу даних з чітким збереженням структури індексів
            nodes_to_save = self.nodes.reset_index()
            edges_to_save = self.edges.reset_index()

            # Додаємо додаткову перевірку для edges: переконуємось, що колонки u, v, key існують
            if not all(col in edges_to_save.columns for col in ['u', 'v', 'key']):
                print("Увага: edges DataFrame не містить необхідних колонок для мультиіндексу")

            # Збереження в базу даних
            nodes_to_save.to_postgis(
                name=f'nodes_{self.city_name.lower().replace(" ", "_")}',
                con=engine, if_exists='replace'
            )
            edges_to_save.to_postgis(
                name=f'edges_{self.city_name.lower().replace(" ", "_")}',
                con=engine, if_exists='replace'
            )

            print(f"Граф успішно завантажено з OSM API та збережено в базу даних.")
            return True

        except Exception as e2:
            print(f"Помилка при завантаженні графа з API: {e2}")
            return False


def get_openweather_pollution(self, lat, lon):
    """Отримання даних про забруднення з OpenWeatherMap API"""
    if not self.openweather_api_key:
        # Генеруємо тестові дані для демонстрації
        return {
            'aqi': random.randint(1, 5),
            'co': random.uniform(200, 2000),
            'no2': random.uniform(10, 100),
            'pm2_5': random.uniform(0, 50),
            'pm10': random.uniform(0, 75)
        }

    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={self.openweather_api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        result = data.get('list', [{}])[0]
        components = result.get('components', {})
        aqi = result.get('main', {}).get('aqi', 0)

        return {
            'aqi': aqi,
            'co': components.get('co', 0),
            'no2': components.get('no2', 0),
            'pm2_5': components.get('pm2_5', 0),
            'pm10': components.get('pm10', 0)
        }
    else:
        print(f"Помилка отримання даних OpenWeatherMap: {response.status_code}")
        return None


def get_waqi_pollution(self, lat, lon):
    """Отримання даних про забруднення з WAQI API"""
    if not self.waqi_api_key:
        # Генеруємо тестові дані для демонстрації
        return {
            'aqi': random.randint(0, 300),
            'pm25': random.uniform(0, 75),
            'pm10': random.uniform(0, 100)
        }

    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={self.waqi_api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data.get('status') == 'ok':
            result = data.get('data', {})
            aqi = result.get('aqi', 0)
            iaqi = result.get('iaqi', {})

            return {
                'aqi': aqi,
                'pm25': iaqi.get('pm25', {}).get('v', 0),
                'pm10': iaqi.get('pm10', {}).get('v', 0)
            }

    print(f"Помилка отримання даних WAQI: {response.status_code}")
    return None


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
            ow_data = get_openweather_pollution(self, lat, lon)
            waqi_data = get_waqi_pollution(self, lat, lon)

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
            # print(node_data)
            # time.sleep(0.01)  # Пауза між запитами до API

        self.pollution_data = pd.DataFrame(pollution_data)
        print(f"Зібрано дані про забруднення для {len(self.pollution_data)} точок.")
        return self.pollution_data
