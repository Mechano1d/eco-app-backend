import base64
from io import BytesIO
import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import time
from pydantic import BaseModel
from scipy.stats import pearsonr
import folium
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import app.data_collector as data_collector
import app.data_analyzer as data_analyzer
from app.pathfinding import pathfinding
from folium.plugins import HeatMap
import random
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, List, Dict, Any, Tuple
import os

# Налаштування з'єднання з базою даних PostgreSQL
url_object = URL.create(
    "postgresql",
    username="postgres",
    password="thesenate",  # plain (unescaped) text
    host="localhost",
    database="ecoanalysis",
)
DATABASE_URL = "postgresql://Mechanoid:thesenate@localhost:5432/ecoanalysis"
engine = create_engine(url_object)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Створення всіх таблиць у базі даних
Base.metadata.create_all(bind=engine)


# Залежність для отримання сесії бази даних
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class RouteRequest(BaseModel):
    start: Tuple[float, float]
    end: Tuple[float, float]

class TransportEcoAnalysis:
    def __init__(self, city_name, openweather_api_key="73883df466fd45fd40e89c4da87d1d65", waqi_api_key=None, db=None):
        """
        Ініціалізація системи аналізу впливу транспорту на екологію

        Параметри:
        ----------
        city_name : str
            Назва міста для аналізу
        openweather_api_key : str
            API ключ для OpenWeatherMap
        waqi_api_key : str
            API ключ для World Air Quality Index
        """
        self.city_name = city_name
        self.openweather_api_key = openweather_api_key
        self.waqi_api_key = waqi_api_key
        self.db = db
        self.engine = engine
        self.traffic_api_key = "1hscno43lFy01NDffAXrTFOWL7NSXSb2"
        self.graph = None
        self.nodes = None
        self.edges = None
        self.pollution_data = pd.DataFrame()

    def plot_city_graph(self):
        """Візуалізація графа доріг міста"""
        if self.graph is None:
            print("Спочатку завантажте граф міста за допомогою get_city_graph()")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        ox.plot_graph(self.graph, ax=ax, node_size=10, edge_linewidth=0.5)
        plt.title(f"Дорожня мережа міста {self.city_name}")
        plt.tight_layout()
        plt.show()

    def sample_nodes(self, n=200):
        """Вибір випадкових точок з графа для аналізу"""
        if self.nodes is None:
            print("Спочатку завантажте граф міста за допомогою get_city_graph()")
            return None

        # Вибираємо випадкові вузли з графа
        sample_nodes = self.nodes.sample(min(n, len(self.nodes)))
        return sample_nodes

    def create_pollution_map(self):
        """Створення карти забруднення з використанням HeatMap"""
        if self.pollution_data.empty:
            print("Спочатку зберіть дані про забруднення")
            return

        # Створюємо карту, центровану на середньому положенні точок
        center_lat = self.pollution_data['latitude'].mean()
        center_lon = self.pollution_data['longitude'].mean()

        map_folium = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        # Додаємо шар доріг
        folium.TileLayer('cartodbpositron').add_to(map_folium)

        # Підготовка даних для HeatMap
        heat_data = []

        for _, row in self.pollution_data.iterrows():
            lat = row['latitude']
            lon = row['longitude']

            # Пріоритетне джерело – OpenWeatherMap, потім WAQI
            if 'ow_aqi' in self.pollution_data.columns and pd.notnull(row['ow_aqi']):
                aqi_value = float(5 - row['ow_aqi'])
            elif 'waqi_aqi' in self.pollution_data.columns and pd.notnull(row['waqi_aqi']):
                aqi_value = float(5 - row['waqi_aqi'])
            else:
                aqi_value = None

            # Додаємо до heat_data лише якщо є AQI
            if aqi_value is not None:
                heat_data.append([lat, lon, aqi_value])

        # Додаємо теплову карту
        if heat_data:
            HeatMap(heat_data, min_opacity=0.1, max_zoom=1, radius=10, blur=0.01).add_to(map_folium)
        else:
            print("Немає валідних даних AQI для побудови теплової карти")

        # Легенда (залишається як довідка)
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; right: 50px; width: 180px; height: 150px;
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    padding: 10px">
        &nbsp; Індекс якості повітря <br>
        &nbsp; <i class="fa fa-circle" style="color:green"></i> Хороший (0-50)<br>
        &nbsp; <i class="fa fa-circle" style="color:yellow"></i> Помірний (51-100)<br>
        &nbsp; <i class="fa fa-circle" style="color:orange"></i> Нездоровий для чутливих груп (101-150)<br>
        &nbsp; <i class="fa fa-circle" style="color:red"></i> Нездоровий (151-200)<br>
        &nbsp; <i class="fa fa-circle" style="color:purple"></i> Дуже нездоровий (>200)<br>
        </div>
        '''

        map_folium.get_root().html.add_child(folium.Element(legend_html))

        return map_folium

# Створюємо екземпляр FastAPI
app = FastAPI(
    title="Система аналізу впливу транспорту на екологію",
    description="API для аналізу впливу транспорту на екологічний стан міста",
    version="1.0.0"
)

# Додаємо CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальний словник для зберігання екземплярів аналізаторів для різних міст
analyzers = {}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Кореневий маршрут з базовою інформацією про API"""
    return """
    <html>
        <head>
            <title>Система аналізу впливу транспорту на екологію міста</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                a { color: #0066cc; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .endpoint { background: #f4f4f4; padding: 10px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Система аналізу впливу транспорту на екологію міста</h1>
            <p>API для аналізу впливу транспортної інфраструктури на екологічні показники міста.</p>

            <h2>Доступні ендпоінти:</h2>
            <div class="endpoint">
                <p><b>GET /cities/{city_name}/initialize</b> - Ініціалізація аналізу для міста</p>
            </div>
            <div class="endpoint">
                <p><b>GET /cities/{city_name}/map</b> - Отримання карти забруднення</p>
            </div>
            <div class="endpoint">
                <p><b>GET /cities/{city_name}/graph</b> - Отримання візуалізації графа доріг</p>
            </div>
            <div class="endpoint">
                <p><b>GET /cities/{city_name}/analyze</b> - Отримання результатів аналізу</p>
            </div>
            <div class="endpoint">
                <p><b>GET /docs</b> - Інтерактивна документація Swagger</p>
            </div>
        </body>
    </html>
    """

# Основні методи


@app.get("/cities/{city_name}/initialize")
async def initialize_city(
        city_name: str,
        openweather_api_key: Optional[str] = Query(None, description="API ключ для OpenWeatherMap"),
        waqi_api_key: Optional[str] = Query(None, description="API ключ для WAQI"),
        db: Session = Depends(get_db)
):
    """
    Ініціалізація аналізу для вказаного міста
    """
    try:
        # Створюємо новий екземпляр аналізатора для міста
        AnalysisApp = TransportEcoAnalysis(
            city_name=city_name,
            openweather_api_key=openweather_api_key,
            waqi_api_key=waqi_api_key,
            db=db
        )

        # Отримуємо граф міста
        if not data_collector.get_city_graph(AnalysisApp, engine):
            raise HTTPException(status_code=404, detail=f"Не вдалося отримати граф доріг для міста {city_name}")

        # Зберігаємо аналізатор у глобальний словник
        analyzers[city_name] = AnalysisApp

        return {
            "status": "success",
            "message": f"Аналіз для міста {city_name} ініціалізовано",
            "city_info": {
                "name": city_name,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cities/{city_name}/collect_data")
async def collect_data(
        city_name: str,
        num_points: Optional[int] = Query(-1, description="Кількість точок для аналізу"),
):
    """
    Збір даних для вказаного міста
    """
    try:
        AnalysisApp = analyzers[city_name]

        if num_points == -1:
            num_points = len(AnalysisApp.nodes)



        # Збираємо дані про забруднення
        data_collector.collect_pollution_data(AnalysisApp, num_points=num_points)

        # Розраховуємо інтенсивність руху
        data_analyzer.calculate_traffic_intensity(AnalysisApp)

        # Зберігаємо аналізатор у глобальний словник
        analyzers[city_name] = AnalysisApp

        # Конветрація типів numpy
        pollution_data_clean = AnalysisApp.pollution_data.astype(object).where(pd.notnull(AnalysisApp.pollution_data), None)

        return {
            "status": "success",
            "message": f"Зібрано дані з {len(AnalysisApp.pollution_data)} точок даних для міста {city_name}",
            "analyzed_points": len(AnalysisApp.pollution_data),
            "data": pollution_data_clean
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cities/{city_name}/pollution")
async def get_city_pollution(city_name: str):
    """
        Отримання інтерактивної карти забруднення для міста
        """
    if city_name not in analyzers:
        raise HTTPException(status_code=404, detail=f"Місто {city_name} не ініціалізовано")
    AnalysisApp = analyzers[city_name]
    try:
        pollution_data = AnalysisApp.pollution_data.to_dict(orient="records")
        clean_data = [
            {
                "latitude": float(item["latitude"]),
                "longitude": float(item["longitude"]),
                "aqi": int(item["ow_aqi"])
            }
            for item in pollution_data
        ]
        return JSONResponse(content=clean_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cities/{city_name}/route")
async def calculate_route(city_name: str, data: RouteRequest):
    if city_name not in analyzers:
        raise HTTPException(status_code=404, detail=f"Місто {city_name} не ініціалізовано")

    AnalysisApp = analyzers[city_name]
    pathfinder = pathfinding(AnalysisApp)
    print(f"data:{data.start}")
    print(f"data:{data.end}")
    start = ox.nearest_nodes(AnalysisApp.graph, data.start[1], data.start[0])
    end = ox.nearest_nodes(AnalysisApp.graph, data.end[1], data.end[0])
    print(f"start:{start}")
    print(f"end:{end}")
    try:
        route = pathfinder.pathfind(start, end)
        latlons = [(AnalysisApp.nodes.loc[node].geometry.y,
                    AnalysisApp.nodes.loc[node].geometry.x) for node in route]
        print(route)
        return latlons
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cities/{city_name}/full_analysis", response_model=Dict[str, Any])
async def full_analysis(city_name: str):
    """
    Повний аналіз впливу транспорту на екологію:
    кореляція, регресія, кластеризація, гарячі точки, розподіл AQI
    """
    if city_name not in analyzers:
        raise HTTPException(status_code=404, detail=f"Місто {city_name} не ініціалізовано")

    AnalysisApp = analyzers[city_name]

    try:



        # Гарячі точки
        hotspots = []
        if 'ow_aqi' in AnalysisApp.pollution_data.columns:
            top = AnalysisApp.pollution_data.sort_values('ow_aqi', ascending=False).head(3)
            source = "OpenWeatherMap"
            aqi_col = 'ow_aqi'
        elif 'waqi_aqi' in AnalysisApp.pollution_data.columns:
            top = AnalysisApp.pollution_data.sort_values('waqi_aqi', ascending=False).head(3)
            source = "WAQI"
            aqi_col = 'waqi_aqi'
        else:
            top = []
            aqi_col = None
            source = ""

        for _, row in top.iterrows():
            hotspots.append({
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "aqi": float(row[aqi_col]),
                "source": source
            })

        # Кластеризація
        cluster_data = data_analyzer.analyze_clusters(self=AnalysisApp, n_clusters=5)

        # Кореляції
        correlation_results = data_analyzer.analyze_correlations(AnalysisApp)

        # Регресійні моделі
        regressions = data_analyzer.analyze_regressions(AnalysisApp)

        # Розподіл AQI
        aqi_distribution = data_analyzer.aqi_distribution(AnalysisApp)

        co_distribution = data_analyzer.param_distribution(AnalysisApp, "ow_co")

        no2_distribution = data_analyzer.param_distribution(AnalysisApp, "ow_no2")

        pm_2_5_distribution = data_analyzer.param_distribution(AnalysisApp, "ow_pm2_5")

        pm_10_distribution = data_analyzer.param_distribution(AnalysisApp, "ow_pm10")

        # Підсумок
        analysis_results = {
            "city_name": city_name,
            "analyzed_points": len(AnalysisApp.pollution_data),
            "correlations": correlation_results,
            "regression_models": regressions,
            "clusters": cluster_data["points"],
            "cluster_summary": cluster_data["cluster_summary"],
            "aqi_distribution": aqi_distribution,
            "co_distribution": co_distribution,
            "no2_distribution": no2_distribution,
            "pm_2_5_distribution": pm_2_5_distribution,
            "pm_10_distribution": pm_10_distribution,
            "hotspots": hotspots,
            "recommendations": [
                "Моніторинг забруднення у виявлених гарячих точках",
                "Регулювання транспортних потоків у районах з високим рівнем забруднення",
                "Розгляд можливості створення зон з низьким рівнем викидів"
            ]
        }

        return analysis_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Додаткові методи


@app.get("/cities/{city_name}/map", response_class=HTMLResponse)
async def get_city_map(city_name: str):
    """
    Отримання інтерактивної карти забруднення для міста
    """
    if city_name not in analyzers:
        raise HTTPException(status_code=404, detail=f"Місто {city_name} не ініціалізовано")
    AnalysisApp = analyzers[city_name]
    try:
        # Отримуємо HTML-код карти
        pollution_map = AnalysisApp.create_pollution_map()
        if pollution_map:
            return pollution_map.get_root().render()
        else:
            raise HTTPException(status_code=500, detail="Не вдалося створити карту забруднення")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cities/{city_name}/graph", response_class=HTMLResponse)
async def get_city_graph(city_name: str):
    """
    Отримання візуалізації графа доріг міста
    """
    if city_name not in analyzers:
        raise HTTPException(status_code=404, detail=f"Місто {city_name} не ініціалізовано")
    analyzer = analyzers[city_name]
    try:
        # Створюємо візуалізацію графа
        fig, ax = analyzer.plot_city_graph()
        if fig:
            # Зберігаємо графік у байтовий потік
            buffer = BytesIO()
            fig.savefig(buffer, format="png")
            buffer.seek(0)
            # Кодуємо зображення у base64
            graph_image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

            # Формуємо HTML з зображенням
            html_content = f"""
            <html>
                <head>
                    <title>Граф доріг міста {city_name}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                        h1 {{ color: #333; }}
                        img {{ max-width: 100%; }}
                    </style>
                </head>
                <body>
                    <h1>Граф доріг міста {city_name}</h1>
                    <img src="data:image/png;base64,{graph_image_base64}" alt="Граф доріг">
                </body>
            </html>
            """
            return html_content
        else:
            raise HTTPException(status_code=500, detail="Не вдалося створити візуалізацію графа")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск застосунку при виконанні скрипта напряму
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)