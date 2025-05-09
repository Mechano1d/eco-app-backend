from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


# Моделі для бази даних
class City(Base):
    __tablename__ = "cities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.now)
    nodes_count = Column(Integer)
    edges_count = Column(Integer)

    nodes = relationship("Node", back_populates="city", cascade="all, delete-orphan")
    pollution_data = relationship("PollutionData", back_populates="city", cascade="all, delete-orphan")


class Node(Base):
    __tablename__ = "nodes"

    id = Column(Integer, primary_key=True, index=True)
    osm_id = Column(String, index=True)
    city_id = Column(Integer, ForeignKey("cities.id"))
    geometry = Column(Geometry(geometry_type='POINT', srid=4326))
    road_count = Column(Integer)
    traffic_intensity = Column(Float, nullable=True)

    city = relationship("City", back_populates="nodes")
    pollution_data = relationship("PollutionData", back_populates="node", cascade="all, delete-orphan")


class PollutionData(Base):
    __tablename__ = "pollution_data"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(Integer, ForeignKey("nodes.id"))
    city_id = Column(Integer, ForeignKey("cities.id"))
    measured_at = Column(DateTime, default=datetime.now)
    ow_aqi = Column(Float, nullable=True)
    ow_co = Column(Float, nullable=True)
    ow_no2 = Column(Float, nullable=True)
    ow_pm2_5 = Column(Float, nullable=True)
    ow_pm10 = Column(Float, nullable=True)
    waqi_aqi = Column(Float, nullable=True)
    waqi_pm25 = Column(Float, nullable=True)
    waqi_pm10 = Column(Float, nullable=True)

    node = relationship("Node", back_populates="pollution_data")
    city = relationship("City", back_populates="pollution_data")