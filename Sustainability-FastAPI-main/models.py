from sqlalchemy import Column, Integer, Float, DateTime
from database import Base
import datetime

class TrainingData(Base):
    """ Stores the training dataset in MySQL """
    __tablename__ = "training_data"
    
    id = Column(Integer, primary_key=True, index=True)
    production_volume = Column(Float, nullable=False)
    waste_generated = Column(Float, nullable=False)
    employee_count = Column(Float, nullable=False)
    energy_consumption = Column(Float, nullable=False)
    carbon_emission = Column(Float, nullable=False)

class PredictionResult(Base):
    """ Stores predictions in MySQL """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    predicted_energy = Column(Float, nullable=False)
    predicted_emission = Column(Float, nullable=False)
    production_volume = Column(Float, nullable=False)
    waste_generated = Column(Float, nullable=False)
    employee_count = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
