import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException,Query
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import uvicorn
from xgboost import XGBRegressor  # ✅ XGBoost Import

# FastAPI Initialization
app = FastAPI()
origins = [
    "https://ecofy-predicts.netlify.app",  # Your frontend URL
    "http://localhost:3000",  # For local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # ✅ Allows frontend to access backend
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # ✅ Allow all headers
)

# Database Configuration (Replace with your DB credentials)
# DATABASE_URL = "postgresql://neondb_owner:npg_waTG4VbY3KsJ@ep-solitary-field-a8mj30kp-pooler.eastus2.azure.neon.tech/neondb?sslmode=require"
DATABASE_URL = "postgresql://neondb_owner:npg_tqQ57UlGjvNO@ep-ancient-base-a1tmbt57-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define Prediction Model (SQLAlchemy)
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    predicted_energy = Column(Float, nullable=False)
    predicted_emission = Column(Float, nullable=False)
    production_volume = Column(Float, nullable=False)
    waste_generated = Column(Float, nullable=False)
    employee_count = Column(Integer, nullable=False)

# Create Tables
Base.metadata.create_all(bind=engine)

# Global Models
energy_model = None
emission_model = None

@app.post("/train/")
async def train_model(file: UploadFile = File(...)):
    """ Train XGBoost models for energy consumption and carbon emissions. """
    global energy_model, emission_model
    
    contents = await file.read()
    data = pd.read_csv(io.BytesIO(contents))
    
    # Handle missing values
    data = data.dropna()

    required_columns = ['production_volume', 'waste_generated', 'employee_count', 'energy_consumption', 'carbon_emission']
    if not all(col in data.columns for col in required_columns):
        raise HTTPException(status_code=400, detail=f"Dataset must contain columns: {', '.join(required_columns)}")

    # Train XGBoost Models
    X = data[['production_volume', 'waste_generated', 'employee_count']]
    y_energy = data['energy_consumption']
    y_emission = data['carbon_emission']

    energy_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    emission_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

    energy_model.fit(X, y_energy)
    emission_model.fit(X, y_emission)

    return {"message": "XGBoost models trained successfully!"}



@app.get("/analyze/")
async def analyze(
    production_volume: float, waste_generated: float, employee_count: int, years: int = 1
):
    """ Analyze predictions based on input parameters. """
    global energy_model, emission_model
    
    if energy_model is None or emission_model is None:
        raise HTTPException(status_code=500, detail="Models are not trained yet! Please train them first.")
    
    try:
        input_features = np.array([[production_volume, waste_generated, employee_count]])
        predicted_energy = float(energy_model.predict(input_features)[0])
        predicted_emission = float(emission_model.predict(input_features)[0])
        growth_factor = 1 + (years * 0.05)
        predicted_energy *= growth_factor
        predicted_emission *= growth_factor
        
        return {
            "production_volume": production_volume,
            "waste_generated": waste_generated,
            "employee_count": employee_count,
            "years": years,
            "predicted_energy": round(predicted_energy, 2),
            "predicted_emission": round(predicted_emission, 2)
        }
    except Exception as e:
        logging.error(f"Error analyzing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/get_updated_values/")
async def get_updated_values(years: int = Query(5, description="Number of years for analysis")):
    # Dummy data - Replace with actual logic
    return {
        "production_volume": 120 + years * 2,
        "waste_generated": 30 + years,
        "employee_count": 50 + years // 2
    }
@app.post("/predict/")
async def predict_energy_and_emission(file: UploadFile = File(...), k: int = Form(...)):
    """ Predicts energy consumption and carbon emission for future years and stores in MySQL. """
    if energy_model is None or emission_model is None:
        raise HTTPException(status_code=500, detail="Models have not been trained yet. Please upload a training dataset.")
    
    contents = await file.read()
    data = pd.read_csv(io.BytesIO(contents))
    data = data.dropna()

    required_columns = ['production_volume', 'waste_generated', 'employee_count']
    if not all(col in data.columns for col in required_columns):
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(required_columns)}")

    # Compute future employee count
    mean_values = data[required_columns].mean()
    future_employee_count = int(mean_values['employee_count'] + (k * 10))  # Convert to int

    future_data = np.array([[mean_values['production_volume'], mean_values['waste_generated'], future_employee_count]])
    predicted_energy = float(energy_model.predict(future_data)[0])  # Convert to float
    predicted_emission = float(emission_model.predict(future_data)[0])  # Convert to float

    # Save Prediction to Database
    db = SessionLocal()
    new_prediction = Prediction(
        predicted_energy=predicted_energy,
        predicted_emission=predicted_emission,
        production_volume=float(mean_values['production_volume']),
        waste_generated=float(mean_values['waste_generated']),
        employee_count=future_employee_count
    )
    db.add(new_prediction)
    db.commit()
    db.close()

    # Generate Improved Prediction Plot
    plt.figure(figsize=(12, 6))
    categories = ['Production Volume', 'Waste Generated', 'Employee Count', 'Predicted Energy Consumption', 'Predicted Carbon Emission']
    values = [
        round(mean_values['production_volume'], 2),
        round(mean_values['waste_generated'], 2),
        future_employee_count,
        round(predicted_energy, 2),
        round(predicted_emission, 2)
    ]
    colors = ['blue', 'orange', 'red', 'green', 'purple']

    plt.bar(categories, values, color=colors)
    plt.xlabel('Factors')
    plt.ylabel('Values')
    plt.title(f'Predicted Values for Future {k} Years')
    
    # Save Plot
    img_path = "prediction_plot.png"
    plt.savefig(img_path)
    plt.close()

    return JSONResponse(content={
        "predicted_energy_consumption": f"{predicted_energy:.2f} kWh",
        "predicted_carbon_emission": f"{predicted_emission:.2f} kg CO₂",
        "production_volume": mean_values['production_volume'],
        "waste_generated": mean_values['waste_generated'],
        "employee_count": future_employee_count,
        "message": f"Predicted energy consumption after {k} years: {predicted_energy:.2f} kWh and predicted carbon emission: {predicted_emission:.2f} kg CO₂",
                "plot_url": "/plot/"
    })

@app.get("/visualize/")
async def visualize(k: int = Query(...), chart_type: str = Query("line")):
    """ Generates different visualization charts based on predicted values. """
    try:
        years = list(range(1, k+1))
        values = [i * 10 for i in years]  # Replace with actual predictions

        plt.figure(figsize=(6, 4))
        if chart_type == "line":
            plt.plot(years, values, marker="o", linestyle="-", color="b")
            plt.xlabel("Years")
            plt.ylabel("Predicted Values")
        elif chart_type == "bar":
            plt.bar(years, values, color="blue")
            plt.xlabel("Years")
            plt.ylabel("Predicted Values")
        elif chart_type == "scatter":
            plt.scatter(years, values, color="red")
            plt.xlabel("Years")
            plt.ylabel("Predicted Values")
        elif chart_type == "pie":
            plt.pie(values, labels=[f"Year {y}" for y in years], autopct="%1.1f%%")
        
        plt.title(f"{chart_type.capitalize()} Chart of Predictions")

        # Convert plot to response
        img_io = io.BytesIO()
        plt.savefig(img_io, format="png")
        img_io.seek(0)
        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}
@app.get("/plot/")
async def get_plot():
    """ Returns the prediction plot image. """
    img_path = "prediction_plot.png"
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Plot not found")
    return StreamingResponse(open(img_path, "rb"), media_type="image/png")


@app.get("/", response_class=HTMLResponse)
@app.head("/")
async def main_page():
    """ Returns an HTML form for file upload. """
    html_content = """
    <html>
    <head><title>Energy and Emission Prediction</title></head>
    <body>
        <h2>Upload CSV for Training</h2>
        <form action="/train/" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <button type="submit">Train Model</button>
        </form>
        <h2>Upload CSV for Prediction</h2>
        <form action="/predict/" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="number" name="k" placeholder="Enter number of years">
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import os
    import uvicorn

    PORT = int(os.getenv("PORT", 8000))  # Get the PORT from the environment (Render provides this dynamically)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
