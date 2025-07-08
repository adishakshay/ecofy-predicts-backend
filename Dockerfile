# Use the official Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "emission_model:app", "--host", "0.0.0.0", "--port", "8000"]
