# Use the official Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Use the correct startup command
CMD ["uvicorn", "emission_model:app", "--host", "0.0.0.0", "--port", "8000"]
