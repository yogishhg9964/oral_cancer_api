# Start with an official TensorFlow image, which has many dependencies pre-installed
FROM tensorflow/tensorflow:2.15.0

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the remaining dependencies
# TensorFlow is already installed, so this will be much faster
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy the application code and models
COPY ./app /app/app
COPY ./models_store /app/models_store

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]