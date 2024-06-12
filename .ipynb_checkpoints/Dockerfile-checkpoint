# Use the official Python image from the Docker Hub
FROM python:3.10.14

# Set the working directory in the container
WORKDIR /

# Copy the requirements.txt file into the container
COPY requirements.txt requirements.txt

# Install the Python dependencies
RUN python -m pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port your application will run on
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
