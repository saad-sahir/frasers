# Use a lightweight Python base image
FROM python:3.9-slim

# Set up a working directory
WORKDIR /app

# Copy the Python script and weights into the container
COPY revsc1.py /app/revsc1.py
COPY requirements.txt /app/requirements.txt
COPY weights/ /app/weights/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenGL library required by OpenCV
RUN apt-get update && apt-get install -y libgl1

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

RUN pip install opencv-python-headless
    
# Set the default command to run the scenario script
CMD ["python", "revsc1.py"]