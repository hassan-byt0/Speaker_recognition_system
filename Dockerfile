# Use the official PyTorch image as the base image
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install dependencies including FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Copy the requirements file into the container
COPY app/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code into the container
COPY app /app

# Copy the datasets into the container
COPY Dataset /app/Dataset

# Make port 80 available to the world outside this container (if applicable)
EXPOSE 80

# Define environment variable (if needed)
ENV NAME SpeakerRecognition

# Command to run the application
CMD ["python", "main.py"]
