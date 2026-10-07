# Use Python 3.11 on Ubuntu — PyBullet works on Linux, not macOS
FROM python:3.11-slim-bookworm

# Install system dependencies for PyBullet, OpenCV, and C/C++ compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    x11-apps \
    xvfb \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyBullet explicitly (needs Linux)
RUN pip install --no-cache-dir pybullet

# Copy entire project
COPY . .

# Add near the bottom of Dockerfile before CMD
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python", "main.py", "--interactive"]