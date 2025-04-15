#!/bin/bash

# ANPR System Setup Script for Orange Pi 5 Ultra
# This script sets up all the dependencies and configurations required for the ANPR system

echo "Setting up ANPR System for Orange Pi 5 Ultra..."

# Exit on error
set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

# Update system
echo "Updating system packages..."
apt update
apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
apt install -y python3-dev python3-pip libopencv-dev libopenblas-dev \
               python3-tk mesa-opencl-icd ocl-icd-opencl-dev clinfo \
               libxcb-dri2-0 libxcb-dri3-0 libwayland-client0 libwayland-server0 \
               libx11-xcb1 ffmpeg libavcodec-dev libavformat-dev git cmake build-essential

# Setup Mali GPU drivers
echo "Setting up Mali GPU drivers for Orange Pi 5 Ultra..."
cd /usr/lib
wget https://github.com/JeffyCN/mirrors/raw/libmali/lib/aarch64-linux-gnu/libmali-valhall-g610-g6p0-x11-wayland-gbm.so -O libmali-g610.so

# Setup Mali firmware
cd /lib/firmware
wget https://github.com/JeffyCN/mirrors/raw/libmali/firmware/g610/mali_csffw.bin

# Configure OpenCL
echo "Configuring OpenCL..."
mkdir -p /etc/OpenCL/vendors
echo "/usr/lib/libmali-g610.so" > /etc/OpenCL/vendors/mali.icd

# Setup GPIO permissions
echo "Setting up GPIO permissions..."
echo 'SUBSYSTEM=="gpio", KERNEL=="gpiochip*", ACTION=="add", PROGRAM="/bin/sh -c '\''chown root:gpio /dev/gpiochip* && chmod 660 /dev/gpiochip*'\''"' > /etc/udev/rules.d/10-gpio.rules
groupadd -f gpio
usermod -a -G gpio $USER
udevadm control --reload-rules && udevadm trigger

# Install RKNN Toolkit
echo "Installing RKNN Toolkit..."
# Determine Python version
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
RKNN_WHEEL="rknn_toolkit2-2.3.2-cp${PY_VERSION/./}-cp${PY_VERSION/./}-linux_aarch64.whl"

mkdir -p /tmp/rknn_setup
cd /tmp/rknn_setup

# Download the correct RKNN wheel
echo "Downloading RKNN Toolkit for Python $PY_VERSION..."
wget "https://github.com/airockchip/rknn-toolkit2/releases/download/v2.3.2/$RKNN_WHEEL"

# Install the wheel
pip3 install "$RKNN_WHEEL"

# Create directory structure
echo "Creating directory structure..."
mkdir -p /opt/anpr-system
mkdir -p /opt/anpr-system/models
mkdir -p /opt/anpr-system/captures
mkdir -p /opt/anpr-system/logs

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Copy YOLO model if it exists in the current directory
if [ -f "models/yolo11s.rknn" ]; then
    echo "Copying YOLO model..."
    cp models/yolo11s.rknn /opt/anpr-system/models/
fi

# Create default allowlist if it doesn't exist
if [ ! -f "/opt/anpr-system/allowlist.txt" ]; then
    echo "Creating default allowlist..."
    touch /opt/anpr-system/allowlist.txt
fi

# Create systemd service
echo "Creating systemd service..."
cat > /etc/systemd/system/anpr-system.service << EOL
[Unit]
Description=ANPR System Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/anpr-system
ExecStart=/usr/bin/python3 /opt/anpr-system/main.py
Restart=on-failure
RestartSec=5
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=anpr-system

[Install]
WantedBy=multi-user.target
EOL

# Enable and start the service
echo "Enabling and starting ANPR service..."
systemctl daemon-reload
systemctl enable anpr-system.service
systemctl start anpr-system.service

# Create convenience scripts
echo "Creating convenience scripts..."
cat > /opt/anpr-system/run_gui.sh << EOL
#!/bin/bash
cd /opt/anpr-system
python3 gui.py
EOL

chmod +x /opt/anpr-system/run_gui.sh

# Verify installation
echo "Verifying installation..."
python3 -c "import numpy, cv2, PIL, OPi.GPIO; from rknnlite.api import RKNNLite; print('Basic dependencies loaded successfully')"

echo "Verifying OpenCL setup..."
clinfo | grep -q "Mali" && echo "Mali GPU detected!" || echo "Mali GPU not detected. Please check GPU driver installation."

echo "========================================"
echo "ANPR System installation complete!"
echo "Run the GUI with: /opt/anpr-system/run_gui.sh"
echo "Check service status with: systemctl status anpr-system"
echo "========================================"
