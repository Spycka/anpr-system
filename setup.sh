#!/bin/bash
# Setup script for ANPR System on Orange Pi 5 Ultra
# This script installs all dependencies and prepares the environment

# Exit on error
set -e

# Text formatting
BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BOLD}${BLUE}"
echo "========================================================"
echo "   ANPR System Setup for Orange Pi 5 Ultra              "
echo "========================================================"
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}Warning: Not running as root. Some operations may fail.${NC}"
  echo "Consider running with sudo if you encounter permission issues."
  echo ""
  sleep 2
fi

# Function to print status messages
print_status() {
  echo -e "${BOLD}${BLUE}==> ${NC}${BOLD}$1${NC}"
}

# Function to print success messages
print_success() {
  echo -e "${BOLD}${GREEN}==> $1${NC}"
}

# Function to print error messages
print_error() {
  echo -e "${BOLD}${RED}==> Error: $1${NC}"
}

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Create directory structure if it doesn't exist
create_directory_structure() {
  print_status "Creating directory structure..."
  
  # Create directories
  mkdir -p anpr_system/{detectors,vision,input,utils,models,captures,logs}
  
  # Create __init__.py files
  touch anpr_system/{detectors,vision,input,utils}/__init__.py
  
  print_success "Directory structure created"
}

# Update system
update_system() {
  print_status "Updating package lists..."
  apt-get update
  print_success "Package lists updated"
}

# Install system dependencies
install_system_dependencies() {
  print_status "Installing system dependencies..."
  
  # Install required packages
  apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    python3-numpy \
    python3-scipy \
    python3-sklearn \
    git \
    cmake \
    build-essential \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk2.0-dev \
    pkg-config
  
  print_success "System dependencies installed"
}

# Install Python dependencies
install_python_dependencies() {
  print_status "Installing Python dependencies..."
  
  # Upgrade pip
  python3 -m pip install --upgrade pip
  
  # Install required Python packages
  python3 -m pip install \
    opencv-python \
    numpy \
    torch \
    torchvision \
    scikit-learn \
    easyocr \
    pyopencl \
    gstreamer-python \
    tqdm \
    pillow \
    matplotlib
  
  print_success "Python dependencies installed"
}

# Install RKNN toolkit (for NPU support)
install_rknn_toolkit() {
  print_status "Checking for RKNN toolkit..."
  
  # Check if rknn_toolkit_lite is already installed
  if python3 -c "import rknnlite" 2>/dev/null; then
    print_success "RKNN toolkit already installed"
    return
  fi
  
  print_status "Installing RKNN toolkit for NPU support..."
  
  # Clone repository
  if [ ! -d "rknn-toolkit2" ]; then
    git clone https://github.com/rockchip-linux/rknn-toolkit2.git
  fi
  
  # Install rknn-toolkit2-lite
  cd rknn-toolkit2/rknn_toolkit_lite2
  python3 -m pip install -e .
  cd ../..
  
  # Verify installation
  if python3 -c "import rknnlite" 2>/dev/null; then
    print_success "RKNN toolkit installed successfully"
  else
    print_error "Failed to install RKNN toolkit"
    echo "Please install manually following the documentation at:"
    echo "https://github.com/rockchip-linux/rknn-toolkit2"
  fi
}

# Download model files
download_models() {
  print_status "Downloading model files..."
  
  # Create models directory
  mkdir -p anpr_system/models
  cd anpr_system/models
  
  # Note for user
  echo "For YOLO11s.pt and YOLO11s.rknn model files, please download from:"
  echo "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt"
  echo "and convert to RKNN format for optimal performance."
  echo ""
  echo "For ResNet18 vehicle make model, download a pre-trained model or train your own."
  
  # Create placeholder for models to be added manually
  echo "# Add model files here" > README.md
  echo "# Required models:" >> README.md
  echo "# - YOLO11s.pt (for GPU/CPU detection)" >> README.md
  echo "# - YOLO11s.rknn (for NPU acceleration)" >> README.md
  echo "# - resnet18_vehicle_make.pth (for vehicle make detection)" >> README.md
  
  cd ../..
  
  print_success "Model directory prepared"
}

# Create sample allowlist
create_sample_allowlist() {
  print_status "Creating sample allowlist..."
  
  # Create sample allowlist.txt
  cat > anpr_system/allowlist.txt << EOF
# Sample allowlist - one plate per line
# Add your authorized license plates below
ABC123
XYZ789
EOF

  # Create sample JSON allowlist
  cat > anpr_system/allowlist.json << EOF
{
  "ABC123": {
    "make": "Toyota",
    "color": "Blue",
    "owner": "John Doe",
    "notes": "Employee"
  },
  "XYZ789": {
    "make": "Honda",
    "color": "Red",
    "owner": "Jane Smith",
    "notes": "Visitor"
  }
}
EOF

  print_success "Sample allowlist created"
}

# Create systemd service file
create_service_file() {
  print_status "Creating systemd service file..."
  
  # Get current directory
  CURRENT_DIR=$(pwd)
  
  # Create service file
  cat > anpr-system.service << EOF
[Unit]
Description=ANPR System Service for Orange Pi 5 Ultra
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$CURRENT_DIR/anpr_system
ExecStart=/usr/bin/python3 $CURRENT_DIR/anpr_system/main.py --rtsp-url=rtsp://username:password@camera-ip:554/stream --resolution=720p --show-video --allowlist=allowlist.txt --save-dir=captures
Restart=on-failure
RestartSec=5
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=anpr-system

[Install]
WantedBy=multi-user.target
EOF

  echo "Created service file: anpr-system.service"
  echo "To install the service, run:"
  echo "  sudo cp anpr-system.service /etc/systemd/system/"
  echo "  sudo systemctl daemon-reload"
  echo "  sudo systemctl enable anpr-system.service"
  echo "  sudo systemctl start anpr-system.service"
  
  print_success "Service file created"
}

# Create simulation test images
create_test_images() {
  print_status "Creating test images for simulation..."
  
  # Create simulation directory
  mkdir -p anpr_system/simulation_images
  
  # Create Python script to generate test images
  cat > anpr_system/create_test_images.py << EOF
#!/usr/bin/env python3
"""
Generate test images for ANPR system simulation.
"""
import os
import cv2
import numpy as np

# Create simulation directory
sim_dir = "simulation_images"
os.makedirs(sim_dir, exist_ok=True)

# Generate test images
num_images = 5
image_size = (720, 1280, 3)  # 720p
plate_texts = ["ABC123", "XYZ789", "DEF456", "GHI789", "JKL012"]

for i in range(num_images):
    # Create blank image
    img = np.zeros(image_size, dtype=np.uint8)
    
    # Add gradient background
    for y in range(image_size[0]):
        for x in range(image_size[1]):
            img[y, x] = [
                int(255 * (y / image_size[0])),
                int(255 * (x / image_size[1])),
                int(255 * ((x + y) / (image_size[0] + image_size[1])))
            ]
    
    # Add car shape
    car_color = (0, 0, 255) if i % 2 == 0 else (0, 255, 0)
    cv2.rectangle(img, (400, 300), (900, 500), car_color, -1)
    cv2.rectangle(img, (500, 200), (800, 300), car_color, -1)
    
    # Add license plate
    plate_color = (200, 200, 200)
    plate_x1, plate_y1 = 550, 400
    plate_x2, plate_y2 = 750, 450
    cv2.rectangle(img, (plate_x1, plate_y1), (plate_x2, plate_y2), plate_color, -1)
    cv2.rectangle(img, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 0, 0), 2)
    
    # Add plate text
    plate_text = plate_texts[i % len(plate_texts)]
    cv2.putText(
        img, plate_text,
        (plate_x1 + 10, plate_y1 + 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
    )
    
    # Add frame counter
    cv2.putText(
        img, f"Frame {i+1}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    
    # Save image
    filename = os.path.join(sim_dir, f"test_frame_{i+1:03d}.jpg")
    cv2.imwrite(filename, img)
    print(f"Created: {filename}")

print(f"Created {num_images} test images in {sim_dir}")
EOF

  # Make it executable
  chmod +x anpr_system/create_test_images.py
  
  # Run the script
  cd anpr_system
  python3 create_test_images.py
  cd ..
  
  print_success "Test images created for simulation"
}

# Main setup process
main() {
  print_status "Starting setup for ANPR System..."
  
  # Create directory structure
  create_directory_structure
  
  # Update system and install dependencies
  if command_exists apt-get; then
    update_system
    install_system_dependencies
  else
    print_error "apt-get not found. Please install system dependencies manually."
  fi
  
  # Install Python dependencies
  install_python_dependencies
  
  # Install RKNN toolkit (if available)
  install_rknn_toolkit
  
  # Download models
  download_models
  
  # Create sample allowlist
  create_sample_allowlist
  
  # Create service file
  create_service_file
  
  # Create test images
  create_test_images
  
  # Copy Python modules
  print_status "Copying Python modules..."
  
  # Note: at this point, you would typically copy your Python modules
  # from the repository to the anpr_system directory, but since we're
  # working with the files directly, we'll skip this step and just
  # provide instructions.
  
  echo "Please ensure all Python modules are placed in the anpr_system directory."
  echo "The full directory structure should be:"
  echo "  anpr_system/"
  echo "  ├── main.py"
  echo "  ├── detectors/"
  echo "  │   ├── __init__.py"
  echo "  │   ├── yolo11_gpu.py"
  echo "  │   └── yolo11_rknn.py"
  echo "  ├── vision/"
  echo "  │   ├── __init__.py"
  echo "  │   ├── ocr.py"
  echo "  │   └── skew.py"
  echo "  ├── input/"
  echo "  │   ├── __init__.py"
  echo "  │   └── camera.py"
  echo "  ├── utils/"
  echo "  │   ├── __init__.py"
  echo "  │   ├── hardware.py"
  echo "  │   ├── logger.py"
  echo "  │   ├── plate_checker.py"
  echo "  │   ├── vehicle_make.py"
  echo "  │   ├── vehicle_color.py"
  echo "  │   └── gpio.py"
  echo "  ├── models/"
  echo "  ├── captures/"
  echo "  └── logs/"
  
  print_success "Setup completed successfully!"
  print_status "Next steps:"
  echo "1. Download or train the required models and place them in anpr_system/models/"
  echo "2. Update the allowlist with your authorized license plates"
  echo "3. Modify the RTSP URL in main.py or when running to point to your camera"
  echo "4. Run the ANPR system with:"
  echo "   cd anpr_system && python3 main.py --rtsp-url=your-camera-url --show-video"
  echo ""
  echo "For testing without a camera, use:"
  echo "   python3 main.py --simulate=simulation_images --show-video --show-debug"
  echo ""
  echo "Enjoy your ANPR system!"
}

# Run the main function
main
