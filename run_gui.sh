# Parse command line arguments
DEBUG_ARG=""
MOCK_ARG=""
CLASSIFICATION_ARG=""
MAKE_ARG=""
COLOR_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_ARG="--debug"
            shift
            ;;
        --mock)
            MOCK_ARG="--mock"
            shift
            ;;
        --enable-classification)
            CLASSIFICATION_ARG="--enable-classification"
            shift
            ;;
        --enable-make)
            MAKE_ARG="--enable-make"
            shift
            ;;
        --enable-color)
            COLOR_ARG="--enable-color"
            shift
            ;;
        *)
            # unknown option
            shift
            ;;
    esac
done#!/bin/bash

# ANPR System GUI Launcher Script
# This script checks dependencies and launches the GUI application

# Set working directory to script location
cd "$(dirname "$0")"

# Display header
echo "======================================================"
echo "  ANPR System GUI for Orange Pi 5 Ultra - RK3588 NPU  "
echo "======================================================"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.x."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python $PYTHON_VERSION"

# Check for virtual environment or create one
if [ ! -d "venv" ]; then
    echo "Virtual environment not found, creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check for required Python packages
echo "Checking Python dependencies..."
MISSING_DEPS=0

check_package() {
    python3 -c "import $1" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Warning: Package '$1' is not installed."
        MISSING_DEPS=1
    fi
}

check_package numpy
check_package cv2
check_package PIL
check_package tkinter
check_package threading
check_package queue
check_package logging

# Check for RKNN modules
echo "Checking RKNN modules..."
if python3 -c "from rknnlite.api import RKNNLite" 2>/dev/null; then
    echo "RKNN Lite found - using RKNNLite API"
elif python3 -c "from rknn.api import RKNN" 2>/dev/null; then
    echo "RKNN Toolkit found - using RKNN API"
else
    echo "Warning: Neither RKNN Lite nor RKNN Toolkit found. Running in mock mode."
    MOCK_ARG="--mock"
fi

# Check for GPIO module
echo "Checking GPIO module..."
if ! python3 -c "import OPi.GPIO" 2>/dev/null; then
    echo "Warning: OPi.GPIO module not found. GPIO control will be simulated."
    MOCK_ARG="--mock"
fi

# Check for model file
if [ ! -f "models/yolo11s.rknn" ]; then
    echo "Warning: YOLO model file not found at models/yolo11s.rknn"
    if [ -z "$MOCK_ARG" ]; then
        echo "         Running in mock mode."
        MOCK_ARG="--mock"
    fi
fi

# Check for root if not in mock mode
if [ -z "$MOCK_ARG" ] && [ "$EUID" -ne 0 ]; then
    echo "Warning: Not running as root. GPIO control may not work."
    echo "         Consider running with 'sudo ./run_gui.sh' or configure udev rules."
fi

# Create necessary directories
mkdir -p models captures logs

# Parse command line arguments
DEBUG_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_ARG="--debug"
            shift
            ;;
        --mock)
            MOCK_ARG="--mock"
            shift
            ;;
        *)
            # unknown option
            shift
            ;;
    esac
done

# Launch GUI application
echo "Starting ANPR System GUI..."
if [ $MISSING_DEPS -eq 1 ]; then
    echo "Warning: Some dependencies are missing. The application may not work correctly."
    echo "         Install dependencies with: pip install -r requirements.txt"
fi

# Check for vehicle classification model
if [ "$MAKE_ARG" != "" ] && [ ! -f "models/resnet18_makes.rknn" ]; then
    echo "Warning: Make detection model not found at models/resnet18_makes.rknn"
    echo "         Make detection may not work correctly."
fi

echo "Command: python3 gui.py $MOCK_ARG $DEBUG_ARG $CLASSIFICATION_ARG $MAKE_ARG $COLOR_ARG"
python3 gui.py $MOCK_ARG $DEBUG_ARG $CLASSIFICATION_ARG $MAKE_ARG $COLOR_ARG
