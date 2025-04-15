#!/bin/bash

# Script to convert a PyTorch ResNet18 model to RKNN format for make detection

# Set working directory to script location
cd "$(dirname "$0")"

# Display header
echo "======================================================"
echo "  ResNet18 to RKNN Converter for Vehicle Make Detection  "
echo "======================================================"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.x."
    exit 1
fi

# Check if PyTorch is installed
if ! python3 -c "import torch" &> /dev/null; then
    echo "Error: PyTorch is not installed. Please install PyTorch."
    echo "pip install torch==2.0.1 torchvision==0.15.2"
    exit 1
fi

# Check if RKNN is installed
if ! python3 -c "from rknn.api import RKNN" &> /dev/null; then
    echo "Error: RKNN Toolkit is not installed. Please install RKNN Toolkit."
    echo "For Python 3.10: pip install rknn_toolkit2-2.3.2-cp310-cp310-linux_aarch64.whl"
    exit 1
fi

# Check for model argument
if [ -z "$1" ]; then
    echo "Error: Please provide the path to the ResNet18 PyTorch model (.pth or .pt)."
    echo "Usage: $0 <model_path> [output_path]"
    exit 1
fi

MODEL_PATH="$1"
OUTPUT_PATH="${2:-models/resnet18_makes.rknn}"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Python script for model conversion
python3 - << EOF
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

# Check if the model file exists
if not os.path.exists("$MODEL_PATH"):
    print(f"Error: Model file not found: $MODEL_PATH")
    sys.exit(1)

print("Loading PyTorch model...")

try:
    # Load the model
    # If it's a custom model, you may need to modify this part
    model = torch.load("$MODEL_PATH", map_location='cpu')
    
    print("Converting to ONNX format...")
    
    # Set up dummy input for tracing
    dummy_input = Variable(torch.randn(1, 3, 224, 224))
    
    # Export the model to ONNX
    onnx_path = "$MODEL_PATH.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True, 
                     opset_version=11, input_names=['input'], 
                     output_names=['output'])
    
    print(f"ONNX model saved to: {onnx_path}")
    
    # Convert ONNX to RKNN
    print("Converting ONNX to RKNN...")
    
    from rknn.api import RKNN
    
    # Create RKNN object
    rknn = RKNN(verbose=True)
    
    # Pre-process config
    print("Configuring preprocessing...")
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], 
                target_platform='rk3588')
    
    # Load ONNX model
    print(f"Loading ONNX model: {onnx_path}")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"Failed to load ONNX model: {ret}")
        sys.exit(1)
    
    # Build model
    print("Building RKNN model...")
    ret = rknn.build(do_quantization=True)
    
    if ret != 0:
        print(f"Failed to build RKNN model: {ret}")
        sys.exit(1)
    
    # Export RKNN model
    print(f"Exporting RKNN model to: $OUTPUT_PATH")
    ret = rknn.export_rknn("$OUTPUT_PATH")
    if ret != 0:
        print(f"Failed to export RKNN model: {ret}")
        sys.exit(1)
    
    print("RKNN model conversion successful!")
    print(f"Model saved to: $OUTPUT_PATH")
    
except Exception as e:
    print(f"Error during model conversion: {str(e)}")
    sys.exit(1)
EOF

# Check if conversion was successful
if [ -f "$OUTPUT_PATH" ]; then
    echo "Conversion completed successfully."
    echo "ResNet18 RKNN model saved to: $OUTPUT_PATH"
else
    echo "Error: Conversion failed. RKNN model not created."
    exit 1
fi
