#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO11s to RKNN Model Converter for Orange Pi 5 Ultra
Converts YOLO11s PyTorch/ONNX model to RKNN format optimized for RK3588 NPU

This script handles conversion of a YOLO11s model to RKNN format
for use with the ANPR system on Orange Pi 5 Ultra.
"""

import os
import argparse
import numpy as np
from PIL import Image
import cv2

# Version-specific imports for RKNN
try:
    from rknn.api import RKNN
    print("Using RKNN Toolkit v2")
except ImportError:
    print("Error: RKNN Toolkit v2 not found. Please install rknn-toolkit2.")
    print("For Python 3.10: pip install rknn_toolkit2-2.3.2-cp310-cp310-linux_aarch64.whl")
    exit(1)

def create_dataset_list(images_dir, output_file, num_images=20):
    """
    Create a dataset list file for RKNN quantization
    
    Args:
        images_dir: Directory containing images
        output_file: Path to output file
        num_images: Number of images to include
        
    Returns:
        Path to dataset list file
    """
    print(f"Creating dataset list from {images_dir}")
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    # Limit number of images
    if len(image_files) > num_images:
        import random
        random.shuffle(image_files)
        image_files = image_files[:num_images]
    
    # Write to file
    with open(output_file, 'w') as f:
        for image_file in image_files:
            f.write(f"{image_file}\n")
    
    print(f"Created dataset list with {len(image_files)} images: {output_file}")
    return output_file

def create_synthetic_dataset_list(output_file, num_images=20, size=(640, 640)):
    """
    Create a dataset list file with synthetic images for RKNN quantization,
    optimized for license plate detection
    
    Args:
        output_file: Path to output file
        num_images: Number of images to generate
        size: Image size (width, height)
        
    Returns:
        Path to dataset list file
    """
    print(f"Creating synthetic dataset images for quantization")
    
    # Create directory for synthetic images if it doesn't exist
    output_dir = os.path.dirname(output_file)
    synthetic_dir = os.path.join(output_dir, 'synthetic_dataset')
    os.makedirs(synthetic_dir, exist_ok=True)
    
    # Generate synthetic images
    image_files = []
    
    for i in range(num_images):
        # Create random image with appropriate distribution for YOLO models
        # Using more realistic image statistics (mean around 114 as in YOLOv8)
        img = np.random.randint(80, 150, (size[1], size[0], 3), dtype=np.uint8)
        
        # Add some shapes to make it more realistic for license plate detection
        # Rectangle (simulated license plate)
        x1 = np.random.randint(size[0] // 4, size[0] * 3 // 4)
        y1 = np.random.randint(size[1] // 4, size[1] * 3 // 4)
        w = np.random.randint(size[0] // 10, size[0] // 5)
        h = np.random.randint(size[1] // 20, size[1] // 10)
        color = tuple(np.random.randint(180, 255, 3).tolist())  # Brighter color for plate
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, -1)  # Filled rectangle
        
        # Add darker region inside to simulate plate text
        inner_color = tuple(np.random.randint(10, 60, 3).tolist())  # Dark for text
        for j in range(4):  # Add several small rectangles to simulate characters
            cx = x1 + int(w * (j + 1) / 5)
            cy = y1 + h // 2
            cw = w // 10
            ch = h // 2
            cv2.rectangle(img, (cx-cw//2, cy-ch//2), (cx+cw//2, cy+ch//2), inner_color, -1)
        
        # Add car-like shapes in background
        bg_color = tuple(np.random.randint(50, 120, 3).tolist())
        cv2.rectangle(img, (size[0]//4, size[1]//3), (size[0]*3//4, size[1]*2//3), bg_color, -1)
        
        # Save image
        output_path = os.path.join(synthetic_dir, f'synthetic_{i:03d}.jpg')
        cv2.imwrite(output_path, img)
        image_files.append(output_path)
    
    # Write to file
    with open(output_file, 'w') as f:
        for image_file in image_files:
            f.write(f"{image_file}\n")
    
    print(f"Created synthetic dataset with {len(image_files)} images: {output_file}")
    return output_file

def verify_onnx_model(model_path):
    """
    Verify that ONNX model is valid
    
    Args:
        model_path: Path to ONNX model
        
    Returns:
        True if model is valid, False otherwise
    """
    try:
        import onnx
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(f"ONNX model verified: {model_path}")
        return True
    except Exception as e:
        print(f"Error verifying ONNX model: {str(e)}")
        return False

def convert_yolo_to_rknn(model_path, output_path, dataset_path, target_platform='rk3588', 
                         do_quantization=True, verbose=True):
    """
    Convert YOLO model (ONNX format) to RKNN format following airockchip/ultralytics_yolo11 approach
    
    Args:
        model_path: Path to input model (ONNX format)
        output_path: Path to output RKNN model
        dataset_path: Path to dataset list file for quantization
        target_platform: Target platform (default: 'rk3588')
        do_quantization: Whether to perform quantization (default: True)
        verbose: Whether to print verbose output
        
    Returns:
        True if conversion successful, False otherwise
    """
    print(f"Converting {model_path} to RKNN format for {target_platform}")
    
    # Create RKNN object
    rknn = RKNN(verbose=verbose)
    
    # Pre-process config (using airockchip recommendations)
    print("Configuring preprocessing...")
    
    # Aligned with ultralytics_yolo11 approach
    # Note: We're not using mean/std normalization as in traditional CNNs
    # Instead, keeping uint8 range which works better with RKNN quantization
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], 
                target_platform=target_platform,
                optimization_level=3,  # Higher optimization level for RK3588
                quantized_dtype='asymmetric_quantized-u8',  # Use unsigned int8 for NPU
                quantized_algorithm='normal',
                batch_size=1)
    
    # Load ONNX model
    print(f"Loading ONNX model: {model_path}")
    ret = rknn.load_onnx(model=model_path, inputs=['images'])
    if ret != 0:
        print(f"Failed to load ONNX model: {ret}")
        return False
    
    # Build model
    print("Building RKNN model...")
    if do_quantization:
        print(f"Using dataset for quantization: {dataset_path}")
        ret = rknn.build(do_quantization=True, dataset=dataset_path,
                         pre_compile=True)  # Pre-compile for RK3588
    else:
        ret = rknn.build(do_quantization=False, pre_compile=True)
    
    if ret != 0:
        print(f"Failed to build RKNN model: {ret}")
        return False
    
    # Export RKNN model
    print(f"Exporting RKNN model to: {output_path}")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"Failed to export RKNN model: {ret}")
        return False
    
    print("RKNN model conversion successful!")
    print(f"Model saved to: {output_path}")
    return True0:
        print(f"Failed to export RKNN model: {ret}")
        return False
    
    print("RKNN model conversion successful!")
    print(f"Model saved to: {output_path}")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Convert YOLO model to RKNN format for RK3588')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input model path (ONNX format)')
    
    parser.add_argument('--output', type=str, default='models/yolo11s.rknn',
                        help='Output RKNN model path')
    
    parser.add_argument('--dataset-dir', type=str, default=None,
                        help='Directory containing images for quantization')
    
    parser.add_argument('--dataset-file', type=str, default=None,
                        help='Dataset list file for quantization')
    
    parser.add_argument('--synthetic-dataset', action='store_true',
                        help='Create synthetic dataset for quantization')
    
    parser.add_argument('--target', type=str, default='rk3588',
                        help='Target platform (default: rk3588)')
    
    parser.add_argument('--no-quantization', action='store_true',
                        help='Disable quantization')
    
    parser.add_argument('--optimization-level', type=int, default=3,
                        help='Optimization level (0-3, default: 3)')
    
    parser.add_argument('--pre-compile', action='store_true', default=True,
                        help='Enable pre-compilation for target platform')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Verify input model
    if not os.path.exists(args.input):
        print(f"Error: Input model not found: {args.input}")
        return
    
    # Verify input is ONNX
    if not args.input.lower().endswith('.onnx'):
        print(f"Error: Input model must be in ONNX format: {args.input}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Verify ONNX model structure
    if verify_onnx_model(args.input):
        print("ONNX model verified successfully")
    else:
        print("Warning: ONNX model verification failed, but continuing conversion")
    
    # Prepare dataset for quantization
    dataset_path = None
    if not args.no_quantization:
        if args.dataset_file and os.path.exists(args.dataset_file):
            dataset_path = args.dataset_file
            print(f"Using provided dataset file: {dataset_path}")
        elif args.dataset_dir and os.path.isdir(args.dataset_dir):
            dataset_path = 'dataset.txt'
            dataset_path = create_dataset_list(args.dataset_dir, dataset_path)
        elif args.synthetic_dataset:
            dataset_path = 'synthetic_dataset.txt'
            dataset_path = create_synthetic_dataset_list(dataset_path)
        else:
            print("Warning: No dataset provided for quantization. Creating synthetic dataset.")
            dataset_path = 'synthetic_dataset.txt'
            dataset_path = create_synthetic_dataset_list(dataset_path)
    
    # Convert model
    success = convert_yolo_to_rknn(
        args.input, args.output, dataset_path,
        target_platform=args.target,
        do_quantization=not args.no_quantization,
        verbose=args.verbose
    )
    
    if success:
        print("Model conversion completed successfully!")
        print(f"Model saved to: {args.output}")
    else:
        print("Model conversion failed.")

if __name__ == "__main__":
    main()
