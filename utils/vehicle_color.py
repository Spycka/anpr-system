#!/usr/bin/env python3
"""
Vehicle color detection using K-means clustering.
Analyzes vehicle images to determine the most prominent colors.
"""
import os
import time
import logging
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional, Union
from collections import Counter

# Try importing scikit-learn for K-means
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import logger
from utils.logger import get_logger
logger = get_logger("vehicle_color")

class ColorDetector:
    """
    Vehicle color detector using K-means clustering.
    
    Identifies the dominant colors in a vehicle image and maps them
    to common color names (red, blue, green, black, white, etc.).
    """
    
    # Basic color name mapping (RGB format)
    COLOR_NAMES = {
        (0, 0, 0): "black",
        (255, 255, 255): "white",
        (128, 128, 128): "gray",
        (255, 0, 0): "red",
        (0, 255, 0): "green",
        (0, 0, 255): "blue",
        (255, 255, 0): "yellow",
        (255, 0, 255): "magenta",
        (0, 255, 255): "cyan",
        (165, 42, 42): "brown",
        (255, 165, 0): "orange",
        (128, 0, 0): "maroon",
        (128, 0, 128): "purple",
        (0, 128, 0): "dark green",
        (0, 0, 128): "navy",
        (255, 192, 203): "pink",
        (192, 192, 192): "silver"
    }
    
    def __init__(
        self,
        n_clusters: int = 5,
        samples_limit: int = 10000,
        exclude_plate_region: bool = True,
        verbose: bool = False
    ):
        """
        Initialize color detector.
        
        Args:
            n_clusters: Number of color clusters for K-means
            samples_limit: Maximum number of pixels to sample
            exclude_plate_region: Whether to exclude license plate region from analysis
            verbose: Enable verbose logging
        """
        self.n_clusters = n_clusters
        self.samples_limit = samples_limit
        self.exclude_plate_region = exclude_plate_region
        self.verbose = verbose
        
        # Check if sklearn is available
        self.sklearn_available = SKLEARN_AVAILABLE
        
        if not self.sklearn_available:
            logger.warning("scikit-learn not installed. Using OpenCV K-means fallback.")
        
        logger.info(f"Color detector initialized (n_clusters={n_clusters}, "
                  f"exclude_plate_region={exclude_plate_region})")
    
    def _get_color_name(self, rgb_color: Tuple[int, int, int]) -> str:
        """
        Get the nearest color name for an RGB color.
        
        Args:
            rgb_color: RGB color tuple
            
        Returns:
            str: Color name
        """
        min_distance = float('inf')
        color_name = "unknown"
        
        # Convert to RGB if given in BGR
        if isinstance(rgb_color, np.ndarray) and rgb_color.shape[0] == 3:
            r, g, b = rgb_color
        else:
            r, g, b = rgb_color
        
        # Ensure values are in valid range
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))
        
        # Convert from BGR to RGB if needed
        rgb = (r, g, b)
        
        # Calculate Euclidean distance to known colors
        for color_rgb, name in self.COLOR_NAMES.items():
            # Calculate color distance
            distance = np.sqrt(
                (rgb[0] - color_rgb[0]) ** 2 +
                (rgb[1] - color_rgb[1]) ** 2 +
                (rgb[2] - color_rgb[2]) ** 2
            )
            
            if distance < min_distance:
                min_distance = distance
                color_name = name
        
        return color_name
    
    def _create_mask(self, image: np.ndarray, plate_box: Optional[List[int]] = None) -> np.ndarray:
        """
        Create analysis mask, optionally excluding the license plate region.
        
        Args:
            image: Input image
            plate_box: Optional license plate bounding box [x1, y1, x2, y2]
            
        Returns:
            np.ndarray: Binary mask for analysis
        """
        height, width = image.shape[:2]
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Exclude license plate region if provided
        if self.exclude_plate_region and plate_box is not None:
            x1, y1, x2, y2 = plate_box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            # Create a slightly larger exclusion area
            pad = 10
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(width, x2 + pad), min(height, y2 + pad)
            
            mask[y1:y2, x1:x2] = 0
        
        return mask
    
    def _extract_samples(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract color samples from the image using the mask.
        
        Args:
            image: Input image
            mask: Binary mask
            
        Returns:
            np.ndarray: Array of color samples
        """
        # Convert image to RGB for consistent color analysis
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # If grayscale, convert to 3-channel
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Get non-zero mask coordinates
        mask_indices = np.where(mask > 0)
        
        # If no valid pixels, return empty array
        if len(mask_indices[0]) == 0:
            return np.array([])
        
        # Convert to list of coordinates
        coords = list(zip(mask_indices[0], mask_indices[1]))
        
        # Sample pixels if too many
        if len(coords) > self.samples_limit:
            # Random sampling without replacement
            sampled_indices = np.random.choice(
                len(coords), 
                size=self.samples_limit, 
                replace=False
            )
            coords = [coords[i] for i in sampled_indices]
        
        # Extract RGB values
        samples = np.array([rgb_image[y, x] for y, x in coords])
        
        return samples
    
    def _cluster_colors_sklearn(self, samples: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Cluster colors using scikit-learn K-means.
        
        Args:
            samples: Array of color samples
            
        Returns:
            List[Tuple[np.ndarray, float]]: List of (color, weight) tuples
        """
        try:
            # Handle empty samples
            if len(samples) == 0:
                return []
            
            # Use fewer clusters if we have few samples
            n_clusters = min(self.n_clusters, len(samples))
            
            # Cluster colors
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(samples)
            
            # Get cluster centers and counts
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Count occurrences of each label
            label_counts = Counter(labels)
            total = len(samples)
            
            # Prepare weighted colors
            weighted_colors = []
            for cluster_id, center in enumerate(centers):
                count = label_counts.get(cluster_id, 0)
                weight = count / total
                weighted_colors.append((center, weight))
            
            # Sort by weight (descending)
            weighted_colors.sort(key=lambda x: x[1], reverse=True)
            
            return weighted_colors
        
        except Exception as e:
            logger.error(f"Error in sklearn K-means clustering: {str(e)}")
            return []
    
    def _cluster_colors_opencv(self, samples: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Cluster colors using OpenCV K-means.
        
        Args:
            samples: Array of color samples
            
        Returns:
            List[Tuple[np.ndarray, float]]: List of (color, weight) tuples
        """
        try:
            # Handle empty samples
            if len(samples) == 0:
                return []
            
            # Convert to float32
            samples_float = np.float32(samples)
            
            # Use fewer clusters if we have few samples
            n_clusters = min(self.n_clusters, len(samples))
            
            # Define criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            
            # Apply K-means
            _, labels, centers = cv2.kmeans(
                samples_float, 
                n_clusters, 
                None, 
                criteria, 
                10, 
                cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Count occurrences of each label
            labels = labels.flatten()
            label_counts = Counter(labels)
            total = len(samples)
            
            # Prepare weighted colors
            weighted_colors = []
            for cluster_id, center in enumerate(centers):
                count = label_counts.get(cluster_id, 0)
                weight = count / total
                weighted_colors.append((center, weight))
            
            # Sort by weight (descending)
            weighted_colors.sort(key=lambda x: x[1], reverse=True)
            
            return weighted_colors
        
        except Exception as e:
            logger.error(f"Error in OpenCV K-means clustering: {str(e)}")
            return []
    
    def detect_color(self, image: np.ndarray, 
                     plate_box: Optional[List[int]] = None,
                     return_details: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Detect the dominant color of a vehicle.
        
        Args:
            image: Input image (BGR format)
            plate_box: Optional license plate bounding box [x1, y1, x2, y2]
            return_details: Return detailed color information
            
        Returns:
            Union[str, Dict[str, Any]]:
                - If return_details=False: Main color name
                - If return_details=True: Dict with color details
        """
        start_time = time.time()
        
        try:
            # Create analysis mask
            mask = self._create_mask(image, plate_box)
            
            # Extract color samples
            samples = self._extract_samples(image, mask)
            
            # Check if we have samples
            if len(samples) == 0:
                logger.warning("No valid pixels for color analysis")
                return "unknown" if not return_details else {
                    'main_color': "unknown",
                    'colors': [],
                    'processing_time': time.time() - start_time
                }
            
            # Cluster colors
            if self.sklearn_available:
                weighted_colors = self._cluster_colors_sklearn(samples)
            else:
                weighted_colors = self._cluster_colors_opencv(samples)
            
            # Get color names and weights
            named_colors = []
            for color, weight in weighted_colors:
                color_name = self._get_color_name(color)
                named_colors.append({
                    'name': color_name,
                    'rgb': tuple(int(c) for c in color),
                    'weight': weight
                })
            
            # Get main color
            main_color = named_colors[0]['name'] if named_colors else "unknown"
            
            # Log results
            processing_time = time.time() - start_time
            
            if self.verbose:
                color_list = ", ".join([f"{c['name']} ({c['weight']:.2f})" for c in named_colors[:3]])
                logger.info(f"Vehicle color: {main_color} [{color_list}] in {processing_time:.3f}s")
            
            if return_details:
                return {
                    'main_color': main_color,
                    'colors': named_colors,
                    'processing_time': processing_time
                }
            else:
                return main_color
        
        except Exception as e:
            logger.error(f"Error detecting vehicle color: {str(e)}")
            
            if return_details:
                return {
                    'main_color': "unknown",
                    'colors': [],
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }
            else:
                return "unknown"
    
    def visualize_colors(self, image: np.ndarray, 
                         plate_box: Optional[List[int]] = None) -> np.ndarray:
        """
        Create a visualization of the detected colors.
        
        Args:
            image: Input image
            plate_box: Optional license plate bounding box
            
        Returns:
            np.ndarray: Visualization image
        """
        # Detect colors with details
        result = self.detect_color(image, plate_box, return_details=True)
        
        # Copy original image
        vis_img = image.copy()
        height, width = vis_img.shape[:2]
        
        # Create color palette height
        palette_height = 50
        
        # If we have colors to display
        if result['colors']:
            # Create color palette at the bottom
            palette = np.zeros((palette_height, width, 3), dtype=np.uint8)
            
            # Normalize weights for display
            total_weight = sum(color['weight'] for color in result['colors'])
            if total_weight > 0:
                normalized_weights = [color['weight'] / total_weight for color in result['colors']]
            else:
                normalized_weights = [1/len(result['colors'])] * len(result['colors'])
            
            # Calculate bar widths
            widths = [int(width * w) for w in normalized_weights]
            
            # Adjust to ensure total width is correct
            if sum(widths) < width:
                widths[0] += width - sum(widths)
            
            # Draw color bars
            x = 0
            for i, color in enumerate(result['colors']):
                # Get RGB color
                r, g, b = color['rgb']
                # Draw color bar (OpenCV uses BGR)
                palette[:, x:x+widths[i]] = (b, g, r)
                
                # Draw color name
                if widths[i] > 50:
                    text = f"{color['name']} ({color['weight']:.2f})"
                    cv2.putText(
                        palette, text, (x + 5, palette_height - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if sum(color['rgb'])/3 > 128 else (255, 255, 255), 
                        1, cv2.LINE_AA
                    )
                
                x += widths[i]
            
            # Stack images
            vis_img = np.vstack([vis_img, palette])
            
            # Draw overall result at top
            label = f"Vehicle color: {result['main_color']}"
            cv2.putText(
                vis_img, label, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA
            )
            
            # Draw box around license plate if provided
            if plate_box is not None:
                x1, y1, x2, y2 = plate_box
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    vis_img, "License plate", (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                )
        
        return vis_img

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create color detector
    detector = ColorDetector(
        n_clusters=5,
        verbose=True
    )
    
    # Load test image (replace with your own)
    img_path = "test_vehicle.jpg"
    img = cv2.imread(img_path)
    
    if img is not None:
        # Define license plate region (if known)
        plate_box = [100, 300, 300, 350]  # Example [x1, y1, x2, y2]
        
        # Detect vehicle color
        result = detector.detect_color(img, plate_box, return_details=True)
        
        # Print results
        print(f"Main vehicle color: {result['main_color']}")
        print("\nDetected colors:")
        for i, color in enumerate(result['colors']):
            print(f"  {i+1}. {color['name']} (RGB: {color['rgb']}, weight: {color['weight']:.2f})")
        
        # Create visualization
        vis_img = detector.visualize_colors(img, plate_box)
        
        # Display result
        cv2.imshow("Vehicle Color", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error: Could not load image from {img_path}")
