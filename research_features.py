import numpy as np
import cv2
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class ResearchBasedFeatureExtractor:
    """
    Feature extraction based on published research findings
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._setup_models()
        
    def _setup_models(self):
        """Setup research-validated models"""
        try:
            # Load CLIP for semantic understanding (research-backed)
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            print("✅ CLIP model loaded for research-based feature extraction")
        except Exception as e:
            print(f"⚠️ CLIP model failed to load: {e}")
            self.clip_model = None
    
    def extract_psydraw_features(self, image: np.ndarray, child_age: int) -> Dict:
        """
        Extract features based on PsyDraw research (Zhang et al., 2024)
        """
        features = {
            'visual_elements': self._extract_visual_elements(image),
            'spatial_organization': self._extract_spatial_features(image),
            'emotional_indicators': self._extract_emotional_features(image),
            'developmental_markers': self._extract_developmental_features(image, child_age),
            'semantic_content': self._extract_semantic_features(image)
        }
        
        return features
    
    def _extract_visual_elements(self, image: np.ndarray) -> Dict:
        """Extract visual elements following research protocols"""
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Color analysis (research-based)
        color_features = {
            'hue_distribution': self._analyze_hue_distribution(hsv),
            'saturation_analysis': self._analyze_saturation(hsv),
            'brightness_patterns': self._analyze_brightness_patterns(lab),
            'color_harmony': self._calculate_color_harmony(image)
        }
        
        # Shape analysis (following HTP research)
        shape_features = {
            'geometric_primitives': self._detect_geometric_primitives(image),
            'organic_shapes': self._detect_organic_shapes(image),
            'symmetry_analysis': self._analyze_symmetry(image),
            'proportion_analysis': self._analyze_proportions(image)
        }
        
        return {
            'color_features': color_features,
            'shape_features': shape_features
        }
    
    def _extract_spatial_features(self, image: np.ndarray) -> Dict:
        """Extract spatial organization features"""
        height, width = image.shape[:2]
        
        # Divide into research-standard regions
        regions = {
            'upper_third': image[:height//3, :],
            'middle_third': image[height//3:2*height//3, :],
            'lower_third': image[2*height//3:, :],
            'left_half': image[:, :width//2],
            'right_half': image[:, width//2:],
            'center_region': image[height//4:3*height//4, width//4:3*width//4]
        }
        
        spatial_features = {}
        for region_name, region in regions.items():
            spatial_features[region_name] = {
                'activity_level': self._calculate_activity_level(region),
                'complexity': self._calculate_region_complexity(region),
                'dominant_colors': self._get_dominant_colors(region)
            }
        
        # Calculate balance and composition
        spatial_features['composition_analysis'] = {
            'vertical_balance': self._calculate_vertical_balance(image),
            'horizontal_balance': self._calculate_horizontal_balance(image),
            'golden_ratio_compliance': self._check_golden_ratio(image),
            'rule_of_thirds_usage': self._analyze_rule_of_thirds(image)
        }
        
        return spatial_features
    
    def _extract_emotional_features(self, image: np.ndarray) -> Dict:
        """Extract emotion-related features based on research"""
        emotional_features = {
            'color_emotion_mapping': self._map_colors_to_emotions(image),
            'stroke_analysis': self._analyze_stroke_patterns(image),
            'pressure_indicators': self._estimate_pressure_patterns(image),
            'energy_level': self._calculate_drawing_energy(image)
        }
        
        return emotional_features
    
    def _extract_developmental_features(self, image: np.ndarray, child_age: int) -> Dict:
        """Extract developmental markers based on research"""
        developmental_features = {
            'fine_motor_indicators': self._assess_fine_motor_skills(image),
            'cognitive_complexity': self._assess_cognitive_complexity(image, child_age),
            'representational_level': self._assess_representational_level(image),
            'planning_indicators': self._assess_planning_skills(image)
        }
        
        return developmental_features
    
    def _extract_semantic_features(self, image: np.ndarray) -> Dict:
        """Extract semantic content using CLIP"""
        if self.clip_model is None:
            return {'error': 'CLIP model not available'}
        
        try:
            # Convert image for CLIP
            from PIL import Image
            pil_image = Image.fromarray(image)
            
            # Define research-based categories
            categories = [
                "a child's drawing of a person",
                "a child's drawing of a house",
                "a child's drawing of a tree",
                "a child's drawing of an animal",
                "a child's drawing of a family",
                "a child's drawing showing happiness",
                "a child's drawing showing sadness",
                "a child's drawing showing fear",
                "a child's drawing showing anger",
                "a child's drawing with detailed features",
                "a child's drawing with simple features",
                "a child's drawing with bright colors",
                "a child's drawing with dark colors"
            ]
            
            # Process with CLIP
            inputs = self.clip_processor(
                text=categories, 
                images=pil_image, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Create semantic feature vector
            semantic_features = {}
            for i, category in enumerate(categories):
                semantic_features[category.replace("a child's drawing ", "")] = float(probs[0][i])
            
            return semantic_features
            
        except Exception as e:
            return {'error': f'Semantic extraction failed: {e}'}
    
    # Helper methods for feature extraction
    def _analyze_hue_distribution(self, hsv_image: np.ndarray) -> Dict:
        """Analyze hue distribution patterns"""
        hue_channel = hsv_image[:, :, 0]
        
        # Calculate hue histogram
        hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Find dominant hues
        dominant_hues = np.argsort(hist)[-5:]  # Top 5 hues
        
        return {
            'hue_diversity': float(np.sum(hist > 0.01)),  # Number of significant hues
            'hue_entropy': float(-np.sum(hist * np.log(hist + 1e-10))),
            'dominant_hues': [int(h) for h in dominant_hues],
            'warm_cold_ratio': float(np.sum(hist[0:30]) + np.sum(hist[150:180])) / float(np.sum(hist[30:150]) + 1e-10)
        }
    
    def _analyze_saturation(self, hsv_image: np.ndarray) -> Dict:
        """Analyze saturation patterns"""
        sat_channel = hsv_image[:, :, 1]
        
        return {
            'average_saturation': float(np.mean(sat_channel)),
            'saturation_variance': float(np.var(sat_channel)),
            'high_saturation_ratio': float(np.sum(sat_channel > 200) / sat_channel.size),
            'low_saturation_ratio': float(np.sum(sat_channel < 50) / sat_channel.size)
        }
    
    def _analyze_brightness_patterns(self, lab_image: np.ndarray) -> Dict:
        """Analyze brightness patterns in LAB space"""
        l_channel = lab_image[:, :, 0]
        
        return {
            'average_brightness': float(np.mean(l_channel)),
            'brightness_contrast': float(np.std(l_channel)),
            'bright_regions_ratio': float(np.sum(l_channel > 200) / l_channel.size),
            'dark_regions_ratio': float(np.sum(l_channel < 50) / l_channel.size)
        }
    
    def _calculate_color_harmony(self, image: np.ndarray) -> float:
        """Calculate color harmony score"""
        # Convert to HSV for harmony analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hues = hsv[:, :, 0].flatten()
        
        # Remove black/white/gray pixels
        mask = hsv[:, :, 1] > 30  # Saturation threshold
        hues = hues[mask.flatten()]
        
        if len(hues) < 10:
            return 0.5  # Default for low-saturation images
        
        # Calculate hue differences
        unique_hues = np.unique(hues)
        if len(unique_hues) < 2:
            return 1.0  # Monochromatic = perfect harmony
        
        # Simple harmony measure based on hue clustering
        try:
            kmeans = KMeans(n_clusters=min(5, len(unique_hues)), random_state=42)
            kmeans.fit(unique_hues.reshape(-1, 1))
            
            # Harmony is higher when colors cluster well
            inertia = kmeans.inertia_
            harmony_score = 1.0 / (1.0 + inertia / 1000.0)  # Normalize
            
            return float(np.clip(harmony_score, 0, 1))
        except:
            return 0.5
    
    def _detect_geometric_primitives(self, image: np.ndarray) -> Dict:
        """Detect basic geometric shapes"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = {
            'circles': 0,
            'rectangles': 0,
            'triangles': 0,
            'lines': 0
        }
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip tiny shapes
                continue
            
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Classify shape
            if len(approx) == 3:
                shapes['triangles'] += 1
            elif len(approx) == 4:
                shapes['rectangles'] += 1
            elif len(approx) > 8:
                # Check if it's circular
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7:
                    shapes['circles'] += 1
            elif len(approx) == 2:
                shapes['lines'] += 1
        
        return shapes
    
    def _calculate_activity_level(self, region: np.ndarray) -> float:
        """Calculate activity level in a region"""
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate color variance
        color_variance = np.var(region.reshape(-1, 3), axis=0).mean()
        
        # Combine metrics
        activity = (edge_density * 0.7 + color_variance / 255.0 * 0.3)
        
        return float(activity)
    
    def _assess_fine_motor_skills(self, image: np.ndarray) -> Dict:
        """Assess fine motor skill indicators"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Analyze line quality
        edges = cv2.Canny(gray, 50, 150)
        
        # Find lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            line_qualities = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                line_qualities.append(length)
            
            avg_line_length = np.mean(line_qualities) if line_qualities else 0
            line_consistency = 1.0 - (np.std(line_qualities) / (np.mean(line_qualities) + 1e-10)) if line_qualities else 0
        else:
            avg_line_length = 0
            line_consistency = 0
        
        return {
            'line_control': float(np.clip(line_consistency, 0, 1)),
            'average_line_length': float(avg_line_length),
            'drawing_precision': float(np.clip(avg_line_length / 100.0, 0, 1))
        }
    
    def _assess_cognitive_complexity(self, image: np.ndarray, child_age: int) -> Dict:
        """Assess cognitive complexity indicators"""
        # Count distinct objects/regions
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use watershed segmentation to count objects
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_maxima
        from scipy import ndimage
        
        # Distance transform
        dist_transform = cv2.distanceTransform(gray, cv2.DIST_L2, 5)
        
        # Find peaks (object centers)
        peaks = peak_local_maxima(dist_transform, min_distance=20, threshold_abs=0.3)
        
        object_count = len(peaks[0]) if len(peaks) > 0 else 0
        
        # Expected complexity for age
        expected_objects = {
            2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9
        }
        
        expected = expected_objects.get(child_age, 5)
        complexity_ratio = object_count / expected
        
        return {
            'object_count': object_count,
            'expected_for_age': expected,
            'complexity_ratio': float(complexity_ratio),
            'cognitive_level': 'above_expected' if complexity_ratio > 1.2 else 'age_appropriate' if complexity_ratio > 0.8 else 'below_expected'
        }

