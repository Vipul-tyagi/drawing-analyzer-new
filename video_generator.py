import os
import torch
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import sys
import cv2
from scipy import ndimage

# Import your enhanced components
try:
    from ai_element_classifier import AIElementClassifier
    AI_CLASSIFIER_AVAILABLE = True
    print("✅ AI Element Classifier is ready!")
except ImportError:
    AI_CLASSIFIER_AVAILABLE = False
    print("❌ AI Element Classifier not available")

try:
    from smart_animator import SmartAnimator
    SMART_ANIMATOR_AVAILABLE = True
    print("✅ Smart Animator is ready!")
except ImportError:
    SMART_ANIMATOR_AVAILABLE = False
    print("❌ Smart Animator not available")

try:
    from sam_element_splitter import SAMElementSplitter
    SAM_SPLITTER_AVAILABLE = True
    print("✅ SAM Element Splitter is ready!")
except ImportError:
    SAM_SPLITTER_AVAILABLE = False
    print("❌ SAM Element Splitter not available")

# Enhanced imports for AI-powered features
try:
    import clip
    CLIP_AVAILABLE = True
    print("✅ CLIP is ready for AI classification!")
except ImportError:
    CLIP_AVAILABLE = False
    print("❌ CLIP not available. Install with: pip install clip-by-openai")

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
    print("✅ SAM is ready for advanced segmentation!")
except ImportError:
    SAM_AVAILABLE = False
    print("❌ SAM not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")

# Remove problematic imports and use alternatives
try:
    from skimage import measure, morphology
    from skimage.segmentation import watershed
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️ Scikit-image not available, using OpenCV alternatives")

# Setup FFmpeg automatically
def setup_ffmpeg():
    """Automatically configure FFmpeg for MoviePy"""
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg_path
        print(f"✅ FFmpeg configured automatically")
        return True
    except Exception as e:
        print(f"⚠️ FFmpeg auto-configuration failed: {e}")
        return False

# Run FFmpeg setup before importing MoviePy
setup_ffmpeg()

# Try to import video generation libraries
try:
    from moviepy.editor import *
    MOVIEPY_AVAILABLE = True
    print("✅ MoviePy is ready!")
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    print(f"❌ MoviePy not available: {e}")

# FIX: Add missing HuggingFace imports
try:
    from diffusers import LTXPipeline
    from diffusers.utils import export_to_video
    HF_AVAILABLE = True
    print("✅ HuggingFace Diffusers is ready!")
except ImportError as e:
    HF_AVAILABLE = False
    print(f"❌ HuggingFace not available: {e}")

# FIX: Add missing Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
    print("✅ Selenium is ready!")
except ImportError as e:
    SELENIUM_AVAILABLE = False
    print(f"❌ Selenium not available: {e}")

class ElementSplitter:
    """
    Fallback element splitter with traditional methods
    """
    
    def __init__(self):
        self.elements = []
    
    def split_drawing_elements(self, image_path):
        """Split a drawing into individual elements using traditional methods"""
        print("🎨 Analyzing drawing with traditional methods...")
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Could not load image: {image_path}")
                return []
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Method 1: Color-based segmentation
            color_elements = self._segment_by_color(img_rgb, hsv)
            
            # Method 2: Contour-based segmentation
            contour_elements = self._segment_by_contours(img_rgb, gray)
            
            # Method 3: Region-based segmentation
            region_elements = self._segment_by_regions(img_rgb, gray)
            
            # Combine and filter elements
            all_elements = color_elements + contour_elements + region_elements
            filtered_elements = self._filter_and_merge_elements(all_elements, img_rgb.shape)
            
            print(f"✅ Found {len(filtered_elements)} drawable elements")
            return filtered_elements
            
        except Exception as e:
            print(f"⚠️ Element splitting failed: {e}")
            return []
    
    def _segment_by_color(self, img_rgb, hsv):
        """Segment drawing by color regions"""
        elements = []
        
        try:
            # Define color ranges for common drawing colors
            color_ranges = {
                'red': ([0, 50, 50], [10, 255, 255]),
                'blue': ([100, 50, 50], [130, 255, 255]),
                'green': ([40, 50, 50], [80, 255, 255]),
                'yellow': ([20, 50, 50], [40, 255, 255]),
                'purple': ([130, 50, 50], [160, 255, 255]),
                'orange': ([10, 50, 50], [20, 255, 255])
            }
            
            for color_name, (lower, upper) in color_ranges.items():
                # Create mask for this color
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                
                # Clean up mask
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours in this color
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Filter small noise
                        # Create element mask
                        element_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(element_mask, [contour], 255)
                        
                        # Extract element
                        element_img = img_rgb.copy()
                        element_img[element_mask == 0] = [255, 255, 255]  # White background
                        
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        elements.append({
                            'type': f'{color_name}_region',
                            'image': element_img[y:y+h, x:x+w],
                            'mask': element_mask[y:y+h, x:x+w],
                            'bbox': (x, y, w, h),
                            'center': (x + w//2, y + h//2),
                            'area': area
                        })
        except Exception as e:
            print(f"⚠️ Color segmentation failed: {e}")
        
        return elements
    
    def _segment_by_contours(self, img_rgb, gray):
        """Segment drawing by contours/shapes"""
        elements = []
        
        try:
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to close gaps
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small elements
                    # Create element mask
                    element_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(element_mask, [contour], 255)
                    
                    # Extract element
                    element_img = img_rgb.copy()
                    element_img[element_mask == 0] = [255, 255, 255]  # White background
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Classify shape
                    shape_type = self._classify_shape(contour)
                    
                    elements.append({
                        'type': f'{shape_type}_shape',
                        'image': element_img[y:y+h, x:x+w],
                        'mask': element_mask[y:y+h, x:x+w],
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': area
                    })
        except Exception as e:
            print(f"⚠️ Contour segmentation failed: {e}")
        
        return elements
    
    def _segment_by_regions(self, img_rgb, gray):
        """Segment drawing by connected regions"""
        elements = []
        
        try:
            # Threshold to get binary image
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Remove noise
            kernel = np.ones((2,2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(binary)
            
            for label in range(1, num_labels):  # Skip background (label 0)
                # Create mask for this component
                component_mask = (labels == label).astype(np.uint8) * 255
                
                # Check size
                area = np.sum(component_mask > 0)
                if area > 800:  # Filter small components
                    # Extract element
                    element_img = img_rgb.copy()
                    element_img[component_mask == 0] = [255, 255, 255]  # White background
                    
                    # Get bounding box
                    coords = np.where(component_mask > 0)
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    
                    w, h = x_max - x_min, y_max - y_min
                    
                    elements.append({
                        'type': 'connected_region',
                        'image': element_img[y_min:y_max+1, x_min:x_max+1],
                        'mask': component_mask[y_min:y_max+1, x_min:x_max+1],
                        'bbox': (x_min, y_min, w, h),
                        'center': (x_min + w//2, y_min + h//2),
                        'area': area
                    })
        except Exception as e:
            print(f"⚠️ Region segmentation failed: {e}")
        
        return elements
    
    def _classify_shape(self, contour):
        """Classify the shape of a contour"""
        try:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Classify based on number of vertices
            vertices = len(approx)
            
            if vertices == 3:
                return "triangle"
            elif vertices == 4:
                # Check if it's a square or rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.8 <= aspect_ratio <= 1.2:
                    return "square"
                else:
                    return "rectangle"
            elif vertices > 8:
                # Check if it's circular
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:
                        return "circle"
            
            return "polygon"
        except Exception as e:
            return "unknown_shape"
    
    def _filter_and_merge_elements(self, elements, img_shape):
        """Filter out duplicate/overlapping elements and merge similar ones"""
        if not elements:
            return []
        
        try:
            # Sort by area (largest first)
            elements.sort(key=lambda x: x['area'], reverse=True)
            
            filtered = []
            for element in elements:
                # Check if this element significantly overlaps with existing ones
                is_duplicate = False
                for existing in filtered:
                    overlap = self._calculate_overlap(element['bbox'], existing['bbox'])
                    if overlap > 0.7:  # 70% overlap threshold
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered.append(element)
            
            # Limit to top 8 elements to avoid too much complexity
            return filtered[:8]
        except Exception as e:
            print(f"⚠️ Element filtering failed: {e}")
            return elements[:8] if elements else []
    
    def _calculate_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        try:
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            
            # Calculate intersection
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate union
            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - intersection_area
            
            return intersection_area / union_area if union_area > 0 else 0.0
        except Exception as e:
            return 0.0

class VideoGenerator:
    """
    Enhanced video generator with AI-powered component animation
    Integrates all the separate components you've created
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components based on availability
        if SAM_SPLITTER_AVAILABLE:
            self.element_splitter = SAMElementSplitter()
            print("✅ Using SAM Element Splitter")
        else:
            self.element_splitter = ElementSplitter()
            print("⚠️ Using traditional Element Splitter")
        
        if AI_CLASSIFIER_AVAILABLE:
            self.ai_classifier = AIElementClassifier()
            print("✅ Using AI Element Classifier")
        else:
            self.ai_classifier = None
            print("⚠️ AI Element Classifier not available")
        
        if SMART_ANIMATOR_AVAILABLE:
            self.smart_animator = SmartAnimator()
            print("✅ Using Smart Animator")
        else:
            self.smart_animator = None
            print("⚠️ Smart Animator not available")
        
        self.hf_pipeline = None
        print(f"🖥️ Using device: {self.device}")
        
        if MOVIEPY_AVAILABLE:
            self._test_moviepy()
    
    def _test_moviepy(self):
        """Test if MoviePy is working properly"""
        try:
            test_clip = ColorClip(size=(100, 100), color=(255, 0, 0), duration=1)
            print("✅ MoviePy basic functionality test passed")
            return True
        except Exception as e:
            print(f"⚠️ MoviePy test failed: {e}")
            return False
    
    def generate_memory_video(self, image_path, analysis_results, custom_text=None, 
                            animation_style='intelligent', user_story=None):
        """
        Main function with intelligent animation as default
        """
        print("🎬 Starting AI-powered animated video generation...")
        
        if not os.path.exists(image_path):
            return {'error': f'Image file not found: {image_path}'}
        
        available_methods = {
            'moviepy': MOVIEPY_AVAILABLE,
            'huggingface': HF_AVAILABLE and torch.cuda.is_available(),
            'selenium': SELENIUM_AVAILABLE,
            'ai_classifier': AI_CLASSIFIER_AVAILABLE,
            'smart_animator': SMART_ANIMATOR_AVAILABLE,
            'sam_splitter': SAM_SPLITTER_AVAILABLE
        }
        
        print(f"📊 Available methods: {available_methods}")
        
        # Try intelligent animation first (NEW - Main feature)
        if animation_style == 'intelligent' and MOVIEPY_AVAILABLE and self.ai_classifier and self.smart_animator:
            try:
                print("🤖 Trying AI-powered intelligent component animation...")
                result = self.create_intelligent_animation_video(image_path, analysis_results, custom_text, user_story)
                if result and 'video_path' in result:
                    return result
            except Exception as e:
                print(f"⚠️ Intelligent animation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Try element-based animation (your existing method)
        if animation_style == 'elements' and MOVIEPY_AVAILABLE:
            try:
                print("🎨 Trying element-based animation...")
                result = self.create_element_animation_video(image_path, analysis_results, custom_text)
                if result and 'video_path' in result:
                    return result
            except Exception as e:
                print(f"⚠️ Element animation failed: {e}")
        
        # Keep all your existing fallback methods
        if animation_style == 'particle' and MOVIEPY_AVAILABLE:
            try:
                print("✨ Trying particle animation...")
                result = self.create_particle_animation_video(image_path, analysis_results, custom_text)
                if result and 'video_path' in result:
                    return result
            except Exception as e:
                print(f"⚠️ Particle animation failed: {e}")
        
        if animation_style == 'floating' and MOVIEPY_AVAILABLE:
            try:
                print("🌊 Trying floating animation...")
                result = self.create_floating_animation_video(image_path, analysis_results, custom_text)
                if result and 'video_path' in result:
                    return result
            except Exception as e:
                print(f"⚠️ Floating animation failed: {e}")
        
        # Standard fallback
        if MOVIEPY_AVAILABLE:
            try:
                print("🎬 Trying standard animated video...")
                result = self.create_moviepy_story_video(image_path, analysis_results, custom_text)
                if result and 'video_path' in result:
                    return result
            except Exception as e:
                print(f"⚠️ Standard animation failed: {e}")
        
        # Final fallback
        try:
            result = self.create_simple_slideshow(image_path, analysis_results, custom_text)
            if result and 'video_path' in result:
                return result
        except Exception as e:
            print(f"⚠️ Simple slideshow failed: {e}")
        
        return {
            'error': 'All video generation methods failed',
            'available_methods': available_methods,
            'suggestions': [
                'Install moviepy: pip install moviepy imageio imageio-ffmpeg',
                'Install CLIP: pip install clip-by-openai',
                'Install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git',
                'Install ffmpeg system-wide',
                'Check if image file exists and is readable',
                'Make sure ai_element_classifier.py, smart_animator.py, and sam_element_splitter.py are in the same directory'
            ]
        }
    
    def create_intelligent_animation_video(self, image_path, analysis_results, custom_text=None, user_story=None):
        """
        NEW METHOD: Create video with AI-powered component animation
        Uses your separate AI classifier and smart animator files
        """
        if not MOVIEPY_AVAILABLE:
            return {'error': 'MoviePy not available'}
        
        print("🤖 Creating AI-powered intelligent component animation video...")
        try:
            duration = 15
            
            # Step 1: Split the drawing into elements using SAM or traditional methods
            print("🔍 Segmenting drawing elements...")
            elements = self.element_splitter.split_drawing_elements(image_path)
            
            if not elements:
                print("⚠️ No elements found, falling back to whole image animation")
                return self.create_element_animation_video(image_path, analysis_results, custom_text)
            
            # Step 2: Classify each element using AI
            print("🏷️ Classifying elements with AI...")
            classified_elements = []
            
            for i, element in enumerate(elements):
                if self.ai_classifier:
                    # Use your AI classifier
                    classification = self.ai_classifier.classify_element(element['image'], element)
                else:
                    # Fallback classification
                    classification = {
                        'label': 'unknown',
                        'confidence': 0.5,
                        'animation_type': 'float',
                        'animation_speed': 'medium',
                        'animation_pattern': 'gentle_motion',
                        'layer': 'foreground'
                    }
                
                classified_elements.append({
                    **element,
                    'classification': classification,
                    'index': i
                })
                print(f"🏷️ Element {i}: {classification['label']} "
                      f"(confidence: {classification['confidence']:.2f})")
            
            # Step 3: Sort elements by layer for proper rendering order
            layer_order = {'background': 0, 'midground': 1, 'foreground': 2}
            classified_elements.sort(key=lambda x: layer_order.get(x['classification']['layer'], 1))
            
            # Step 4: Create background
            img = Image.open(image_path)
            background = Image.new('RGB', img.size, 'white')
            background_clip = ImageClip(np.array(background)).set_duration(duration).resize(height=720)
            
            # Step 5: Create intelligent animations for each classified element
            animated_clips = []
            
            for element in classified_elements:
                try:
                    # Save element as temporary file
                    element_pil = Image.fromarray(element['image'])
                    temp_path = f"temp_smart_element_{element['index']}.png"
                    element_pil.save(temp_path)
                    
                    # Create base clip
                    element_clip = ImageClip(temp_path).set_duration(duration)
                    
                    # Apply intelligent animation using your smart animator
                    if self.smart_animator:
                        animated_clip = self.smart_animator.create_object_animation(
                            element_clip, 
                            element, 
                            element['classification'],
                            user_story
                        )
                    else:
                        # Fallback to simple animation
                        animated_clip = self._create_simple_element_animation(element_clip, element, duration)
                    
                    if animated_clip:
                        animated_clips.append(animated_clip)
                        print(f"✅ Animated {element['classification']['label']} successfully")
                    
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    print(f"⚠️ Failed to animate element {element['index']}: {e}")
                    continue
            
            if not animated_clips:
                print("⚠️ No animated clips created, falling back")
                return self.create_element_animation_video(image_path, analysis_results, custom_text)
            
            # Step 6: Combine all clips in proper layer order
            all_clips = [background_clip] + animated_clips
            final_clip = CompositeVideoClip(all_clips)
            
            # Step 7: Save video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"intelligent_animation_{timestamp}.mp4"
            
            print("🎬 Rendering AI-powered animation video...")
            final_clip.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                bitrate='3000k',
                verbose=False,
                logger=None
            )
            
            story = user_story or custom_text or self._create_story_narrative(analysis_results, custom_text)
            
            return {
                'video_path': output_path,
                'narrative_used': story,
                'generation_method': 'ai_powered_intelligent_animation',
                'cost': 0.0,
                'duration_seconds': duration,
                'elements_found': len(classified_elements),
                'classifications': [elem['classification']['label'] for elem in classified_elements],
                'layers_used': list(set([elem['classification']['layer'] for elem in classified_elements]))
            }
            
        except Exception as e:
            print(f"❌ Failed to create intelligent animation: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Intelligent animation failed: {str(e)}'}
    
    def _create_simple_element_animation(self, element_clip, element_info, duration):
        """Simple fallback animation if smart animator is not available"""
        try:
            x, y, w, h = element_info['bbox']
            
            def simple_float(t):
                float_x = x + 10 * np.sin(2 * np.pi * t / 6)
                float_y = y + 5 * np.cos(2 * np.pi * t / 8)
                return (int(float_x), int(float_y))
            
            return element_clip.set_position(simple_float)
        except Exception as e:
            return element_clip.set_position((x, y))
    
    # Keep ALL your existing methods exactly as they are
    def create_element_animation_video(self, image_path, analysis_results, custom_text=None):
        """Create animation where individual drawing elements move separately"""
        if not MOVIEPY_AVAILABLE:
            return {'error': 'MoviePy not available'}

        print("🎨 Creating element-based animation video...")
        try:
            duration = 15

            # Step 1: Split the drawing into elements
            elements = self.element_splitter.split_drawing_elements(image_path)

            if not elements:
                print("⚠️ No elements found, falling back to whole image animation")
                return self.create_moviepy_story_video(image_path, analysis_results, custom_text)

            # Step 2: Create background (white or original image faded)
            img = Image.open(image_path)
            background = Image.new('RGB', img.size, 'white')
            background_clip = ImageClip(np.array(background)).set_duration(duration).resize(height=720)

            # Step 3: Create animated clips for each element
            element_clips = []

            for i, element in enumerate(elements):
                try:
                    # Convert element image to PIL
                    element_pil = Image.fromarray(element['image'])

                    # Save element as temporary file
                    temp_path = f"temp_element_{i}.png"
                    element_pil.save(temp_path)

                    # Create animation for this element
                    element_clip = self._create_element_animation(
                        temp_path, 
                        element, 
                        duration, 
                        i
                    )

                    if element_clip:
                        element_clips.append(element_clip)

                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                except Exception as e:
                    print(f"⚠️ Failed to process element {i}: {e}")
                    continue

            if not element_clips:
                print("⚠️ No element clips created, falling back")
                return self.create_moviepy_story_video(image_path, analysis_results, custom_text)

            # Step 4: Combine all clips
            all_clips = [background_clip] + element_clips
            final_clip = CompositeVideoClip(all_clips)

            # Step 5: Save video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"element_animation_{timestamp}.mp4"

            print("🎬 Rendering element-based animation...")
            final_clip.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                bitrate='3000k',
                verbose=False,
                logger=None
            )

            story = self._create_story_narrative(analysis_results, custom_text)

            return {
                'video_path': output_path,
                'narrative_used': story,
                'generation_method': 'element_based_animation',
                'cost': 0.0,
                'duration_seconds': duration,
                'elements_found': len(elements)
            }

        except Exception as e:
            print(f"❌ Failed to create element animation: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Element animation failed: {str(e)}'}

    def _create_element_animation(self, element_path, element_info, duration, index):
        """Create animation for a single element"""
        try:
            # Get element properties
            x, y, w, h = element_info['bbox']
            center_x, center_y = element_info['center']
            element_type = element_info['type']

            # Scale positions for 720p video
            scale_factor = 720 / max(h, 400)  # Better scaling
            scaled_center_x = int(center_x * scale_factor)
            scaled_center_y = int(center_y * scale_factor)

            # Create base clip
            element_clip = ImageClip(element_path).set_duration(duration)

            # Define different animation patterns based on index
            animation_patterns = [
                'spiral_in',
                'bounce_in',
                'slide_from_left',
                'slide_from_right',
                'fade_and_grow',
                'rotate_in'
            ]

            pattern = animation_patterns[index % len(animation_patterns)]

            # Apply animation pattern
            if pattern == 'spiral_in':
                def pos_func(t):
                    if t < 3:  # First 3 seconds: spiral in
                        progress = t / 3
                        angle = 4 * np.pi * (1 - progress)
                        radius = 300 * (1 - progress)
                        x = scaled_center_x + radius * np.cos(angle)
                        y = scaled_center_y + radius * np.sin(angle)
                    else:  # Rest: gentle floating
                        float_t = t - 3
                        x = scaled_center_x + 20 * np.sin(2 * np.pi * float_t / 4)
                        y = scaled_center_y + 10 * np.cos(2 * np.pi * float_t / 3)
                    return (max(0, int(x)), max(0, int(y)))

                element_clip = element_clip.set_position(pos_func)

            elif pattern == 'bounce_in':
                def pos_func(t):
                    if t < 2:  # First 2 seconds: bounce in from top
                        progress = t / 2
                        bounce = abs(np.sin(np.pi * progress * 3)) * (1 - progress)
                        x = scaled_center_x
                        y = -100 + (scaled_center_y + 100) * progress + bounce * 100
                    else:  # Rest: slight bounce
                        bounce_t = t - 2
                        x = scaled_center_x
                        y = scaled_center_y + 15 * abs(np.sin(2 * np.pi * bounce_t / 2))
                    return (max(0, int(x)), max(0, int(y)))

                element_clip = element_clip.set_position(pos_func)

            elif pattern == 'slide_from_left':
                def pos_func(t):
                    if t < 2.5:  # First 2.5 seconds: slide in
                        progress = t / 2.5
                        eased_progress = 1 - (1 - progress) ** 3
                        x = -200 + (scaled_center_x + 200) * eased_progress
                        y = scaled_center_y
                    else:  # Rest: gentle sway
                        sway_t = t - 2.5
                        x = scaled_center_x + 25 * np.sin(2 * np.pi * sway_t / 5)
                        y = scaled_center_y
                    return (max(0, int(x)), max(0, int(y)))

                element_clip = element_clip.set_position(pos_func)

            elif pattern == 'slide_from_right':
                def pos_func(t):
                    if t < 2.5:  # First 2.5 seconds: slide in
                        progress = t / 2.5
                        eased_progress = 1 - (1 - progress) ** 3
                        x = 1280 + 200 - (1280 + 200 - scaled_center_x) * eased_progress
                        y = scaled_center_y
                    else:  # Rest: gentle sway
                        sway_t = t - 2.5
                        x = scaled_center_x + 25 * np.sin(2 * np.pi * sway_t / 5)
                        y = scaled_center_y
                    return (max(0, int(x)), max(0, int(y)))

                element_clip = element_clip.set_position(pos_func)

            elif pattern == 'fade_and_grow':
                def resize_func(t):
                    if t < 3:  # First 3 seconds: grow
                        progress = t / 3
                        return 0.1 + 0.9 * (1 - (1 - progress) ** 2)
                    else:  # Rest: breathing
                        breath_t = t - 3
                        return 1.0 + 0.1 * np.sin(2 * np.pi * breath_t / 4)

                element_clip = (element_clip
                               .set_position((scaled_center_x, scaled_center_y))
                               .resize(resize_func)
                               .fadein(3))

            elif pattern == 'rotate_in':
                def rotate_func(t):
                    if t < 3:  # First 3 seconds: rotate in
                        return 360 * (t / 3) ** 2
                    else:  # Rest: gentle rotation
                        return 360 + 30 * np.sin(2 * np.pi * (t - 3) / 6)

                element_clip = (element_clip
                               .set_position((scaled_center_x, scaled_center_y))
                               .rotate(rotate_func))

            else:  # Default: simple position
                element_clip = element_clip.set_position((scaled_center_x, scaled_center_y))

            # Add entrance delay based on index
            start_delay = index * 0.5  # Stagger entrances
            element_clip = element_clip.set_start(start_delay)

            return element_clip

        except Exception as e:
            print(f"⚠️ Failed to animate element {index}: {e}")
            return None

    def create_particle_animation_video(self, image_path, analysis_results, custom_text=None):
        """FIXED Particle animation without text issues"""
        print("✨ Creating WORKING particle animation...")
        try:
            duration = 12

            # Background
            background_clip = ImageClip(image_path).set_duration(duration).resize(height=720)

            # Create particle clips with WORKING animation
            particle_clips = []
            num_particles = 8

            for i in range(num_particles):
                angle_offset = 2 * np.pi * i / num_particles

                def make_particle_pos_func(offset):
                    def pos_func(t):
                        # Spiral out for first half, spiral in for second half
                        progress = t / duration
                        if progress < 0.5:
                            # Spiral outward
                            radius = 300 * (progress * 2)
                        else:
                            # Spiral inward
                            radius = 300 * (2 - progress * 2)

                        angle = offset + 2 * np.pi * t / 3
                        x = 640 + radius * np.cos(angle)
                        y = 360 + radius * np.sin(angle)
                        return (int(x), int(y))
                    return pos_func

                particle = (ImageClip(image_path)
                           .set_duration(duration)
                           .resize(height=120)
                           .set_position(make_particle_pos_func(angle_offset))
                           .set_opacity(0.6))

                particle_clips.append(particle)

            # Main image appears in center
            main_clip = (ImageClip(image_path)
                        .set_duration(duration)
                        .resize(height=720)
                        .set_position('center')
                        .fadein(2)
                        .set_start(6))

            # Combine all clips WITHOUT ANY TEXT
            all_clips = [background_clip] + particle_clips + [main_clip]
            final_clip = CompositeVideoClip(all_clips)

            # Save video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pure_particle_animation_{timestamp}.mp4"

            final_clip.write_videofile(output_path, fps=24, codec='libx264', bitrate='3000k')

            story = self._create_story_narrative(analysis_results, custom_text)

            return {
                'video_path': output_path,
                'narrative_used': story,
                'generation_method': 'pure_particle_animation_no_text',
                'cost': 0.0,
                'duration_seconds': duration
            }

        except Exception as e:
            print(f"❌ Particle animation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Particle animation failed: {str(e)}'}

    def create_floating_animation_video(self, image_path, analysis_results, custom_text=None):
        """FIXED Floating animation without text issues"""
        print("🌊 Creating WORKING floating animation...")
        try:
            duration = 12

            # Background
            background_clip = ImageClip(image_path).set_duration(duration).resize(height=720)

            # Main floating image
            def float_pos_func(t):
                x = 640 + 40 * np.sin(2 * np.pi * t / 5)
                y = 360 + 60 * np.sin(2 * np.pi * t / 3)
                return (int(x), int(y))

            float_clip = (ImageClip(image_path)
                         .set_duration(duration)
                         .resize(height=600)
                         .set_position(float_pos_func)
                         .set_opacity(0.8))

            # Orbiting smaller copies
            orbit_clips = []
            for i in range(4):
                angle_offset = i * np.pi / 2  # 90 degrees apart

                def make_orbit_func(offset):
                    def pos_func(t):
                        angle = 2 * np.pi * t / 8 + offset
                        x = 640 + 180 * np.cos(angle)
                        y = 360 + 180 * np.sin(angle)
                        return (int(x), int(y))
                    return pos_func

                orbit_clip = (ImageClip(image_path)
                             .set_duration(duration)
                             .resize(height=200)
                             .set_position(make_orbit_func(angle_offset))
                             .set_opacity(0.4))

                orbit_clips.append(orbit_clip)

            # Combine clips WITHOUT ANY TEXT
            all_clips = [background_clip] + orbit_clips + [float_clip]
            final_clip = CompositeVideoClip(all_clips)

            # Save video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pure_floating_animation_{timestamp}.mp4"

            final_clip.write_videofile(output_path, fps=24, codec='libx264', bitrate='3000k')

            story = self._create_story_narrative(analysis_results, custom_text)

            return {
                'video_path': output_path,
                'narrative_used': story,
                'generation_method': 'pure_floating_animation_no_text',
                'cost': 0.0,
                'duration_seconds': duration
            }

        except Exception as e:
            print(f"❌ Floating animation failed: {e}")
            return {'error': f'Floating animation failed: {str(e)}'}

    def create_moviepy_story_video(self, image_path, analysis_results, custom_text=None):
        """Fallback to standard animation if element splitting fails"""
        if not MOVIEPY_AVAILABLE:
            return {'error': 'MoviePy not available'}

        print("🎬 Creating standard animated video...")
        try:
            duration = 10
            background_clip = ImageClip(image_path).set_duration(duration).resize(height=720)

            # Simple circular animation as fallback
            def pos_func(t):
                center_x, center_y = 640, 360
                radius = 120
                x = center_x + radius * np.cos(2 * np.pi * t / 4)
                y = center_y + radius * np.sin(2 * np.pi * t / 4)
                return (int(x), int(y))

            moving_clip = (ImageClip(image_path)
                          .set_duration(duration)
                          .resize(height=400)
                          .set_position(pos_func)
                          .set_opacity(0.7))

            final_clip = CompositeVideoClip([background_clip, moving_clip])

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"standard_animated_video_{timestamp}.mp4"

            final_clip.write_videofile(output_path, fps=24, codec='libx264', verbose=False, logger=None)

            story = self._create_story_narrative(analysis_results, custom_text)

            return {
                'video_path': output_path,
                'narrative_used': story,
                'generation_method': 'standard_animation_fallback',
                'cost': 0.0,
                'duration_seconds': duration
            }

        except Exception as e:
            return {'error': f'Standard animation failed: {str(e)}'}

    def create_simple_slideshow(self, image_path, analysis_results, custom_text=None):
        """
        Ultra-simple fallback: Create video using OpenCV if MoviePy fails
        """
        try:
            import cv2
            from PIL import Image

            print("🔄 Creating simple slideshow with OpenCV...")

            img = Image.open(image_path)
            img = img.resize((1280, 720))
            img_array = np.array(img)

            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"simple_video_{timestamp}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 1.0, (1280, 720))

            for i in range(8):
                out.write(img_array)

            out.release()

            story = self._create_story_narrative(analysis_results, custom_text)

            return {
                'video_path': output_path,
                'narrative_used': story,
                'generation_method': 'simple_slideshow_opencv',
                'cost': 0.0,
                'duration_seconds': 8
            }

        except Exception as e:
            return {'error': f'Simple slideshow failed: {str(e)}'}

    def _create_story_narrative(self, analysis_results, custom_text=None):
        """Create a nice story about the drawing"""
        if custom_text:
            return custom_text

        try:
            child_age = analysis_results['input_info']['child_age']
            drawing_context = analysis_results['input_info']['drawing_context']
            blip_description = analysis_results['traditional_analysis']['blip_description']

            story = f"Watch as this amazing {child_age}-year-old's drawing comes to life! Each element moves and dances, showing the magic of {blip_description}. Every piece tells its own story before coming together as one beautiful creation!"

            return story

        except Exception as e:
            return "Watch this wonderful drawing come to life as each element moves and dances!"

# Test function
def test_video_generator():
    """Test the video generator with element splitting"""
    print("🧪 Testing Enhanced Video Generator...")
    print("=" * 50)

    print("📊 DEPENDENCY STATUS:")
    print(f"MoviePy: {'✅' if MOVIEPY_AVAILABLE else '❌'}")
    print(f"CLIP: {'✅' if CLIP_AVAILABLE else '❌'}")
    print(f"SAM: {'✅' if SAM_AVAILABLE else '❌'}")
    print(f"AI Classifier: {'✅' if AI_CLASSIFIER_AVAILABLE else '❌'}")
    print(f"Smart Animator: {'✅' if SMART_ANIMATOR_AVAILABLE else '❌'}")
    print(f"SAM Splitter: {'✅' if SAM_SPLITTER_AVAILABLE else '❌'}")
    print(f"GPU Available: {'✅' if torch.cuda.is_available() else '❌'}")
    print(f"Selenium: {'✅' if SELENIUM_AVAILABLE else '❌'}")

    print(f"\n🖥️ SYSTEM INFO:")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")

    print(f"\n💡 RECOMMENDATIONS:")
    if AI_CLASSIFIER_AVAILABLE and SMART_ANIMATOR_AVAILABLE and MOVIEPY_AVAILABLE:
        print("✅ Full AI-powered intelligent animation should work!")
    elif MOVIEPY_AVAILABLE:
        print("⚠️ Basic element animation available")
    else:
        print("❌ Install MoviePy: pip install moviepy imageio imageio-ffmpeg")

    print("=" * 50)

if __name__ == "__main__":
    test_video_generator()
