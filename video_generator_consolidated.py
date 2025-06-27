import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip, TextClip, CompositeVideoClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import tempfile

class ConsolidatedVideoGenerator:
    """
    Consolidated video generator with scientific analysis integration
    """
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.video_styles = {
            'analysis_walkthrough': self._create_analysis_walkthrough,
            'character_animation': self._create_character_animation,
            'drawing_evolution': self._create_drawing_evolution,
            'story_animation': self._create_story_animation,
            'scientific_presentation': self._create_scientific_presentation
        }
        
    def create_comprehensive_video(self, image_path: str, analysis_results: Dict, options: Dict) -> str:
        """Create comprehensive video with analysis integration"""
        
        try:
            style = options.get('style', 'analysis_walkthrough')
            duration = options.get('duration', 15)
            
            print(f"🎬 Creating video with style: {style}")
            
            if style in self.video_styles:
                video_path = self.video_styles[style](image_path, analysis_results, options)
            else:
                video_path = self._create_default_video(image_path, analysis_results, options)
            
            if video_path and os.path.exists(video_path):
                print(f"✅ Video created successfully: {video_path}")
                return video_path
            else:
                print("❌ Video creation failed")
                return None
                
        except Exception as e:
            print(f"❌ Video generation error: {str(e)}")
            return None
    
    def _create_analysis_walkthrough(self, image_path: str, results: Dict, options: Dict) -> str:
        """Create video showing analysis process step by step"""
        
        duration = options.get('duration', 15)
        base_image = cv2.imread(image_path)
        
        if base_image is None:
            return None
        
        frames = []
        fps = 30
        
        # Scene 1: Original drawing (3 seconds)
        scene1_frames = self._create_text_overlay_scene(
            base_image, "Original Drawing", 3, fps
        )
        frames.extend(scene1_frames)
        
        # Scene 2: AI Description (4 seconds)
        blip_desc = results.get('traditional_analysis', {}).get('blip_description', 'AI Analysis')
        scene2_frames = self._create_text_overlay_scene(
            base_image, f"AI Sees: {blip_desc[:50]}...", 4, fps
        )
        frames.extend(scene2_frames)
        
        # Scene 3: Color Analysis (3 seconds)
        color_info = results.get('traditional_analysis', {}).get('color_analysis', {})
        color_text = f"Colors: {color_info.get('dominant_color', 'Mixed')}"
        scene3_frames = self._create_highlighted_analysis_scene(
            base_image, color_text, 3, fps, highlight_type='color'
        )
        frames.extend(scene3_frames)
        
        # Scene 4: Developmental Assessment (3 seconds)
        dev_info = results.get('traditional_analysis', {}).get('developmental_assessment', {})
        dev_text = f"Development: {dev_info.get('level', 'unknown').replace('_', ' ').title()}"
        scene4_frames = self._create_text_overlay_scene(
            base_image, dev_text, 3, fps
        )
        frames.extend(scene4_frames)
        
        # Scene 5: Emotional Indicators (2 seconds)
        emotional_info = results.get('traditional_analysis', {}).get('emotional_indicators', {})
        emotion_text = f"Mood: {emotional_info.get('overall_mood', 'neutral').title()}"
        scene5_frames = self._create_text_overlay_scene(
            base_image, emotion_text, 2, fps
        )
        frames.extend(scene5_frames)
        
        return self._create_video_from_frames(frames, "analysis_walkthrough.mp4", fps)
    
    def _create_character_animation(self, image_path: str, results: Dict, options: Dict) -> str:
        """Create simple character animation"""
        
        base_image = cv2.imread(image_path)
        if base_image is None:
            return None
        
        frames = []
        fps = 30
        duration = options.get('duration', 10)
        total_frames = duration * fps
        
        # Create breathing/pulsing animation
        for i in range(total_frames):
            # Create pulsing effect
            scale = 1.0 + 0.05 * np.sin(i * 0.1)
            frame = self._apply_scale_effect(base_image, scale)
            
            # Add sparkle effects every 30 frames
            if i % 30 == 0:
                frame = self._add_sparkle_effects(frame)
            
            frames.append(frame)
        
        return self._create_video_from_frames(frames, "character_animation.mp4", fps)
    
    def _create_drawing_evolution(self, image_path: str, results: Dict, options: Dict) -> str:
        """Create drawing evolution animation"""
        
        base_image = cv2.imread(image_path)
        if base_image is None:
            return None
        
        frames = []
        fps = 30
        duration = options.get('duration', 12)
        total_frames = duration * fps
        
        # Create evolution from simple to complex
        for i in range(total_frames):
            progress = i / total_frames
            
            # Gradually reveal the drawing
            frame = self._create_progressive_reveal(base_image, progress)
            
            # Add development stage text
            if i % (fps * 2) == 0:  # Every 2 seconds
                stage = self._get_development_stage(progress)
                frame = self._add_text_to_frame(frame, stage, position='bottom')
            
            frames.append(frame)
        
        return self._create_video_from_frames(frames, "drawing_evolution.mp4", fps)
    
    def _create_story_animation(self, image_path: str, results: Dict, options: Dict) -> str:
        """Create story-based animation"""
        
        base_image = cv2.imread(image_path)
        if base_image is None:
            return None
        
        frames = []
        fps = 30
        duration = options.get('duration', 15)
        
        # Create story scenes based on analysis
        story_elements = self._extract_story_elements(results)
        
        scene_duration = duration // len(story_elements) if story_elements else duration
        
        for element in story_elements:
            scene_frames = self._create_story_scene(base_image, element, scene_duration, fps)
            frames.extend(scene_frames)
        
        return self._create_video_from_frames(frames, "story_animation.mp4", fps)
    
    def _create_scientific_presentation(self, image_path: str, results: Dict, options: Dict) -> str:
        """Create scientific presentation style video"""
        
        base_image = cv2.imread(image_path)
        if base_image is None:
            return None
        
        frames = []
        fps = 30
        duration = options.get('duration', 20)
        
        # Title slide (2 seconds)
        title_frames = self._create_title_slide("Scientific Analysis Report", 2, fps, base_image.shape[:2])
        frames.extend(title_frames)
        
        # Drawing presentation (3 seconds)
        drawing_frames = self._create_scientific_slide(
            base_image, "Subject Drawing", "Child's artistic expression", 3, fps
        )
        frames.extend(drawing_frames)
        
        # Analysis results slides
        analysis_sections = [
            ("Developmental Assessment", self._format_developmental_data(results)),
            ("Color Analysis", self._format_color_data(results)),
            ("Emotional Indicators", self._format_emotional_data(results)),
            ("Statistical Validation", self._format_validation_data(results))
        ]
        
        section_duration = (duration - 5) // len(analysis_sections)
        
        for title, data in analysis_sections:
            section_frames = self._create_data_slide(title, data, section_duration, fps, base_image.shape[:2])
            frames.extend(section_frames)
        
        return self._create_video_from_frames(frames, "scientific_presentation.mp4", fps)
    
    def _create_default_video(self, image_path: str, results: Dict, options: Dict) -> str:
        """Create default video when style not recognized"""
        return self._create_analysis_walkthrough(image_path, results, options)
    
    # Helper methods for video creation
    
    def _create_text_overlay_scene(self, base_image: np.ndarray, text: str, duration: int, fps: int) -> List[np.ndarray]:
        """Create scene with text overlay"""
        frames = []
        total_frames = duration * fps
        
        for i in range(total_frames):
            frame = base_image.copy()
            frame = self._add_text_to_frame(frame, text, position='bottom')
            frames.append(frame)
        
        return frames
    
    def _create_highlighted_analysis_scene(self, base_image: np.ndarray, text: str, duration: int, fps: int, highlight_type: str) -> List[np.ndarray]:
        """Create scene with highlighted analysis"""
        frames = []
        total_frames = duration * fps
        
        for i in range(total_frames):
            frame = base_image.copy()
            
            # Add highlighting based on type
            if highlight_type == 'color':
                frame = self._add_color_highlighting(frame)
            elif highlight_type == 'shapes':
                frame = self._add_shape_highlighting(frame)
            
            frame = self._add_text_to_frame(frame, text, position='bottom')
            frames.append(frame)
        
        return frames
    
    def _add_text_to_frame(self, frame: np.ndarray, text: str, position: str = 'bottom') -> np.ndarray:
        """Add text overlay to frame"""
        # Convert to PIL for text rendering
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        if position == 'bottom':
            x = (pil_image.width - text_width) // 2
            y = pil_image.height - text_height - 30
        elif position == 'top':
            x = (pil_image.width - text_width) // 2
            y = 30
        else:  # center
            x = (pil_image.width - text_width) // 2
            y = (pil_image.height - text_height) // 2
        
        # Draw background rectangle
        draw.rectangle([x-10, y-5, x+text_width+10, y+text_height+5], fill=(0, 0, 0, 180))
        
        # Draw text
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _apply_scale_effect(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Apply scaling effect to image"""
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create canvas and center the scaled image
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        y_offset = max(0, (height - new_height) // 2)
        x_offset = max(0, (width - new_width) // 2)
        
        # Calculate crop if image is larger than canvas
        if new_height > height or new_width > width:
            crop_y = max(0, (new_height - height) // 2)
            crop_x = max(0, (new_width - width) // 2)
            resized = resized[crop_y:crop_y+height, crop_x:crop_x+width]
            canvas = resized
        else:
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas
    
    def _add_sparkle_effects(self, image: np.ndarray) -> np.ndarray:
        """Add sparkle effects to image"""
        result = image.copy()
        height, width = result.shape[:2]
        
        # Add random sparkles
        for _ in range(10):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            cv2.circle(result, (x, y), 3, (255, 255, 255), -1)
        
        return result
    
    def _create_progressive_reveal(self, image: np.ndarray, progress: float) -> np.ndarray:
        """Create progressive reveal effect"""
        height, width = image.shape[:2]
        
        # Create mask based on progress
        mask = np.zeros((height, width), dtype=np.uint8)
        reveal_height = int(height * progress)
        mask[:reveal_height, :] = 255
        
        # Apply mask
        result = np.ones_like(image) * 255  # White background
        result[mask > 0] = image[mask > 0]
        
        return result
    
    def _get_development_stage(self, progress: float) -> str:
        """Get development stage based on progress"""
        if progress < 0.25:
            return "Scribbling Stage"
        elif progress < 0.5:
            return "Pre-Schematic Stage"
        elif progress < 0.75:
            return "Schematic Stage"
        else:
            return "Realistic Stage"
    
    def _extract_story_elements(self, results: Dict) -> List[str]:
        """Extract story elements from analysis results"""
        elements = []
        
        # Extract from AI description
        description = results.get('traditional_analysis', {}).get('blip_description', '')
        if 'person' in description.lower():
            elements.append("Character Introduction")
        if 'house' in description.lower():
            elements.append("Setting: Home")
        if 'tree' in description.lower():
            elements.append("Natural Environment")
        
        # Add emotional context
        mood = results.get('traditional_analysis', {}).get('emotional_indicators', {}).get('overall_mood', 'neutral')
        if mood != 'neutral':
            elements.append(f"Emotional Context: {mood.title()}")
        
        return elements if elements else ["Drawing Analysis", "Visual Elements", "Interpretation"]
    
    def _create_story_scene(self, base_image: np.ndarray, element: str, duration: int, fps: int) -> List[np.ndarray]:
        """Create frames for a story scene"""
        frames = []
        total_frames = duration * fps
        
        for i in range(total_frames):
            frame = base_image.copy()
            
            # Add scene-specific effects
            if "Character" in element:
                frame = self._add_character_highlight(frame)
            elif "Setting" in element:
                frame = self._add_setting_highlight(frame)
            
            frame = self._add_text_to_frame(frame, element, position='top')
            frames.append(frame)
        
        return frames
    
    def _add_character_highlight(self, image: np.ndarray) -> np.ndarray:
        """Add character highlighting effect"""
        # Simple brightness enhancement
        enhanced = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
        return enhanced
    
    def _add_setting_highlight(self, image: np.ndarray) -> np.ndarray:
        """Add setting highlighting effect"""
        # Simple contrast enhancement
        enhanced = cv2.convertScaleAbs(image, alpha=1.05, beta=5)
        return enhanced
    
    def _add_color_highlighting(self, image: np.ndarray) -> np.ndarray:
        """Add color highlighting effect"""
        # Enhance saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _add_shape_highlighting(self, image: np.ndarray) -> np.ndarray:
        """Add shape highlighting effect"""
        # Edge enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
    
    def _create_title_slide(self, title: str, duration: int, fps: int, frame_size: tuple) -> List[np.ndarray]:
        """Create title slide frames"""
        frames = []
        total_frames = duration * fps
        height, width = frame_size
        
        for i in range(total_frames):
            # Create blank slide
            slide = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Add title
            slide = self._add_text_to_frame(slide, title, position='center')
            frames.append(slide)
        
        return frames
    
    def _create_scientific_slide(self, image: np.ndarray, title: str, subtitle: str, duration: int, fps: int) -> List[np.ndarray]:
        """Create scientific presentation slide"""
        frames = []
        total_frames = duration * fps
        
        for i in range(total_frames):
            # Create slide layout
            slide = self._create_slide_layout(image, title, subtitle)
            frames.append(slide)
        
        return frames
    
    def _create_slide_layout(self, image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
        """Create slide layout with image and text"""
        height, width = image.shape[:2]
        
        # Create slide background
        slide = np.ones((height + 100, width, 3), dtype=np.uint8) * 240
        
        # Add image
        slide[50:50+height, :width] = image
        
        # Add title and subtitle
        slide = self._add_text_to_frame(slide, title, position='top')
        
        return slide
    
    def _create_data_slide(self, title: str, data: str, duration: int, fps: int, frame_size: tuple) -> List[np.ndarray]:
        """Create data presentation slide"""
        frames = []
        total_frames = duration * fps
        height, width = frame_size
        
        for i in range(total_frames):
            # Create data slide
            slide = np.ones((height, width, 3), dtype=np.uint8) * 245
            
            # Add title
            slide = self._add_text_to_frame(slide, title, position='top')
            
            # Add data (simplified)
            slide = self._add_text_to_frame(slide, data, position='center')
            
            frames.append(slide)
        
        return frames
    
    def _format_developmental_data(self, results: Dict) -> str:
        """Format developmental data for presentation"""
        dev_data = results.get('traditional_analysis', {}).get('developmental_assessment', {})
        level = dev_data.get('level', 'unknown').replace('_', ' ').title()
        return f"Level: {level}"
    
    def _format_color_data(self, results: Dict) -> str:
        """Format color data for presentation"""
        color_data = results.get('traditional_analysis', {}).get('color_analysis', {})
        dominant = color_data.get('dominant_color', 'Unknown')
        diversity = color_data.get('color_diversity', 0)
        return f"Dominant: {dominant}, Diversity: {diversity}"
    
    def _format_emotional_data(self, results: Dict) -> str:
        """Format emotional data for presentation"""
        emotional_data = results.get('traditional_analysis', {}).get('emotional_indicators', {})
        mood = emotional_data.get('overall_mood', 'neutral').title()
        return f"Overall Mood: {mood}"
    
    def _format_validation_data(self, results: Dict) -> str:
        """Format validation data for presentation"""
        if 'scientific_validation' in results:
            confidence = results.get('confidence_scores', {}).get('overall', 0)
            return f"Confidence: {confidence:.1%}"
        return "Validation: Standard Analysis"
    
    def _create_video_from_frames(self, frames: List[np.ndarray], filename: str, fps: int = 30) -> str:
        """Create video from list of frames"""
        
        if not frames:
            return None
        
        try:
            # Create output path
            output_path = os.path.join(self.temp_dir, filename)
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write frames
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            # Verify file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
            else:
                return None
                
        except Exception as e:
            print(f"Error creating video: {str(e)}")
            return None
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
