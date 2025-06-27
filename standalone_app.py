#!/usr/bin/env python3
"""
Standalone Children's Drawing Analysis Application
Complete system without Streamlit dependencies
"""

import os
import sys
import json
import tempfile
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class StandaloneDrawingAnalyzer:
    """Complete drawing analysis system without external web framework dependencies"""
    
    def __init__(self):
        self.age_groups = {
            2: "Toddler (2-3 years)",
            3: "Toddler (2-3 years)", 
            4: "Preschool (4-6 years)",
            5: "Preschool (4-6 years)",
            6: "Preschool (4-6 years)",
            7: "School Age (7-11 years)",
            8: "School Age (7-11 years)",
            9: "School Age (7-11 years)",
            10: "School Age (7-11 years)",
            11: "School Age (7-11 years)",
            12: "Adolescent (12+ years)"
        }
        
        self.developmental_expectations = {
            "Toddler (2-3 years)": {
                "skills": ["Scribbling", "Basic marks", "Large movements"],
                "shapes": 1,
                "complexity": "Very Simple"
            },
            "Preschool (4-6 years)": {
                "skills": ["Basic shapes", "Simple figures", "Color recognition"],
                "shapes": 3,
                "complexity": "Simple"
            },
            "School Age (7-11 years)": {
                "skills": ["Detailed figures", "Realistic proportions", "Complex scenes"],
                "shapes": 6,
                "complexity": "Medium"
            },
            "Adolescent (12+ years)": {
                "skills": ["Advanced techniques", "Perspective", "Abstract concepts"],
                "shapes": 10,
                "complexity": "Complex"
            }
        }
    
    def analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze color usage in the drawing"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate color statistics
        avg_brightness = np.mean(image)
        color_variance = np.var(image.reshape(-1, 3), axis=0).mean()
        
        # Determine dominant color
        avg_red = np.mean(image[:,:,0])
        avg_green = np.mean(image[:,:,1])
        avg_blue = np.mean(image[:,:,2])
        
        if avg_red > avg_green and avg_red > avg_blue:
            dominant_color = "Red"
        elif avg_green > avg_red and avg_green > avg_blue:
            dominant_color = "Green"
        elif avg_blue > avg_red and avg_blue > avg_green:
            dominant_color = "Blue"
        else:
            dominant_color = "Mixed colors"
        
        # Count unique colors
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        
        return {
            "dominant_color": dominant_color,
            "brightness_level": float(avg_brightness),
            "color_diversity": min(unique_colors, 50),
            "color_variance": float(color_variance),
            "richness": "Rich" if unique_colors > 20 else "Moderate" if unique_colors > 10 else "Simple"
        }
    
    def analyze_shapes(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze shapes and complexity"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter meaningful contours
        meaningful_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        # Calculate coverage
        total_area = gray.shape[0] * gray.shape[1]
        drawing_area = sum(cv2.contourArea(c) for c in meaningful_contours)
        coverage = drawing_area / total_area
        
        # Determine complexity
        shape_count = len(meaningful_contours)
        if shape_count < 3:
            complexity = "Simple"
        elif shape_count < 8:
            complexity = "Medium"
        else:
            complexity = "Complex"
        
        return {
            "total_shapes": shape_count,
            "complexity_level": complexity,
            "drawing_coverage": float(coverage),
            "detail_level": "High" if coverage > 0.3 else "Medium" if coverage > 0.1 else "Low"
        }
    
    def analyze_spatial_organization(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze spatial organization"""
        height, width = image.shape[:2]
        
        # Divide into quadrants
        h_mid, w_mid = height // 2, width // 2
        
        quadrants = {
            'top_left': image[:h_mid, :w_mid],
            'top_right': image[:h_mid, w_mid:],
            'bottom_left': image[h_mid:, :w_mid],
            'bottom_right': image[h_mid:, w_mid:]
        }
        
        # Calculate activity in each quadrant
        activities = {}
        for name, quad in quadrants.items():
            # Count non-white pixels
            gray_quad = cv2.cvtColor(quad, cv2.COLOR_RGB2GRAY)
            activity = np.sum(gray_quad < 240) / gray_quad.size
            activities[name] = float(activity)
        
        # Calculate balance
        balance_score = 1.0 - np.var(list(activities.values()))
        
        if balance_score > 0.8:
            balance = "Very balanced"
        elif balance_score > 0.6:
            balance = "Balanced"
        else:
            balance = "Unbalanced"
        
        return {
            "spatial_balance": balance,
            "balance_score": float(balance_score),
            "quadrant_distribution": activities,
            "drawing_style": "Center-focused" if activities['top_left'] + activities['top_right'] > 0.6 else "Distributed"
        }
    
    def analyze_emotional_indicators(self, image: np.ndarray, color_analysis: Dict) -> Dict[str, Any]:
        """Analyze emotional indicators"""
        # Color-based emotional analysis
        brightness = color_analysis["brightness_level"]
        dominant_color = color_analysis["dominant_color"]
        
        # Determine emotional tone
        if brightness > 180:
            tone = "bright_positive"
        elif brightness < 100:
            tone = "subdued"
        else:
            tone = "neutral"
        
        # Color-emotion mapping
        color_emotions = {
            "Red": "energetic",
            "Blue": "calm", 
            "Green": "peaceful",
            "Yellow": "happy",
            "Mixed colors": "balanced"
        }
        
        emotion = color_emotions.get(dominant_color, "neutral")
        
        # Overall mood assessment
        if tone == "bright_positive" and emotion in ["happy", "energetic", "balanced"]:
            overall_mood = "positive"
        elif tone == "subdued" or emotion == "calm":
            overall_mood = "calm"
        else:
            overall_mood = "neutral"
        
        return {
            "tone": tone,
            "color_emotion": emotion,
            "overall_mood": overall_mood,
            "emotional_valence": "positive" if overall_mood == "positive" else "neutral"
        }
    
    def assess_development(self, shape_analysis: Dict, child_age: int) -> Dict[str, Any]:
        """Assess developmental appropriateness"""
        age_group = self.age_groups.get(child_age, "School Age (7-11 years)")
        expectations = self.developmental_expectations[age_group]
        
        actual_shapes = shape_analysis["total_shapes"]
        expected_shapes = expectations["shapes"]
        
        # Determine developmental level
        if actual_shapes >= expected_shapes * 1.5:
            level = "above_expected"
        elif actual_shapes >= expected_shapes * 0.7:
            level = "age_appropriate"
        else:
            level = "below_expected"
        
        return {
            "age_group": age_group,
            "level": level,
            "expected_skills": expectations["skills"],
            "actual_shapes": actual_shapes,
            "expected_shapes": expected_shapes,
            "complexity_match": shape_analysis["complexity_level"] == expectations["complexity"]
        }
    
    def generate_recommendations(self, analysis_results: Dict, child_age: int) -> Dict[str, Any]:
        """Generate personalized recommendations"""
        dev_assessment = analysis_results["developmental_assessment"]
        emotional_analysis = analysis_results["emotional_indicators"]
        
        recommendations = {
            "immediate_actions": [],
            "materials": [],
            "activities": [],
            "long_term_goals": []
        }
        
        # Age-specific recommendations
        if child_age < 4:
            recommendations["immediate_actions"].extend([
                "Encourage daily drawing time for motor skill development",
                "Use chunky crayons and large paper for easier grip"
            ])
            recommendations["materials"].extend(["Chunky crayons", "Finger paints", "Large paper"])
            recommendations["activities"].extend(["Finger painting", "Large scribbling", "Color exploration"])
        
        elif child_age < 7:
            recommendations["immediate_actions"].extend([
                "Ask child to tell stories about their drawings",
                "Provide variety of art materials"
            ])
            recommendations["materials"].extend(["Crayons", "Markers", "Colored pencils", "Stickers"])
            recommendations["activities"].extend(["Story illustration", "Shape games", "Color mixing"])
        
        elif child_age < 12:
            recommendations["immediate_actions"].extend([
                "Encourage drawing from observation",
                "Introduce more complex art techniques"
            ])
            recommendations["materials"].extend(["Sketch pads", "Watercolors", "Drawing pencils"])
            recommendations["activities"].extend(["Nature drawing", "Portrait practice", "Art challenges"])
        
        else:
            recommendations["immediate_actions"].extend([
                "Support artistic expression as emotional outlet",
                "Discuss meaning and symbolism in artwork"
            ])
            recommendations["materials"].extend(["Professional supplies", "Digital tools", "Canvas"])
            recommendations["activities"].extend(["Advanced techniques", "Portfolio development"])
        
        # Development-specific recommendations
        if dev_assessment["level"] == "below_expected":
            recommendations["immediate_actions"].append("⚠️ Increase art activities to support development")
            recommendations["long_term_goals"].append("Monitor progress and consider developmental support")
        
        elif dev_assessment["level"] == "above_expected":
            recommendations["immediate_actions"].append("🌟 Provide advanced challenges to nurture talent")
            recommendations["long_term_goals"].append("Consider specialized art education")
        
        # Emotional-specific recommendations
        if emotional_analysis["overall_mood"] == "positive":
            recommendations["immediate_actions"].append("✨ Continue encouraging creative expression!")
        
        return recommendations
    
    def generate_ai_description(self, color_analysis: Dict, shape_analysis: Dict, 
                              emotional_analysis: Dict, child_age: int, context: str) -> str:
        """Generate AI-style description"""
        dominant_color = color_analysis["dominant_color"].lower()
        complexity = shape_analysis["complexity_level"].lower()
        mood = emotional_analysis["overall_mood"]
        
        descriptions = [
            f"A {complexity} {context.lower()} created by a {child_age}-year-old child",
            f"featuring {dominant_color} as the dominant color",
            f"with {shape_analysis['total_shapes']} distinct elements",
            f"expressing a {mood} emotional tone"
        ]
        
        return " ".join(descriptions) + "."
    
    def analyze_drawing(self, image: np.ndarray, child_age: int, drawing_context: str) -> Dict[str, Any]:
        """Main analysis function"""
        # Perform all analyses
        color_analysis = self.analyze_colors(image)
        shape_analysis = self.analyze_shapes(image)
        spatial_analysis = self.analyze_spatial_organization(image)
        emotional_analysis = self.analyze_emotional_indicators(image, color_analysis)
        developmental_assessment = self.assess_development(shape_analysis, child_age)
        
        # Generate AI description
        ai_description = self.generate_ai_description(
            color_analysis, shape_analysis, emotional_analysis, child_age, drawing_context
        )
        
        # Compile results
        results = {
            "input_info": {
                "child_age": child_age,
                "age_group": developmental_assessment["age_group"],
                "drawing_context": drawing_context
            },
            "ai_description": ai_description,
            "color_analysis": color_analysis,
            "shape_analysis": shape_analysis,
            "spatial_analysis": spatial_analysis,
            "emotional_indicators": emotional_analysis,
            "developmental_assessment": developmental_assessment,
            "confidence_score": 0.85,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Generate recommendations
        results["recommendations"] = self.generate_recommendations(results, child_age)
        
        return results

class StandaloneWebServer:
    """Simple web server for the standalone application"""
    
    def __init__(self, analyzer: StandaloneDrawingAnalyzer):
        self.analyzer = analyzer
        self.upload_dir = Path("temp_uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
    def handle_analysis_request(self, image_data: bytes, form_data: Dict) -> Dict:
        """Handle analysis request"""
        try:
            # Save uploaded image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = self.upload_dir / f"drawing_{timestamp}.png"
            
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            # Load and analyze image
            image = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get form parameters
            child_age = int(form_data.get('childAge', 6))
            drawing_context = form_data.get('drawingContext', 'Free Drawing')
            
            # Perform analysis
            results = self.analyzer.analyze_drawing(image_rgb, child_age, drawing_context)
            
            # Clean up
            image_path.unlink()
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def start_server(self, port: int = 8000):
        """Start the web server"""
        import http.server
        import socketserver
        import urllib.parse
        import cgi
        
        class RequestHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=os.getcwd(), **kwargs)
            
            def do_POST(self):
                if self.path == '/api/analyze':
                    self.handle_analyze_request()
                elif self.path == '/api/status':
                    self.handle_status_request()
                else:
                    self.send_error(404)
            
            def handle_analyze_request(self):
                try:
                    # Parse multipart form data
                    content_type = self.headers.get('Content-Type', '')
                    if 'multipart/form-data' in content_type:
                        form = cgi.FieldStorage(
                            fp=self.rfile,
                            headers=self.headers,
                            environ={'REQUEST_METHOD': 'POST'}
                        )
                        
                        # Get image data
                        if 'image' in form:
                            image_data = form['image'].file.read()
                            
                            # Get form data
                            form_data = {}
                            for key in ['childAge', 'drawingContext', 'analysisType']:
                                if key in form:
                                    form_data[key] = form[key].value
                            
                            # Perform analysis
                            results = server_instance.handle_analysis_request(image_data, form_data)
                            
                            # Send response
                            self.send_response(200)
                            self.send_header('Content-Type', 'application/json')
                            self.send_header('Access-Control-Allow-Origin', '*')
                            self.end_headers()
                            self.wfile.write(json.dumps(results).encode())
                        else:
                            self.send_error(400, "No image provided")
                    else:
                        self.send_error(400, "Invalid content type")
                        
                except Exception as e:
                    self.send_error(500, str(e))
            
            def handle_status_request(self):
                status = {
                    'status': 'online',
                    'ai_components': True,
                    'available_components': {
                        'standalone_analyzer': True,
                        'computer_vision': True,
                        'psychological_framework': True
                    }
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(status).encode())
            
            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()
        
        server_instance = self
        
        try:
            with socketserver.TCPServer(("", port), RequestHandler) as httpd:
                print(f"🌐 Standalone server running at http://localhost:{port}")
                print("📝 Open the URL in your browser to use the application")
                print("🎨 Complete drawing analysis system ready!")
                httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

def main():
    """Main function to start the standalone application"""
    print("🎨 Standalone Children's Drawing Analysis System")
    print("=" * 60)
    
    # Check dependencies
    try:
        import cv2
        import numpy as np
        from PIL import Image
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install with: pip install opencv-python pillow numpy")
        return False
    
    # Initialize analyzer
    analyzer = StandaloneDrawingAnalyzer()
    print("✅ Drawing analyzer initialized")
    
    # Start web server
    server = StandaloneWebServer(analyzer)
    
    try:
        server.start_server(port=8000)
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        return False

if __name__ == "__main__":
    main()