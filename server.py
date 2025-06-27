#!/usr/bin/env python3
"""
Optional Python backend server for full AI analysis capabilities
Run this if you want the complete AI-powered analysis features
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create upload directory
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# Try to import analysis components
components = {}

def load_components():
    """Load analysis components with error handling"""
    global components
    
    try:
        from enhanced_drawing_analyzer import EnhancedDrawingAnalyzer, ScientificallyValidatedAnalyzer
        components['enhanced_analyzer'] = EnhancedDrawingAnalyzer()
        components['validated_analyzer'] = ScientificallyValidatedAnalyzer()
        print("✅ Enhanced Drawing Analyzer loaded")
    except ImportError as e:
        print(f"⚠️ Enhanced Drawing Analyzer not available: {e}")
        components['enhanced_analyzer'] = None
        components['validated_analyzer'] = None
    
    try:
        from video_generator import VideoGenerator
        components['video_generator'] = VideoGenerator()
        print("✅ Video Generator loaded")
    except ImportError as e:
        print(f"⚠️ Video Generator not available: {e}")
        components['video_generator'] = None
    
    try:
        from ai_analysis_engine import ComprehensiveAIAnalyzer
        components['ai_analyzer'] = ComprehensiveAIAnalyzer()
        print("✅ AI Analysis Engine loaded")
    except ImportError as e:
        print(f"⚠️ AI Analysis Engine not available: {e}")
        components['ai_analyzer'] = None
    
    try:
        from clinical_assessment_advanced import AdvancedClinicalAssessment
        components['clinical_assessment'] = AdvancedClinicalAssessment()
        print("✅ Clinical Assessment loaded")
    except ImportError as e:
        print(f"⚠️ Clinical Assessment not available: {e}")
        components['clinical_assessment'] = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'status': 'online',
        'ai_components': any(components.values()),
        'available_components': {
            'enhanced_analyzer': components.get('enhanced_analyzer') is not None,
            'video_generator': components.get('video_generator') is not None,
            'ai_analyzer': components.get('ai_analyzer') is not None,
            'clinical_assessment': components.get('clinical_assessment') is not None
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_drawing():
    """Analyze uploaded drawing"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get form data
        child_age = int(request.form.get('childAge', 6))
        drawing_context = request.form.get('drawingContext', 'Free Drawing')
        analysis_type = request.form.get('analysisType', 'Basic Analysis')
        generate_pdf = request.form.get('generatePdf', 'false').lower() == 'true'
        generate_video = request.form.get('generateVideo', 'false').lower() == 'true'
        video_style = request.form.get('videoStyle', 'intelligent')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Perform analysis based on type
        results = None
        
        if analysis_type == "Basic Analysis":
            if components['enhanced_analyzer']:
                results = components['enhanced_analyzer'].analyze_drawing_comprehensive(
                    filepath, child_age, drawing_context
                )
        
        elif analysis_type == "Enhanced Analysis":
            if components['enhanced_analyzer']:
                results = components['enhanced_analyzer'].analyze_drawing_with_pdf_report(
                    filepath, child_age, drawing_context, generate_pdf
                )
        
        elif analysis_type == "Scientific Validation":
            if components['validated_analyzer']:
                results = components['validated_analyzer'].analyze_drawing_with_validation(
                    filepath, child_age, drawing_context
                )
        
        elif analysis_type == "Clinical Assessment":
            if components['clinical_assessment'] and components['enhanced_analyzer']:
                import cv2
                # Get basic analysis first
                basic_results = components['enhanced_analyzer'].analyze_drawing_comprehensive(
                    filepath, child_age, drawing_context
                )
                
                # Add clinical assessment
                image = cv2.imread(filepath)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                clinical_results = components['clinical_assessment'].conduct_trauma_assessment(
                    image_rgb, basic_results.get('traditional_analysis', {}), child_age
                )
                
                attachment_results = components['clinical_assessment'].assess_attachment_patterns(
                    image_rgb, basic_results.get('traditional_analysis', {}), drawing_context
                )
                
                results = basic_results
                results['clinical_assessment'] = {
                    'trauma_assessment': clinical_results,
                    'attachment_assessment': attachment_results
                }
        
        elif analysis_type == "AI Multi-Model":
            if components['ai_analyzer']:
                ai_results = components['ai_analyzer'].conduct_multi_ai_analysis(
                    filepath, child_age, drawing_context
                )
                
                if components['enhanced_analyzer']:
                    enhanced_results = components['enhanced_analyzer'].analyze_drawing_comprehensive(
                        filepath, child_age, drawing_context
                    )
                    results = enhanced_results
                    results['ai_multi_analysis'] = ai_results
                else:
                    results = {'ai_multi_analysis': ai_results}
        
        elif analysis_type == "Complete Analysis":
            if components['validated_analyzer']:
                results = components['validated_analyzer'].analyze_drawing_with_psydraw_validation(
                    filepath, child_age, drawing_context
                )
            elif components['enhanced_analyzer']:
                results = components['enhanced_analyzer'].analyze_drawing_with_pdf_report(
                    filepath, child_age, drawing_context, generate_pdf
                )
        
        if not results:
            return jsonify({'error': 'No analysis components available'}), 500
        
        # Generate video if requested
        if generate_video and components['video_generator']:
            try:
                video_result = components['video_generator'].generate_memory_video(
                    filepath,
                    results,
                    f"Watch this amazing {child_age}-year-old's {drawing_context.lower()} come to life!",
                    animation_style=video_style
                )
                
                if 'video_path' in video_result:
                    results['video_path'] = video_result['video_path']
                    results['video_info'] = video_result
            except Exception as e:
                print(f"Video generation failed: {e}")
                results['video_error'] = str(e)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Analysis error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf():
    """Generate PDF report"""
    try:
        data = request.get_json()
        
        if not components.get('enhanced_analyzer'):
            return jsonify({'error': 'PDF generation not available'}), 500
        
        # This would need to be implemented based on your PDF generator
        return jsonify({'error': 'PDF generation endpoint not implemented'}), 501
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-video', methods=['POST'])
def generate_video():
    """Generate memory video"""
    try:
        data = request.get_json()
        
        if not components.get('video_generator'):
            return jsonify({'error': 'Video generation not available'}), 500
        
        # This would need to be implemented based on your video generator
        return jsonify({'error': 'Video generation endpoint not implemented'}), 501
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    return send_file('index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_file(filename)

if __name__ == '__main__':
    print("🚀 Starting Children's Drawing Analysis Server...")
    print("=" * 60)
    
    # Load components
    load_components()
    
    print("\n📊 Component Status:")
    for name, component in components.items():
        status = "✅" if component else "❌"
        print(f"{status} {name}")
    
    print(f"\n🌐 Server starting...")
    print("📝 Access the application at: http://localhost:5000")
    print("⚠️ If Python components are not available, the app will run in demo mode")
    
    app.run(debug=True, host='0.0.0.0', port=5000)