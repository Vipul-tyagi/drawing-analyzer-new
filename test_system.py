#!/usr/bin/env python3
"""
Comprehensive system test for Children's Drawing Analysis System
"""

import os
import sys
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import json

def create_test_image():
    """Create a test drawing image"""
    # Create a colorful test drawing
    img = Image.new('RGB', (600, 400), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a house
    draw.rectangle([150, 200, 450, 350], fill='yellow', outline='black', width=3)
    draw.polygon([150, 200, 300, 120, 450, 200], fill='red', outline='black', width=3)
    draw.rectangle([200, 250, 250, 320], fill='brown', outline='black', width=2)
    draw.ellipse([350, 230, 400, 280], fill='orange', outline='black', width=2)
    
    # Draw sun
    draw.ellipse([480, 50, 550, 120], fill='yellow', outline='orange', width=3)
    
    # Draw clouds
    draw.ellipse([100, 60, 180, 100], fill='white', outline='gray')
    draw.ellipse([300, 40, 380, 80], fill='white', outline='gray')
    
    # Draw tree
    draw.rectangle([80, 280, 100, 350], fill='brown', outline='black', width=2)
    draw.ellipse([60, 240, 120, 300], fill='green', outline='darkgreen', width=2)
    
    return img

def test_imports():
    """Test critical imports"""
    print("🧪 Testing imports...")
    
    results = {}
    
    # Core dependencies
    try:
        import streamlit
        results['streamlit'] = True
        print("✅ Streamlit")
    except ImportError:
        results['streamlit'] = False
        print("❌ Streamlit")
    
    try:
        import cv2
        results['opencv'] = True
        print("✅ OpenCV")
    except ImportError:
        results['opencv'] = False
        print("❌ OpenCV")
    
    try:
        import numpy as np
        results['numpy'] = True
        print("✅ NumPy")
    except ImportError:
        results['numpy'] = False
        print("❌ NumPy")
    
    try:
        from PIL import Image
        results['pillow'] = True
        print("✅ Pillow")
    except ImportError:
        results['pillow'] = False
        print("❌ Pillow")
    
    # AI dependencies
    try:
        import torch
        results['pytorch'] = True
        print("✅ PyTorch")
        
        if torch.cuda.is_available():
            print(f"  🚀 GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠️ No GPU available")
    except ImportError:
        results['pytorch'] = False
        print("❌ PyTorch")
    
    try:
        import transformers
        results['transformers'] = True
        print("✅ Transformers")
    except ImportError:
        results['transformers'] = False
        print("❌ Transformers")
    
    # Video generation
    try:
        import moviepy
        results['moviepy'] = True
        print("✅ MoviePy")
    except ImportError:
        results['moviepy'] = False
        print("❌ MoviePy")
    
    # Report generation
    try:
        import reportlab
        results['reportlab'] = True
        print("✅ ReportLab")
    except ImportError:
        results['reportlab'] = False
        print("❌ ReportLab")
    
    return results

def test_analysis_components():
    """Test analysis components"""
    print("\n🧪 Testing analysis components...")
    
    results = {}
    
    # Enhanced Drawing Analyzer
    try:
        from enhanced_drawing_analyzer import EnhancedDrawingAnalyzer
        analyzer = EnhancedDrawingAnalyzer()
        results['enhanced_analyzer'] = True
        print("✅ Enhanced Drawing Analyzer")
    except ImportError as e:
        results['enhanced_analyzer'] = False
        print(f"❌ Enhanced Drawing Analyzer: {e}")
    
    # Video Generator
    try:
        from video_generator import VideoGenerator
        video_gen = VideoGenerator()
        results['video_generator'] = True
        print("✅ Video Generator")
    except ImportError as e:
        results['video_generator'] = False
        print(f"❌ Video Generator: {e}")
    
    # AI Analysis Engine
    try:
        from ai_analysis_engine import ComprehensiveAIAnalyzer
        ai_analyzer = ComprehensiveAIAnalyzer()
        results['ai_analyzer'] = True
        print("✅ AI Analysis Engine")
    except ImportError as e:
        results['ai_analyzer'] = False
        print(f"❌ AI Analysis Engine: {e}")
    
    # Clinical Assessment
    try:
        from clinical_assessment_advanced import AdvancedClinicalAssessment
        clinical = AdvancedClinicalAssessment()
        results['clinical_assessment'] = True
        print("✅ Clinical Assessment")
    except ImportError as e:
        results['clinical_assessment'] = False
        print(f"❌ Clinical Assessment: {e}")
    
    return results

def test_basic_analysis():
    """Test basic analysis functionality"""
    print("\n🧪 Testing basic analysis...")
    
    try:
        # Create test image
        test_img = create_test_image()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            test_img.save(tmp_file.name)
            temp_path = tmp_file.name
        
        print(f"📸 Test image created: {temp_path}")
        
        # Try basic analysis
        try:
            from enhanced_drawing_analyzer import EnhancedDrawingAnalyzer
            analyzer = EnhancedDrawingAnalyzer()
            
            print("🔍 Running basic analysis...")
            results = analyzer.analyze_drawing_comprehensive(
                temp_path, 
                child_age=6, 
                drawing_context="Test Drawing"
            )
            
            if results and 'error' not in results:
                print("✅ Basic analysis successful")
                
                # Check key components
                if 'traditional_analysis' in results:
                    print("  ✅ Traditional analysis")
                if 'confidence_scores' in results:
                    print(f"  ✅ Confidence: {results['confidence_scores']['overall']:.1%}")
                if 'llm_analyses' in results:
                    print(f"  ✅ LLM analyses: {len(results['llm_analyses'])}")
                
                return True
            else:
                print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
                return False
        
        except ImportError:
            print("⚠️ Enhanced analyzer not available - skipping analysis test")
            return None
        
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        print(f"❌ Basic analysis test failed: {e}")
        return False

def test_video_generation():
    """Test video generation"""
    print("\n🧪 Testing video generation...")
    
    try:
        from video_generator import VideoGenerator
        
        # Create test image
        test_img = create_test_image()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            test_img.save(tmp_file.name)
            temp_path = tmp_file.name
        
        # Mock analysis results
        mock_results = {
            'input_info': {'child_age': 6, 'drawing_context': 'Test Drawing'},
            'traditional_analysis': {
                'blip_description': 'a colorful test drawing',
                'emotional_indicators': {'overall_mood': 'positive'}
            },
            'confidence_scores': {'overall': 0.85}
        }
        
        video_gen = VideoGenerator()
        print("🎬 Generating test video...")
        
        result = video_gen.generate_memory_video(
            temp_path,
            mock_results,
            "Test video generation",
            animation_style='animated'
        )
        
        if result and 'video_path' in result:
            print("✅ Video generation successful")
            print(f"  📁 Video: {result['video_path']}")
            print(f"  🎯 Method: {result['generation_method']}")
            
            # Clean up video
            if os.path.exists(result['video_path']):
                os.unlink(result['video_path'])
            
            return True
        else:
            print(f"❌ Video generation failed: {result.get('error', 'Unknown error')}")
            return False
    
    except ImportError:
        print("⚠️ Video generator not available - skipping video test")
        return None
    except Exception as e:
        print(f"❌ Video generation test failed: {e}")
        return False
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)

def test_streamlit_app():
    """Test Streamlit app startup"""
    print("\n🧪 Testing Streamlit app...")
    
    try:
        import streamlit as st
        
        # Check if app.py exists
        if not os.path.exists('app.py'):
            print("❌ app.py not found")
            return False
        
        print("✅ Streamlit app file exists")
        print("💡 To test the app, run: streamlit run app.py")
        return True
    
    except ImportError:
        print("❌ Streamlit not available")
        return False

def generate_test_report(results):
    """Generate test report"""
    print("\n📊 Test Report")
    print("=" * 60)
    
    # Count successes
    import_results = results.get('imports', {})
    component_results = results.get('components', {})
    
    total_imports = len(import_results)
    successful_imports = sum(import_results.values())
    
    total_components = len(component_results)
    successful_components = sum(component_results.values())
    
    print(f"📦 Dependencies: {successful_imports}/{total_imports} available")
    print(f"🔧 Components: {successful_components}/{total_components} working")
    
    if results.get('basic_analysis'):
        print("✅ Basic analysis: Working")
    elif results.get('basic_analysis') is False:
        print("❌ Basic analysis: Failed")
    else:
        print("⚠️ Basic analysis: Not tested")
    
    if results.get('video_generation'):
        print("✅ Video generation: Working")
    elif results.get('video_generation') is False:
        print("❌ Video generation: Failed")
    else:
        print("⚠️ Video generation: Not tested")
    
    if results.get('streamlit_app'):
        print("✅ Streamlit app: Ready")
    else:
        print("❌ Streamlit app: Not ready")
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if not import_results.get('streamlit'):
        print("  📦 Install Streamlit: pip install streamlit")
    
    if not import_results.get('opencv'):
        print("  📦 Install OpenCV: pip install opencv-python")
    
    if not import_results.get('pytorch'):
        print("  📦 Install PyTorch: pip install torch torchvision")
    
    if not import_results.get('transformers'):
        print("  📦 Install Transformers: pip install transformers")
    
    if not import_results.get('moviepy'):
        print("  📦 Install MoviePy: pip install moviepy")
    
    if not import_results.get('reportlab'):
        print("  📦 Install ReportLab: pip install reportlab")
    
    if successful_imports >= 4:
        print("  🚀 You can start the basic web interface")
    
    if successful_imports >= 6:
        print("  🎯 You can run the full Streamlit application")
    
    print("\n🚀 To start the application:")
    print("  python main.py")
    print("  OR")
    print("  streamlit run app.py")

def main():
    """Main test function"""
    print("🧪 Children's Drawing Analysis System - Comprehensive Test")
    print("=" * 80)
    
    results = {}
    
    # Test imports
    results['imports'] = test_imports()
    
    # Test components
    results['components'] = test_analysis_components()
    
    # Test basic analysis
    results['basic_analysis'] = test_basic_analysis()
    
    # Test video generation
    results['video_generation'] = test_video_generation()
    
    # Test Streamlit app
    results['streamlit_app'] = test_streamlit_app()
    
    # Generate report
    generate_test_report(results)
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Test results saved to: test_results.json")

if __name__ == "__main__":
    main()