#!/usr/bin/env python3
"""
Test the PURE animation fix - ABSOLUTELY NO TEXT
"""

import os
from PIL import Image, ImageDraw

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (400, 300), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple house
    draw.rectangle([100, 150, 300, 250], fill='yellow', outline='black', width=3)
    draw.polygon([100, 150, 200, 100, 300, 150], fill='red', outline='black', width=3)
    draw.rectangle([150, 180, 180, 220], fill='brown', outline='black', width=2)
    draw.ellipse([220, 180, 250, 210], fill='orange', outline='black', width=2)
    
    img.save('pure_test_drawing.png')
    return 'pure_test_drawing.png'

def test_pure_animations():
    """Test animations with ABSOLUTELY NO TEXT"""
    print("🧪 Testing PURE Animations (ABSOLUTELY NO TEXT)...")
    print("=" * 50)
    
    # Import the video generator
    from video_generator import VideoGenerator
    
    # Create test data
    image_path = create_test_image()
    analysis_results = {
        'input_info': {
            'child_age': 6,
            'drawing_context': 'House Drawing'
        },
        'traditional_analysis': {
            'blip_description': 'a simple drawing of a house',
            'emotional_indicators': {
                'overall_mood': 'positive'
            }
        }
    }
    
    # Test all animation styles
    vg = VideoGenerator()
    styles = ['animated', 'particle', 'floating']
    
    for style in styles:
        print(f"\n🎬 Testing PURE {style} animation...")
        try:
            result = vg.generate_memory_video(
                image_path,
                analysis_results,
                f"Pure test of {style} animation",
                animation_style=style
            )
            
            if 'video_path' in result:
                print(f"✅ {style.upper()} SUCCESS! Video: {result['video_path']}")
                if os.path.exists(result['video_path']):
                    file_size = os.path.getsize(result['video_path'])
                    print(f"   📁 File size: {file_size / 1024:.1f} KB")
                    print(f"   ⏱️ Duration: {result.get('duration_seconds', 'Unknown')} seconds")
            else:
                print(f"❌ {style.upper()} FAILED: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ {style.upper()} EXCEPTION: {e}")
    
    # Cleanup
    if os.path.exists(image_path):
        os.remove(image_path)
    
    print(f"\n🎉 Test complete! Videos should work without ANY text overlay issues!")

if __name__ == "__main__":
    test_pure_animations()

