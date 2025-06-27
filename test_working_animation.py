#!/usr/bin/env python3
"""
Test the WORKING animation fix - NO TEXT OVERLAYS
"""

import os
from PIL import Image, ImageDraw

def create_test_image():
    """Create a colorful test image"""
    img = Image.new('RGB', (600, 400), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a house with bright colors
    draw.rectangle([150, 200, 450, 350], fill='yellow', outline='black', width=3)
    draw.polygon([150, 200, 300, 120, 450, 200], fill='red', outline='black', width=3)
    draw.rectangle([200, 250, 250, 320], fill='brown', outline='black', width=2)
    draw.ellipse([350, 230, 400, 280], fill='orange', outline='black', width=2)
    
    # Draw sun
    draw.ellipse([480, 50, 550, 120], fill='yellow', outline='orange', width=3)
    
    # Draw clouds
    draw.ellipse([100, 60, 180, 100], fill='white', outline='gray')
    draw.ellipse([300, 40, 380, 80], fill='white', outline='gray')
    
    img.save('test_working_animation.png')
    return 'test_working_animation.png'

def test_working_animations():
    """Test all WORKING animation styles"""
    print("🧪 Testing WORKING Animation Styles (No Text Overlays)...")
    print("=" * 60)
    
    # Import the fixed video generator
    from video_generator import VideoGenerator
    
    # Create test data
    image_path = create_test_image()
    analysis_results = {
        'input_info': {
            'child_age': 7,
            'drawing_context': 'House Drawing'
        },
        'traditional_analysis': {
            'blip_description': 'a colorful drawing of a house with a sun and clouds',
            'emotional_indicators': {
                'overall_mood': 'positive'
            }
        }
    }
    
    # Test all animation styles
    vg = VideoGenerator()
    styles = ['animated', 'particle', 'floating']
    
    for style in styles:
        print(f"\n🎬 Testing WORKING {style} animation...")
        try:
            result = vg.generate_memory_video(
                image_path,
                analysis_results,
                f"This is a test of WORKING {style} animation!",
                animation_style=style
            )
            
            if 'video_path' in result:
                print(f"✅ {style.upper()} SUCCESS! Video: {result['video_path']}")
                if os.path.exists(result['video_path']):
                    file_size = os.path.getsize(result['video_path'])
                    print(f"   📁 File size: {file_size / 1024:.1f} KB")
                    print(f"   ⏱️ Duration: {result.get('duration_seconds', 'Unknown')} seconds")
                    print(f"   🎯 This video should now have WORKING animations!")
            else:
                print(f"❌ {style.upper()} FAILED: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ {style.upper()} EXCEPTION: {e}")
    
    # Cleanup
    if os.path.exists(image_path):
        os.remove(image_path)
    
    print(f"\n🎉 Test complete! Check the generated videos - they should now have WORKING animations!")
    print("📝 Note: Text overlays have been disabled to avoid font issues.")
    print("🎬 You should see: circular motion, sliding, bouncing, zooming, particles, and floating effects!")

if __name__ == "__main__":
    test_working_animations()

