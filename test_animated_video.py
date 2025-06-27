#!/usr/bin/env python3
"""
Test script for animated video generation
"""

import os
import sys
from PIL import Image, ImageDraw
import numpy as np

def create_test_image():
    """Create a colorful test image"""
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
    
    img.save('test_animated_drawing.png')
    return 'test_animated_drawing.png'

def create_test_analysis():
    """Create dummy analysis results"""
    return {
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

def test_all_animations():
    """Test all animation styles"""
    print("🧪 Testing All Animation Styles...")
    print("=" * 60)
    
    # Setup
    try:
        from video_generator import VideoGenerator, test_video_generator
        print("✅ Video generator imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Run system test
    test_video_generator()
    
    # Create test data
    print("\n📸 Creating test image...")
    image_path = create_test_image()
    analysis_results = create_test_analysis()
    
    # Test all animation styles
    vg = VideoGenerator()
    styles = ['animated', 'particle', 'floating', 'simple']
    
    results = {}
    
    for style in styles:
        print(f"\n🎬 Testing {style} animation...")
        try:
            result = vg.generate_memory_video(
                image_path,
                analysis_results,
                f"This is a test of {style} animation style!",
                animation_style=style
            )
            
            if 'video_path' in result:
                print(f"✅ {style.upper()} SUCCESS! Video: {result['video_path']}")
                if os.path.exists(result['video_path']):
                    file_size = os.path.getsize(result['video_path'])
                    print(f"   📁 File size: {file_size / 1024:.1f} KB")
                    print(f"   ⏱️ Duration: {result.get('duration_seconds', 'Unknown')} seconds")
                    results[style] = True
                else:
                    print(f"   ⚠️ Video path returned but file doesn't exist")
                    results[style] = False
            else:
                print(f"❌ {style.upper()} FAILED: {result.get('error', 'Unknown error')}")
                results[style] = False
                
        except Exception as e:
            print(f"❌ {style.upper()} EXCEPTION: {e}")
            results[style] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 ANIMATION TEST RESULTS:")
    for style, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {style.upper()}: {'Working' if success else 'Failed'}")
    
    successful_count = sum(results.values())
    print(f"\n🎯 {successful_count}/{len(styles)} animation styles working!")
    
    # Cleanup
    if os.path.exists(image_path):
        os.remove(image_path)
    
    return successful_count > 0

if __name__ == "__main__":
    success = test_all_animations()
    
    if success:
        print("\n🎉 ANIMATED VIDEO GENERATION IS WORKING!")
        print("Your app should now create beautiful animated videos.")
    else:
        print("\n❌ ANIMATED VIDEO GENERATION HAS ISSUES")
        print("Please check the error messages above.")

