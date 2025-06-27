#!/usr/bin/env python3
"""
Test element-based animation - FIXED VERSION
"""

import os
from PIL import Image, ImageDraw

def create_complex_test_image():
    """Create a drawing with multiple distinct elements"""
    img = Image.new('RGB', (600, 400), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw house (multiple elements)
    draw.rectangle([150, 200, 350, 300], fill='yellow', outline='black', width=3)  # House body
    draw.polygon([150, 200, 250, 120, 350, 200], fill='red', outline='black', width=3)  # Roof
    draw.rectangle([200, 230, 230, 280], fill='brown', outline='black', width=2)  # Door
    draw.ellipse([280, 230, 320, 270], fill='lightblue', outline='black', width=2)  # Window
    
    # Draw sun (separate element)
    draw.ellipse([450, 50, 520, 120], fill='yellow', outline='orange', width=3)
    
    # Draw tree (separate element)
    draw.rectangle([80, 250, 100, 320], fill='brown', outline='black', width=2)  # Trunk
    draw.ellipse([60, 200, 120, 260], fill='green', outline='darkgreen', width=2)  # Leaves
    
    # Draw clouds (separate elements)
    draw.ellipse([100, 60, 180, 100], fill='white', outline='gray')
    draw.ellipse([300, 40, 380, 80], fill='white', outline='gray')
    
    # Draw flowers (small elements) - FIXED COLORS
    draw.ellipse([400, 280, 420, 300], fill='pink', outline='red')
    draw.ellipse([430, 290, 450, 310], fill='purple', outline='indigo')  # Changed from 'darkpurple' to 'indigo'
    
    # Add more distinct elements for better segmentation
    # Draw a car
    draw.rectangle([450, 320, 550, 360], fill='blue', outline='navy', width=2)  # Car body
    draw.ellipse([460, 350, 480, 370], fill='black', outline='black')  # Wheel 1
    draw.ellipse([520, 350, 540, 370], fill='black', outline='black')  # Wheel 2
    
    # Draw a rainbow arc
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    for i, color in enumerate(colors):
        draw.arc([50, 80, 150, 180], start=0, end=180, fill=color, width=3)
    
    img.save('complex_test_drawing.png')
    return 'complex_test_drawing.png'

def test_element_animations():
    """Test element-based animation"""
    print("🧪 Testing Element-Based Animation...")
    print("=" * 50)
    
    try:
        from video_generator import VideoGenerator
        
        # Create test image with multiple elements
        image_path = create_complex_test_image()
        print(f"✅ Test image created: {image_path}")
        
        analysis_results = {
            'input_info': {
                'child_age': 8,
                'drawing_context': 'House Drawing'
            },
            'traditional_analysis': {
                'blip_description': 'a house with a sun, tree, clouds, flowers, and a car',
                'emotional_indicators': {
                    'overall_mood': 'positive'
                }
            }
        }
        
        # Test element-based animation
        vg = VideoGenerator()
        
        print("\n🎨 Testing element-based animation...")
        try:
            result = vg.generate_memory_video(
                image_path,
                analysis_results,
                "Watch as each part of this drawing comes alive!",
                animation_style='elements'
            )
            
            if 'video_path' in result:
                print(f"✅ ELEMENT ANIMATION SUCCESS! Video: {result['video_path']}")
                if os.path.exists(result['video_path']):
                    file_size = os.path.getsize(result['video_path'])
                    print(f"   📁 File size: {file_size / 1024:.1f} KB")
                    print(f"   ⏱️ Duration: {result.get('duration_seconds', 'Unknown')} seconds")
                    print(f"   🎨 Elements found: {result.get('elements_found', 'Unknown')}")
                    print(f"   🎯 Each element should move separately!")
                else:
                    print("⚠️ Video file was created but cannot be found!")
            else:
                print(f"❌ ELEMENT ANIMATION FAILED: {result.get('error', 'Unknown error')}")
                if 'error' in result:
                    print(f"   Error details: {result['error']}")
                    
        except Exception as e:
            print(f"❌ ELEMENT ANIMATION EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"🧹 Cleaned up test image")
        
        print(f"\n🎉 Test complete! Check the video - you should see individual elements moving!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure video_generator.py is in the same directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_element_animations()
