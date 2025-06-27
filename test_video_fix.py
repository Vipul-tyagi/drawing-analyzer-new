#!/usr/bin/env python3
"""
Test script to verify video generation is working - WITH FFMPEG FIX
"""

import os
import sys

def fix_ffmpeg():
    """Automatically fix FFmpeg issues"""
    try:
        import imageio_ffmpeg
        # This will download FFmpeg if not available
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg_path
        print(f"✅ FFmpeg configured at: {ffmpeg_path}")
        return True
    except Exception as e:
        print(f"❌ Could not configure FFmpeg: {e}")
        return False

def test_video_generation():
    """Test the video generation system"""
    print("🧪 Testing Video Generation Fix...")
    print("=" * 50)
    
    # Step 1: Fix FFmpeg first
    print("🔧 Configuring FFmpeg...")
    if not fix_ffmpeg():
        print("❌ FFmpeg configuration failed. Please install FFmpeg manually.")
        return False
    
    # Step 2: Test imports
    try:
        from video_generator import VideoGenerator, test_video_generator
        print("✅ Video generator imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Step 3: Run system test
    test_video_generator()
    
    # Step 4: Create test data
    print("\n📸 Creating test image...")
    from PIL import Image, ImageDraw
    
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 150, 300, 250], fill='blue', outline='black')
    draw.polygon([100, 150, 200, 100, 300, 150], fill='red', outline='black')
    img.save('test_drawing.png')
    
    analysis_results = {
        'input_info': {'child_age': 6, 'drawing_context': 'Free Drawing'},
        'traditional_analysis': {
            'blip_description': 'a colorful drawing of a house',
            'emotional_indicators': {'overall_mood': 'positive'}
        }
    }
    
    # Step 5: Test video generation
    print("\n🎬 Testing video generation...")
    vg = VideoGenerator()
    
    result = vg.generate_memory_video(
        'test_drawing.png',
        analysis_results,
        "This is a test video to verify the fix is working!"
    )
    
    # Step 6: Check results
    if 'video_path' in result:
        print(f"✅ SUCCESS! Video created: {result['video_path']}")
        print(f"📊 Method used: {result['generation_method']}")
        
        if os.path.exists(result['video_path']):
            file_size = os.path.getsize(result['video_path'])
            print(f"📁 File size: {file_size / 1024:.1f} KB")
            print(f"🎯 Video file exists and is ready to play!")
        
        # Cleanup
        os.remove('test_drawing.png')
        return True
    else:
        print("❌ FAILED! Video generation unsuccessful")
        print(f"Error: {result.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    success = test_video_generation()
    
    if success:
        print("\n🎉 VIDEO GENERATION FIX SUCCESSFUL!")
        print("Your app should now work properly.")
    else:
        print("\n❌ VIDEO GENERATION STILL HAS ISSUES")
        print("Try installing FFmpeg manually with: brew install ffmpeg")
