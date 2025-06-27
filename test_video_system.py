#!/usr/bin/env python3

def test_video_system():
    """Test the entire video system like a 10-year-old would"""
    
    print("🧪 Testing the Video System...")
    print("=" * 50)
    
    # Test 1: Check if packages are installed
    print("📦 Checking if all packages are installed...")
    
    try:
        import moviepy
        print("✅ MoviePy is installed!")
    except ImportError:
        print("❌ MoviePy not installed. Run: pip install moviepy")
    
    try:
        import torch
        print("✅ PyTorch is installed!")
        if torch.cuda.is_available():
            print("🚀 CUDA GPU is available for fast AI videos!")
        else:
            print("⚠️ No GPU found. AI videos will be slow but still work!")
    except ImportError:
        print("❌ PyTorch not installed. Run: pip install torch")
    
    try:
        from diffusers import LTXPipeline
        print("✅ HuggingFace Diffusers is installed!")
    except ImportError:
        print("❌ Diffusers not installed. Run: pip install diffusers")
    
    try:
        from selenium import webdriver
        print("✅ Selenium is installed!")
    except ImportError:
        print("❌ Selenium not installed. Run: pip install selenium")
    
    print("\n" + "=" * 50)
    
    # Test 2: Try creating a video
    print("🎬 Testing video creation...")
    
    try:
        from video_generator import VideoGenerator, test_video_generator
        test_video_generator()
        print("✅ Video generation test completed!")
    except Exception as e:
        print(f"❌ Video generation test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Testing complete!")
    print("\nTo run your app:")
    print("1. Open terminal/command prompt")
    print("2. Type: streamlit run app.py")
    print("3. Upload a drawing")
    print("4. Click 'Generate Memory Video'")
    print("5. Watch the magic happen! ✨")

if __name__ == "__main__":
    test_video_system()

