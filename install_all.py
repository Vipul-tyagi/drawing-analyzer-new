#!/usr/bin/env python3
"""
Complete installation script for Children's Drawing Analysis System
Installs all dependencies and sets up the environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_package(package, description=""):
    """Install a single package with error handling"""
    try:
        print(f"📦 Installing {package}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package, "--upgrade"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}")
        if e.stderr:
            print(f"   Error: {e.stderr.decode()}")
        return False

def install_core_dependencies():
    """Install core dependencies"""
    print("\n📦 Installing core dependencies...")
    
    core_packages = [
        ("streamlit", "Web interface framework"),
        ("opencv-python", "Computer vision library"),
        ("pillow", "Image processing library"),
        ("numpy", "Numerical computing library"),
        ("scipy", "Scientific computing library"),
        ("pandas", "Data manipulation library"),
        ("matplotlib", "Plotting library"),
        ("scikit-learn", "Machine learning library"),
        ("scikit-image", "Image processing library"),
    ]
    
    success_count = 0
    for package, description in core_packages:
        if install_package(package, description):
            success_count += 1
    
    print(f"\n✅ Core dependencies: {success_count}/{len(core_packages)} installed")
    return success_count == len(core_packages)

def install_ai_dependencies():
    """Install AI/ML dependencies"""
    print("\n🤖 Installing AI/ML dependencies...")
    
    ai_packages = [
        ("torch", "PyTorch deep learning framework"),
        ("torchvision", "PyTorch computer vision"),
        ("transformers", "HuggingFace transformers"),
        ("diffusers", "HuggingFace diffusers"),
        ("accelerate", "HuggingFace acceleration"),
    ]
    
    success_count = 0
    for package, description in ai_packages:
        if install_package(package, description):
            success_count += 1
    
    print(f"\n✅ AI dependencies: {success_count}/{len(ai_packages)} installed")
    return success_count >= 3  # At least torch, torchvision, transformers

def install_video_dependencies():
    """Install video generation dependencies"""
    print("\n🎬 Installing video generation dependencies...")
    
    video_packages = [
        ("moviepy", "Video editing library"),
        ("imageio", "Image I/O library"),
        ("imageio-ffmpeg", "FFmpeg for imageio"),
    ]
    
    success_count = 0
    for package, description in video_packages:
        if install_package(package, description):
            success_count += 1
    
    print(f"\n✅ Video dependencies: {success_count}/{len(video_packages)} installed")
    return success_count >= 2

def install_report_dependencies():
    """Install report generation dependencies"""
    print("\n📄 Installing report generation dependencies...")
    
    report_packages = [
        ("reportlab", "PDF generation library"),
        ("plotly", "Interactive plotting library"),
        ("seaborn", "Statistical visualization"),
    ]
    
    success_count = 0
    for package, description in report_packages:
        if install_package(package, description):
            success_count += 1
    
    print(f"\n✅ Report dependencies: {success_count}/{len(report_packages)} installed")
    return success_count >= 1

def install_api_dependencies():
    """Install API integration dependencies"""
    print("\n🔌 Installing API integration dependencies...")
    
    api_packages = [
        ("openai", "OpenAI API client"),
        ("requests", "HTTP library"),
        ("python-dotenv", "Environment variable loader"),
    ]
    
    success_count = 0
    for package, description in api_packages:
        if install_package(package, description):
            success_count += 1
    
    print(f"\n✅ API dependencies: {success_count}/{len(api_packages)} installed")
    return success_count >= 2

def install_optional_dependencies():
    """Install optional advanced dependencies"""
    print("\n🔬 Installing optional advanced dependencies...")
    
    optional_packages = [
        ("clip-by-openai", "CLIP model for image understanding"),
        ("selenium", "Web automation for advanced features"),
        ("webdriver-manager", "WebDriver management"),
        ("psutil", "System monitoring"),
    ]
    
    success_count = 0
    for package, description in optional_packages:
        if install_package(package, description):
            success_count += 1
    
    print(f"\n✅ Optional dependencies: {success_count}/{len(optional_packages)} installed")
    return True  # Optional, so always return True

def setup_environment_file():
    """Create .env file for API keys"""
    print("\n🔧 Setting up environment file...")
    
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    env_content = """# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Perplexity Configuration  
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Optional: Other providers
ANTHROPIC_API_KEY=your_anthropic_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here

# System Configuration
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
"""
    
    try:
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ .env file created successfully")
        print("📝 Please edit .env file with your actual API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "temp",
        "outputs", 
        "reports",
        "videos",
        "temp_uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}/")
    
    return True

def check_system_requirements():
    """Check system requirements"""
    print("\n🔍 Checking system requirements...")
    
    # Check FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg not found - video generation may not work")
        print("💡 Install FFmpeg:")
        
        system = platform.system().lower()
        if system == "darwin":  # macOS
            print("   brew install ffmpeg")
        elif system == "linux":
            print("   sudo apt-get install ffmpeg  # Ubuntu/Debian")
            print("   sudo yum install ffmpeg      # CentOS/RHEL")
        elif system == "windows":
            print("   Download from https://ffmpeg.org/download.html")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"✅ Available memory: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            print("⚠️ Low memory - some AI features may be slow")
        elif memory_gb >= 8:
            print("🚀 Good memory for AI processing")
    except ImportError:
        print("⚠️ Cannot check memory (psutil not available)")
    
    return True

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    # Test core imports
    test_imports = [
        ("streamlit", "Streamlit web framework"),
        ("cv2", "OpenCV computer vision"),
        ("PIL", "Pillow image processing"),
        ("numpy", "NumPy numerical computing"),
        ("torch", "PyTorch deep learning"),
        ("transformers", "HuggingFace transformers"),
    ]
    
    success_count = 0
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"✅ {module} - {description}")
            success_count += 1
        except ImportError:
            print(f"❌ {module} - {description}")
    
    print(f"\n✅ Import test: {success_count}/{len(test_imports)} successful")
    
    # Test GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ No GPU available - using CPU (slower)")
    except ImportError:
        print("⚠️ Cannot check GPU availability")
    
    return success_count >= 4

def main():
    """Main installation function"""
    print("🚀 Children's Drawing Analysis System - Complete Installation")
    print("=" * 80)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Installation aborted - incompatible Python version")
        return False
    
    # Upgrade pip
    print("\n📦 Upgrading pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip upgraded successfully")
    except subprocess.CalledProcessError:
        print("⚠️ pip upgrade failed, continuing...")
    
    # Install dependencies
    results = {
        'core': install_core_dependencies(),
        'ai': install_ai_dependencies(),
        'video': install_video_dependencies(),
        'reports': install_report_dependencies(),
        'api': install_api_dependencies(),
        'optional': install_optional_dependencies(),
    }
    
    # Setup environment
    setup_environment_file()
    create_directories()
    check_system_requirements()
    
    # Test installation
    test_success = test_installation()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Installation Summary:")
    
    for category, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {category.title()} dependencies")
    
    if test_success:
        print("  ✅ Installation test passed")
    else:
        print("  ⚠️ Installation test had issues")
    
    # Next steps
    print("\n🎯 Next Steps:")
    
    if all(results.values()) and test_success:
        print("1. ✅ All dependencies installed successfully!")
        print("2. 📝 Edit .env file with your API keys (optional)")
        print("3. 🚀 Run the application:")
        print("   python main.py")
        print("   OR")
        print("   streamlit run app.py")
    else:
        print("1. ⚠️ Some dependencies failed to install")
        print("2. 🔧 Check error messages above")
        print("3. 💡 Try installing failed packages manually")
        print("4. 🚀 You can still run basic features:")
        print("   python main.py")
    
    print("\n📚 Documentation: README.md")
    print("🆘 Support: Check console output for specific errors")
    
    return all(results.values()) and test_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)