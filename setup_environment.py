#!/usr/bin/env python3
"""
Environment setup script for Children's Drawing Analysis System
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Requirements installed successfully!")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def setup_environment_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("📝 Creating .env file...")
    
    env_content = """# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Perplexity Configuration  
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Optional: Other providers
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional: HuggingFace Token
HUGGINGFACE_TOKEN=your_huggingface_token_here
"""
    
    try:
        with open(env_file, "w") as f:
            f.write(env_content)
        
        print("✅ .env file created successfully!")
        print("📝 Please edit .env file with your actual API keys")
        return True
    
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def check_system_dependencies():
    """Check for system-level dependencies"""
    print("🔍 Checking system dependencies...")
    
    # Check for FFmpeg (needed for video generation)
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
    
    return True

def test_imports():
    """Test critical imports"""
    print("🧪 Testing critical imports...")
    
    critical_imports = [
        "streamlit",
        "PIL",
        "cv2",
        "numpy",
        "torch",
        "transformers"
    ]
    
    failed_imports = []
    
    for module in critical_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️ Failed imports: {', '.join(failed_imports)}")
        print("💡 Try running: pip install -r requirements.txt")
        return False
    
    print("✅ All critical imports successful!")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "temp",
        "outputs",
        "reports",
        "videos"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}/")
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Children's Drawing Analysis System")
    print("=" * 60)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install requirements
    if success and not install_requirements():
        success = False
    
    # Setup environment
    if success and not setup_environment_file():
        success = False
    
    # Check system dependencies
    if success and not check_system_dependencies():
        # Don't fail for system dependencies
        pass
    
    # Test imports
    if success and not test_imports():
        success = False
    
    # Create directories
    if success and not create_directories():
        success = False
    
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit .env file with your API keys")
        print("2. Run: streamlit run app.py")
        print("3. Open your browser to the displayed URL")
        print("\n💡 For video generation, make sure FFmpeg is installed")
    else:
        print("❌ Setup failed!")
        print("Please check the error messages above and try again")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)