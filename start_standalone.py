#!/usr/bin/env python3
"""
Simple startup script for the standalone system
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if minimal dependencies are available"""
    try:
        import cv2
        import numpy as np
        from PIL import Image
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install minimal dependencies"""
    print("📦 Installing minimal dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "opencv-python", "pillow", "numpy"
        ])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main startup function"""
    print("🎨 Starting Standalone Drawing Analysis System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Missing dependencies")
        if input("📦 Install now? (y/n): ").lower() == 'y':
            if not install_dependencies():
                print("❌ Installation failed")
                return False
        else:
            print("💡 Install manually: pip install opencv-python pillow numpy")
            return False
    
    # Check if standalone app exists
    if not Path("standalone_app.py").exists():
        print("❌ standalone_app.py not found")
        return False
    
    # Start the application
    try:
        print("🚀 Starting standalone application...")
        subprocess.run([sys.executable, "standalone_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()