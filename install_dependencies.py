#!/usr/bin/env python3

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install missing packages for PDF generation"""
    print("🔧 Installing dependencies for PDF generation...")
    
    packages = [
        "reportlab",
        "pillow",
        "python-dotenv"
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            install_package(package)
    
    print("✅ All dependencies checked/installed!")

if __name__ == "__main__":
    main()

