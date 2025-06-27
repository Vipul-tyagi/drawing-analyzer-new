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
    """Install all video generation packages"""
    print("🎬 Installing video generation packages...")
    
    packages = [
        "moviepy",
        "diffusers",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "selenium",
        "webdriver-manager"
    ]
    
    for package in packages:
        print(f"📦 Installing {package}...")
        install_package(package)
    
    print("✅ All video packages installed!")

if __name__ == "__main__":
    main()

