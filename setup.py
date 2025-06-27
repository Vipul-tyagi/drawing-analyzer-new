import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Packages installed successfully!")

def setup_environment():
    """Setup environment file"""
    env_file = Path(".env")
    if not env_file.exists():
        print("🔧 Creating .env file...")
        with open(".env", "w") as f:
            f.write("""# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Perplexity Configuration  
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Optional: Other providers
ANTHROPIC_API_KEY=your_anthropic_key_here
""")
        print("📝 Please edit .env file with your actual API keys")
    else:
        print("✅ .env file already exists")

def main():
    print("🚀 Setting up Children's Drawing Analysis System...")
    install_requirements()
    setup_environment()
    print("\n✅ Setup complete!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: streamlit run app.py")

if __name__ == "__main__":
    main()

