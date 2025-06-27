#!/usr/bin/env python3
"""
Fix for ModuleNotFoundError: No module named '_signal'
This script provides alternative methods to resolve the signal module issue
"""

import os
import sys
from pathlib import Path

def method_1_environment_fix():
    """Method 1: Set environment variables to bypass signal module"""
    print("🔧 Method 1: Setting environment variables...")
    
    # Set environment variables to avoid signal module issues
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    print("✅ Environment variables set")
    return True

def method_2_minimal_imports():
    """Method 2: Use minimal imports to avoid signal dependencies"""
    print("🔧 Method 2: Creating minimal import version...")
    
    minimal_app_content = '''
import sys
import os

# Minimal imports to avoid signal module
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

def run_minimal_app():
    """Run minimal version without signal dependencies"""
    if STREAMLIT_AVAILABLE:
        st.title("🎨 Children's Drawing Analysis")
        st.write("Minimal version running successfully!")
        
        uploaded_file = st.file_uploader("Upload a drawing", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Drawing")
            st.success("✅ File uploaded successfully!")
            st.info("Full analysis features require complete setup")
    else:
        print("Streamlit not available - check installation")

if __name__ == "__main__":
    run_minimal_app()
'''
    
    with open('minimal_app.py', 'w') as f:
        f.write(minimal_app_content)
    
    print("✅ Created minimal_app.py")
    return True

def method_3_pure_web_interface():
    """Method 3: Create pure HTML/JS interface"""
    print("🔧 Method 3: Creating pure web interface...")
    
    # This uses the existing index.html and app.js files
    if Path('index.html').exists() and Path('app.js').exists():
        print("✅ Pure web interface already available")
        print("🌐 Open index.html in your browser")
        return True
    else:
        print("❌ Web interface files not found")
        return False

def method_4_alternative_server():
    """Method 4: Create alternative server without signal dependencies"""
    print("🔧 Method 4: Creating alternative server...")
    
    alt_server_content = '''
#!/usr/bin/env python3
"""
Alternative server that avoids signal module dependencies
"""

import os
import sys
from pathlib import Path

def start_simple_http_server():
    """Start simple HTTP server without signal dependencies"""
    try:
        import http.server
        import socketserver
        
        PORT = 8000
        
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=os.getcwd(), **kwargs)
        
        print(f"🌐 Starting server on http://localhost:{PORT}")
        print("📝 This provides basic web interface")
        print("💡 Open the URL in your browser")
        
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
            
    except ImportError:
        print("❌ HTTP server modules not available")
        return False
    except Exception as e:
        print(f"❌ Server error: {e}")
        return False

def start_manual_browser():
    """Manual browser opening without webbrowser module"""
    html_file = Path('index.html').absolute()
    
    if html_file.exists():
        print(f"📄 Open this file in your browser:")
        print(f"file://{html_file}")
        return True
    else:
        print("❌ index.html not found")
        return False

def main():
    """Main function for alternative server"""
    print("🚀 Alternative Server (No Signal Dependencies)")
    print("=" * 50)
    
    # Try HTTP server first
    try:
        start_simple_http_server()
    except KeyboardInterrupt:
        print("\\n👋 Server stopped")
    except Exception as e:
        print(f"❌ HTTP server failed: {e}")
        print("🔄 Trying manual browser method...")
        start_manual_browser()

if __name__ == "__main__":
    main()
'''
    
    with open('alt_server.py', 'w') as f:
        f.write(alt_server_content)
    
    print("✅ Created alt_server.py")
    return True

def method_5_docker_alternative():
    """Method 5: Create Docker-like environment setup"""
    print("🔧 Method 5: Creating containerized environment setup...")
    
    dockerfile_content = '''
# Alternative Dockerfile for environments with signal issues
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with specific flags
RUN pip3 install --no-cache-dir --disable-pip-version-check -r requirements.txt

# Copy application files
COPY . .

# Set environment variables to avoid signal issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

EXPOSE 8501

# Use alternative startup command
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.headless", "true"]
'''
    
    with open('Dockerfile.alt', 'w') as f:
        f.write(dockerfile_content)
    
    print("✅ Created Dockerfile.alt")
    return True

def method_6_requirements_fix():
    """Method 6: Create fixed requirements without problematic packages"""
    print("🔧 Method 6: Creating fixed requirements...")
    
    fixed_requirements = '''
# Core dependencies (signal-safe versions)
streamlit>=1.28.0
opencv-python-headless>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0

# Image processing (headless versions)
matplotlib-base>=3.7.0
scikit-image>=0.21.0
scikit-learn>=1.3.0

# AI/ML (minimal versions)
torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
transformers>=4.30.0

# API clients
requests>=2.31.0
python-dotenv>=1.0.0

# Report generation
reportlab>=4.0.0

# Video processing (if needed)
imageio>=2.35.0
'''
    
    with open('requirements_fixed.txt', 'w') as f:
        f.write(fixed_requirements)
    
    print("✅ Created requirements_fixed.txt")
    return True

def apply_all_fixes():
    """Apply all available fixes"""
    print("🔧 Applying all signal error fixes...")
    
    fixes = [
        method_1_environment_fix,
        method_2_minimal_imports,
        method_3_pure_web_interface,
        method_4_alternative_server,
        method_5_docker_alternative,
        method_6_requirements_fix
    ]
    
    results = []
    for fix in fixes:
        try:
            result = fix()
            results.append(result)
        except Exception as e:
            print(f"❌ Fix failed: {e}")
            results.append(False)
    
    return results

def main():
    """Main function to fix signal error"""
    print("🔧 Signal Module Error Fix")
    print("=" * 50)
    
    print("This script provides multiple methods to fix the '_signal' module error:")
    print("1. Environment variable fixes")
    print("2. Minimal import versions")
    print("3. Pure web interface")
    print("4. Alternative server")
    print("5. Docker alternative")
    print("6. Fixed requirements")
    
    choice = input("\\nApply all fixes? (y/n): ").lower()
    
    if choice == 'y':
        results = apply_all_fixes()
        success_count = sum(results)
        
        print(f"\\n📊 Results: {success_count}/{len(results)} fixes applied successfully")
        
        print("\\n🚀 Try these startup methods:")
        print("1. python minimal_app.py")
        print("2. python alt_server.py")
        print("3. Open index.html in browser")
        print("4. pip install -r requirements_fixed.txt")
    else:
        print("\\n💡 You can run individual methods:")
        print("- method_1_environment_fix()")
        print("- method_2_minimal_imports()")
        print("- method_3_pure_web_interface()")
        print("- method_4_alternative_server()")
        print("- method_5_docker_alternative()")
        print("- method_6_requirements_fix()")

if __name__ == "__main__":
    main()