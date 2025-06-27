"""
Configuration settings for Children's Drawing Analysis System
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
VIDEOS_DIR = BASE_DIR / "videos"

# Create directories
for directory in [TEMP_DIR, OUTPUT_DIR, REPORTS_DIR, VIDEOS_DIR]:
    directory.mkdir(exist_ok=True)

# File upload settings
UPLOAD_FOLDER = str(TEMP_DIR)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Analysis settings
DEFAULT_CHILD_AGE = 6
DEFAULT_DRAWING_CONTEXT = "Free Drawing"
DEFAULT_ANALYSIS_TYPE = "Enhanced Analysis"

# AI Model settings
AI_MODELS = {
    'blip': 'Salesforce/blip-image-captioning-base',
    'clip': 'openai/clip-vit-base-patch32',
    'sam': 'facebook/sam-vit-huge'
}

# Video generation settings
VIDEO_SETTINGS = {
    'default_duration': 15,
    'default_fps': 24,
    'default_resolution': (1280, 720),
    'default_style': 'intelligent'
}

# PDF report settings
PDF_SETTINGS = {
    'page_size': 'letter',
    'margins': {
        'top': 72,
        'bottom': 18,
        'left': 72,
        'right': 72
    }
}

# API Keys (loaded from environment)
API_KEYS = {
    'openai': os.getenv('OPENAI_API_KEY'),
    'perplexity': os.getenv('PERPLEXITY_API_KEY'),
    'anthropic': os.getenv('ANTHROPIC_API_KEY'),
    'huggingface': os.getenv('HUGGINGFACE_TOKEN')
}

# Feature flags
FEATURES = {
    'enable_video_generation': True,
    'enable_pdf_reports': True,
    'enable_ai_analysis': True,
    'enable_clinical_assessment': True,
    'enable_scientific_validation': True
}

# Logging configuration
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': str(BASE_DIR / 'app.log')
}