"""
Utility functions for the Children's Drawing Analysis System
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import shutil

from config import TEMP_DIR, OUTPUT_DIR, LOGGING

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING['level']),
    format=LOGGING['format'],
    handlers=[
        logging.FileHandler(LOGGING['file']),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_directories():
    """Ensure all required directories exist"""
    directories = [TEMP_DIR, OUTPUT_DIR, "reports", "videos"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Directory ensured: {directory}")

def clean_temp_files(max_age_hours: int = 24):
    """Clean up temporary files older than specified hours"""
    try:
        current_time = datetime.now()
        cleaned_count = 0
        
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file():
                file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.total_seconds() > max_age_hours * 3600:
                    file_path.unlink()
                    cleaned_count += 1
        
        logger.info(f"Cleaned {cleaned_count} temporary files")
        return cleaned_count
    
    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")
        return 0

def save_analysis_results(results: Dict[Any, Any], filename: Optional[str] = None) -> str:
    """Save analysis results to JSON file"""
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_results_{timestamp}.json"
        
        filepath = OUTPUT_DIR / filename
        
        # Convert non-serializable objects to strings
        serializable_results = make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Analysis results saved: {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"Error saving analysis results: {e}")
        raise

def load_analysis_results(filepath: str) -> Dict[Any, Any]:
    """Load analysis results from JSON file"""
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Analysis results loaded: {filepath}")
        return results
    
    except Exception as e:
        logger.error(f"Error loading analysis results: {e}")
        raise

def make_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format"""
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)

def validate_image_file(file_path: str) -> bool:
    """Validate that file is a valid image"""
    try:
        from PIL import Image
        
        with Image.open(file_path) as img:
            img.verify()
        
        return True
    
    except Exception as e:
        logger.error(f"Invalid image file {file_path}: {e}")
        return False

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception:
        return 0.0

def create_temp_file(suffix: str = ".tmp") -> str:
    """Create a temporary file and return its path"""
    try:
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=suffix, 
            dir=TEMP_DIR
        )
        temp_file.close()
        return temp_file.name
    
    except Exception as e:
        logger.error(f"Error creating temp file: {e}")
        raise

def copy_file_to_temp(source_path: str, suffix: str = None) -> str:
    """Copy file to temp directory and return new path"""
    try:
        if suffix is None:
            suffix = Path(source_path).suffix
        
        temp_path = create_temp_file(suffix)
        shutil.copy2(source_path, temp_path)
        
        logger.info(f"File copied to temp: {source_path} -> {temp_path}")
        return temp_path
    
    except Exception as e:
        logger.error(f"Error copying file to temp: {e}")
        raise

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def get_age_group(age: int) -> str:
    """Get age group classification"""
    if age < 4:
        return "Toddler (2-3 years)"
    elif age < 7:
        return "Preschool (4-6 years)"
    elif age < 12:
        return "School Age (7-11 years)"
    else:
        return "Adolescent (12+ years)"

def check_system_resources() -> Dict[str, Any]:
    """Check system resources and capabilities"""
    import psutil
    import torch
    
    resources = {
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'disk_free_gb': psutil.disk_usage('.').free / (1024**3),
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if resources['gpu_available']:
        resources['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return resources

def estimate_processing_time(image_size_mb: float, analysis_type: str, has_gpu: bool = False) -> float:
    """Estimate processing time based on image size and analysis type"""
    base_time = {
        'Basic Analysis': 10,
        'Enhanced Analysis': 30,
        'Scientific Validation': 45,
        'Clinical Assessment': 60,
        'AI Multi-Model': 90,
        'Complete Analysis': 120
    }
    
    time_estimate = base_time.get(analysis_type, 30)
    
    # Adjust for image size
    time_estimate *= (1 + image_size_mb / 10)
    
    # Adjust for GPU availability
    if has_gpu:
        time_estimate *= 0.6
    
    return time_estimate

def create_progress_callback(total_steps: int):
    """Create a progress callback function"""
    current_step = [0]  # Use list to allow modification in nested function
    
    def update_progress(step_name: str = None):
        current_step[0] += 1
        progress = current_step[0] / total_steps
        
        if step_name:
            logger.info(f"Progress: {progress:.1%} - {step_name}")
        
        return progress
    
    return update_progress

# Initialize directories on import
setup_directories()