from video_generator_consolidated import VideoGenerator
import json

# Test with sample data
vg = VideoGenerator()

# Mock analysis results
analysis_results = {
    "input_info": {"child_age": 7},
    "traditional_analysis": {"blip_description": "a colorful drawing"}
}

# Test each animation type
for anim_type in ["intelligent", "elements", "particle", "floating", "standard"]:
    print(f"Testing {anim_type}...")
    result = vg.create_animation_video(
        "test_image.jpg",  # Replace with actual image
        analysis_results,
        animation_type=anim_type
    )
    print(f"Result: {result.get('generation_method', 'Failed')}")

