import torch
import torchvision.transforms as transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DrawingAnalyzer:
    def __init__(self):
        """
        This is like the brain of our app - it sets up all the AI models
        """
        print("🤖 Loading AI models... This might take a few minutes the first time!")
        
        # Load the BLIP model (this can understand pictures and describe them)
        self.blip_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        self.blip_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
        
        print("✅ AI models loaded successfully!")

    def _determine_age_group(self, age):
        """
        Figure out what age group a child belongs to
        """
        if age < 4: 
            return "Toddler (2-3 years)"
        elif age < 7: 
            return "Preschool (4-6 years)"
        elif age < 12: 
            return "School Age (7-11 years)"
        else: 
            return "Adolescent (12+ years)"

    def _analyze_colors(self, image):
        """
        Look at what colors are used in the drawing
        """
        # Convert image to numpy array if it isn't already
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        # Calculate average colors
        if len(img_array.shape) == 3:  # Color image
            avg_red = np.mean(img_array[:,:,0])
            avg_green = np.mean(img_array[:,:,1])
            avg_blue = np.mean(img_array[:,:,2])
            brightness = np.mean(img_array)
            
            # Determine dominant color
            if avg_red > avg_green and avg_red > avg_blue:
                dominant_color = "Red"
            elif avg_green > avg_red and avg_green > avg_blue:
                dominant_color = "Green"
            elif avg_blue > avg_red and avg_blue > avg_green:
                dominant_color = "Blue"
            else:
                dominant_color = "Mixed colors"
                
        else:  # Grayscale image
            brightness = np.mean(img_array)
            dominant_color = "Grayscale"
            
        return {
            'dominant_color': dominant_color,
            'brightness_level': float(brightness),
            'color_richness': 'Rich' if brightness > 128 else 'Subtle'
        }

    def _analyze_emotional_indicators(self, image, caption):
        """
        Try to understand the emotions in the drawing
        """
        # Analyze colors for emotional clues
        color_analysis = self._analyze_colors(image)
        
        # Look for emotional words in the AI's description
        positive_words = ['happy', 'smiling', 'bright', 'colorful', 'playing', 'sunny']
        negative_words = ['sad', 'dark', 'crying', 'alone', 'scared', 'angry']
        
        positive_score = sum(1 for word in positive_words if word in caption.lower())
        negative_score = sum(1 for word in negative_words if word in caption.lower())
        
        if positive_score > negative_score:
            emotional_tone = "Positive"
        elif negative_score > positive_score:
            emotional_tone = "Concerning"
        else:
            emotional_tone = "Neutral"
            
        return {
            'emotional_tone': emotional_tone,
            'positive_indicators': positive_score,
            'negative_indicators': negative_score,
            'color_mood': 'Bright and cheerful' if color_analysis['brightness_level'] > 128 else 'Calm and subdued'
        }

    def _generate_recommendations(self, age_group, emotional_analysis):
        """
        Give helpful suggestions based on what we found
        """
        recommendations = []
        
        # Age-appropriate recommendations
        if "Toddler" in age_group:
            recommendations.append("Encourage more drawing time - great for developing fine motor skills!")
            recommendations.append("Try finger paints or chunky crayons for easier grip")
        elif "Preschool" in age_group:
            recommendations.append("Ask the child to tell stories about their drawings")
            recommendations.append("Provide various art materials to explore creativity")
        elif "School Age" in age_group:
            recommendations.append("Encourage drawing from observation (flowers, pets, etc.)")
            recommendations.append("Consider art classes if the child shows strong interest")
        else:
            recommendations.append("Support artistic expression as a healthy outlet")
            recommendations.append("Discuss the meaning behind their artwork")
        
        # Emotional-based recommendations
        if emotional_analysis['emotional_tone'] == "Concerning":
            recommendations.append("⚠️ Consider talking with the child about their feelings")
            recommendations.append("⚠️ If concerns persist, consult with a counselor or teacher")
        elif emotional_analysis['emotional_tone'] == "Positive":
            recommendations.append("✨ Great emotional expression! Keep encouraging creativity")
        
        return recommendations

    def analyze_drawing(self, image_path, child_age, drawing_context="Free Drawing"):
        """
        This is the main function that analyzes a child's drawing
        """
        print(f"🎨 Analyzing drawing for {child_age}-year-old child...")
        
        try:
            # Open and prepare the image
            image = Image.open(image_path).convert('RGB')
            age_group = self._determine_age_group(child_age)
            
            print("🤖 AI is looking at the drawing...")
            
            # Get AI description of the drawing
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            print("🎯 Analyzing emotions and colors...")
            
            # Analyze emotions and colors
            emotional_analysis = self._analyze_emotional_indicators(image, caption)
            color_analysis = self._analyze_colors(image)
            
            print("💡 Generating recommendations...")
            
            # Generate recommendations
            recommendations = self._generate_recommendations(age_group, emotional_analysis)
            
            # Put it all together
            results = {
                'visual_analysis': {
                    'ai_description': caption,
                    'confidence': 0.85,  # Placeholder confidence score
                    'color_analysis': color_analysis
                },
                'developmental_assessment': {
                    'age_group': age_group,
                    'drawing_context': drawing_context,
                    'developmental_stage': 'Appropriate for age' if child_age >= 3 else 'Early development'
                },
                'emotional_analysis': emotional_analysis,
                'recommendations': recommendations,
                'summary': f"This drawing by a {child_age}-year-old shows {emotional_analysis['emotional_tone'].lower()} emotional indicators. The AI describes it as: '{caption}'"
            }
            
            print("✅ Analysis complete!")
            return results
            
        except Exception as e:
            print(f"❌ Error analyzing drawing: {str(e)}")
            return None

# Test function to make sure everything works
def test_analyzer():
    """
    Test our analyzer with a simple colored image
    """
    print("🧪 Testing the Drawing Analyzer...")
    
    # Create a simple test image (a red square)
    test_image = Image.new('RGB', (200, 200), color='red')
    test_image.save('test_drawing.png')
    
    # Create analyzer and test it
    analyzer = DrawingAnalyzer()
    results = analyzer.analyze_drawing('test_drawing.png', 6, "Test Drawing")
    
    if results:
        print("\n🎉 Test successful! Here's what the AI found:")
        print(f"Description: {results['visual_analysis']['ai_description']}")
        print(f"Emotional tone: {results['emotional_analysis']['emotional_tone']}")
        print(f"Age group: {results['developmental_assessment']['age_group']}")
    else:
        print("❌ Test failed!")

if __name__ == "__main__":
    test_analyzer()
