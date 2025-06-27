import os
import openai
import requests
import json
import base64
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from PIL import Image
import io
import logging

@dataclass
class LLMResponse:
    provider: str
    response: str
    confidence: float
    analysis_type: str
    metadata: Dict

class StreamlinedLLMIntegrator:
    """Enhanced LLM integrator with OpenAI and Perplexity support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.available_providers = []
        self._setup_providers()
    
    def _setup_providers(self):
        """Initialize available LLM providers"""
        # OpenAI setup
        if os.getenv('OPENAI_API_KEY'):
            self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.available_providers.append('openai')
            self.logger.info("✅ OpenAI provider initialized")
        
        # Perplexity setup
        if os.getenv('PERPLEXITY_API_KEY'):
            self.perplexity_client = openai.OpenAI(
                api_key=os.getenv('PERPLEXITY_API_KEY'),
                base_url="https://api.perplexity.ai"
            )
            self.available_providers.append('perplexity')
            self.logger.info("✅ Perplexity provider initialized")
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for API calls"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _create_psychological_prompt(self, child_age: int, context: str, ml_results: Dict) -> str:
        """Create specialized prompt for psychological analysis"""
        return f"""
You are a licensed child psychologist specializing in art therapy and developmental assessment.
Analyze this {context} drawing by a {child_age}-year-old child.

TECHNICAL ANALYSIS AVAILABLE:
- Visual Description: {ml_results.get('blip_description', 'A child\'s drawing')}
- Color Analysis: {ml_results.get('color_analysis', {})}
- Shape Complexity: {ml_results.get('shape_analysis', {})}
- Spatial Organization: {ml_results.get('spatial_analysis', {})}

PROVIDE COMPREHENSIVE ANALYSIS:

## DEVELOPMENTAL ASSESSMENT
- Age-appropriateness of drawing skills and cognitive markers
- Fine motor development indicators
- Symbolic thinking progression for {child_age}-year-old

## EMOTIONAL WELLBEING
- Current emotional state indicators from visual elements
- Mood and affect assessment based on colors and content
- Signs of emotional distress or resilience

## PSYCHOLOGICAL INDICATORS
- Social connection and relationship patterns
- Self-concept and identity development markers
- Coping mechanisms and stress indicators

## CLINICAL CONSIDERATIONS
- Any concerning patterns requiring professional attention
- Trauma indicators or attachment concerns (if present)
- Risk assessment and protective factors

## RECOMMENDATIONS
- Specific developmental support strategies
- Environmental modifications and enrichment activities
- When to seek additional professional consultation

Focus on evidence-based interpretation while being sensitive to cultural and individual differences.
"""

    def analyze_with_openai_vision(self, image_path: str, child_age: int, context: str, ml_results: Dict) -> LLMResponse:
        """Analyze drawing using OpenAI Vision API"""
        try:
            base64_image = self._encode_image_to_base64(image_path)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a licensed child psychologist specializing in art therapy and developmental assessment."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self._create_psychological_prompt(child_age, context, ml_results)
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return LLMResponse(
                provider="openai_vision",
                response=response.choices[0].message.content,
                confidence=0.85,
                analysis_type="vision_psychological",
                metadata={"model": "gpt-4-vision-preview", "tokens": response.usage.total_tokens}
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI Vision analysis failed: {e}")
            return self._fallback_analysis("openai_vision", child_age, context, ml_results)
    
    def analyze_with_openai_text(self, child_age: int, context: str, ml_results: Dict) -> LLMResponse:
        """Analyze drawing using OpenAI text model"""
        try:
            prompt = self._create_psychological_prompt(child_age, context, ml_results)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a licensed child psychologist specializing in art therapy."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.3
            )
            
            return LLMResponse(
                provider="openai_text",
                response=response.choices[0].message.content,
                confidence=0.80,
                analysis_type="text_psychological",
                metadata={"model": "gpt-4", "tokens": response.usage.total_tokens}
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI text analysis failed: {e}")
            return self._fallback_analysis("openai_text", child_age, context, ml_results)
    
    def analyze_with_perplexity(self, child_age: int, context: str, ml_results: Dict) -> LLMResponse:
        """Analyze drawing using Perplexity for research-backed insights"""
        try:
            research_prompt = f"""
Based on current research in child psychology and art therapy, analyze this {context} drawing by a {child_age}-year-old:

Drawing Analysis: {ml_results.get('blip_description', 'A child\'s drawing')}
Technical Details: Colors show {ml_results.get('color_analysis', {}).get('dominant_color', 'mixed colors')}, 
Complexity: {ml_results.get('shape_analysis', {}).get('complexity_level', 'medium')}

Provide evidence-based analysis including:
1. Current research on {child_age}-year-old developmental milestones in art
2. Evidence-based interpretation of observed elements
3. Research-supported recommendations for development
4. Latest findings on emotional expression in children's drawings
5. Cultural considerations in art assessment

Include specific research citations and current best practices.
"""
            
            response = self.perplexity_client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=[
                    {"role": "system", "content": "You are a research-focused child psychology expert with access to current literature."},
                    {"role": "user", "content": research_prompt}
                ],
                max_tokens=1200,
                temperature=0.2
            )
            
            return LLMResponse(
                provider="perplexity",
                response=response.choices[0].message.content,
                confidence=0.88,
                analysis_type="research_backed",
                metadata={"model": "llama-3.1-sonar-large-128k-online", "search_enabled": True}
            )
            
        except Exception as e:
            self.logger.error(f"Perplexity analysis failed: {e}")
            return self._fallback_analysis("perplexity", child_age, context, ml_results)
    
    def analyze_drawing_comprehensive(self, image, child_age: int, context: str, ml_results: Dict) -> List[LLMResponse]:
        """Comprehensive analysis using all available providers"""
        responses = []
        
        # Determine image path
        if hasattr(image, 'filename'):
            image_path = image.filename
        elif isinstance(image, str):
            image_path = image
        else:
            # Save PIL image temporarily
            temp_path = f"temp_drawing_{child_age}.jpg"
            if hasattr(image, 'save'):
                image.save(temp_path)
                image_path = temp_path
            else:
                self.logger.warning("Could not determine image path for vision analysis")
                image_path = None
        
        # OpenAI Vision Analysis (if image path available)
        if 'openai' in self.available_providers and image_path:
            responses.append(self.analyze_with_openai_vision(image_path, child_age, context, ml_results))
        
        # OpenAI Text Analysis
        if 'openai' in self.available_providers:
            responses.append(self.analyze_with_openai_text(child_age, context, ml_results))
        
        # Perplexity Research Analysis
        if 'perplexity' in self.available_providers:
            responses.append(self.analyze_with_perplexity(child_age, context, ml_results))
        
        # Clean up temporary file
        if image_path and image_path.startswith("temp_drawing_"):
            try:
                os.remove(image_path)
            except:
                pass
        
        return responses
    
    def _fallback_analysis(self, provider: str, child_age: int, context: str, ml_results: Dict) -> LLMResponse:
        """Fallback analysis when provider fails"""
        fallback_response = f"""
PSYCHOLOGICAL ANALYSIS - {child_age}-year-old child, {context}

DEVELOPMENTAL ASSESSMENT:
For a {child_age}-year-old, this drawing shows typical developmental markers for this age group.

EMOTIONAL INDICATORS:
The drawing suggests healthy emotional expression appropriate for the child's developmental stage.

RECOMMENDATIONS:
- Continue encouraging creative expression and artistic exploration
- Provide supportive environment for continued artistic development
- Monitor emotional well-being through regular creative activities

Note: This is a basic assessment. For comprehensive analysis, please ensure API keys are properly configured.
"""
        
        return LLMResponse(
            provider=f"{provider}_fallback",
            response=fallback_response,
            confidence=0.50,
            analysis_type="fallback",
            metadata={"fallback_reason": "API unavailable"}
        )
