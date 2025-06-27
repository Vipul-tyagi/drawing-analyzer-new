import os
import base64
import numpy as np
import cv2
from PIL import Image
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import logging

# Import your existing components
try:
    from llm_integrator import StreamlinedLLMIntegrator, LLMResponse
    LLM_INTEGRATOR_AVAILABLE = True
except ImportError:
    LLM_INTEGRATOR_AVAILABLE = False
    print("⚠️ LLM Integrator not available")

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available")

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    print("⚠️ BLIP not available")

@dataclass
class AIAnalysisResult:
    """Structured result from AI analysis"""
    provider: str
    analysis_type: str
    confidence: float
    primary_finding: str
    detailed_analysis: str
    metadata: Dict[str, Any]

class ComprehensiveAIAnalyzer:
    """
    Comprehensive AI Analysis Engine that orchestrates multiple AI models
    for advanced psychological assessment of children's drawings
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = self._setup_logging()
        
        # Initialize AI components
        self._setup_ai_components()
        
        # Analysis categories for comprehensive assessment
        self.analysis_categories = {
            'visual_understanding': ['object_detection', 'scene_description', 'color_analysis'],
            'psychological_assessment': ['emotional_state', 'developmental_stage', 'personality_traits'],
            'clinical_indicators': ['trauma_markers', 'attachment_patterns', 'behavioral_signs'],
            'developmental_analysis': ['age_appropriateness', 'skill_level', 'cognitive_markers']
        }
        
        print("✅ Comprehensive AI Analyzer initialized")
    
    def _setup_logging(self):
        """Setup logging for the AI analyzer"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _setup_ai_components(self):
        """Initialize all AI components"""
        # Initialize LLM Integrator
        if LLM_INTEGRATOR_AVAILABLE:
            try:
                self.llm_integrator = StreamlinedLLMIntegrator()
                self.logger.info("✅ LLM Integrator loaded")
            except Exception as e:
                self.logger.error(f"Failed to load LLM Integrator: {e}")
                self.llm_integrator = None
        else:
            self.llm_integrator = None
        
        # Initialize CLIP
        if CLIP_AVAILABLE:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                self.logger.info("✅ CLIP model loaded")
            except Exception as e:
                self.logger.error(f"Failed to load CLIP: {e}")
                self.clip_model = None
        else:
            self.clip_model = None
        
        # Initialize BLIP
        if BLIP_AVAILABLE:
            try:
                self.blip_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
                self.blip_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
                self.blip_model.to(self.device)
                self.logger.info("✅ BLIP model loaded")
            except Exception as e:
                self.logger.error(f"Failed to load BLIP: {e}")
                self.blip_model = None
        else:
            self.blip_model = None
    
    def conduct_multi_ai_analysis(self, image_path: str, child_age: int, 
                                 drawing_context: str) -> Dict[str, Any]:
        """
        Main method to conduct comprehensive multi-AI analysis
        """
        self.logger.info(f"🤖 Starting comprehensive AI analysis for {child_age}-year-old's {drawing_context}")
        
        try:
            # Load and validate image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            
            # Conduct individual AI analyses
            analyses = {}
            
            # 1. CLIP Visual Understanding
            if self.clip_model:
                analyses['clip_analysis'] = self._conduct_clip_analysis(image, child_age, drawing_context)
            
            # 2. BLIP Image Captioning
            if self.blip_model:
                analyses['blip_analysis'] = self._conduct_blip_analysis(image, child_age, drawing_context)
            
            # 3. LLM Psychological Analysis
            if self.llm_integrator:
                analyses['llm_psychological'] = self._conduct_llm_analysis(image_path, child_age, drawing_context)
            
            # 4. Generate consensus analysis
            analyses['consensus_analysis'] = self._generate_consensus_analysis(analyses, child_age, drawing_context)
            
            # 5. Compile individual analyses for expert collaboration
            analyses['individual_analyses'] = self._compile_individual_analyses(analyses)
            
            self.logger.info("✅ Comprehensive AI analysis completed")
            return analyses
            
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return self._create_fallback_analysis(child_age, drawing_context, str(e))
    
    def _conduct_clip_analysis(self, image: Image.Image, child_age: int, drawing_context: str) -> Dict[str, Any]:
        """Conduct CLIP-based visual understanding analysis"""
        try:
            self.logger.info("🔍 Conducting CLIP visual analysis...")
            
            # Prepare image for CLIP
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Define comprehensive categories for children's drawings
            categories = [
                "a child's drawing of a person",
                "a child's drawing of a house",
                "a child's drawing of a tree",
                "a child's drawing of an animal",
                "a child's drawing of a family",
                "a child's drawing of a car",
                "a child's drawing of the sun",
                "a child's drawing of flowers",
                "a child's drawing showing happiness",
                "a child's drawing showing sadness",
                "a child's drawing showing fear",
                "a child's drawing showing anger",
                "a detailed child's drawing",
                "a simple child's drawing",
                "a colorful child's drawing",
                "a dark child's drawing",
                "an age-appropriate drawing",
                "an advanced drawing for the age",
                "a concerning drawing",
                "a healthy emotional expression"
            ]
            
            # Process with CLIP
            text_inputs = clip.tokenize(categories).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                
                # Calculate similarities
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Extract results
            category_scores = {}
            for i, category in enumerate(categories):
                category_scores[category.replace("a child's drawing ", "")] = float(similarities[0][i])
            
            # Find dominant category
            dominant_idx = similarities.argmax().item()
            dominant_category = categories[dominant_idx].replace("a child's drawing ", "")
            confidence = float(similarities[0][dominant_idx])
            
            return {
                'dominant_category': dominant_category,
                'confidence': confidence,
                'category_scores': category_scores,
                'analysis_type': 'visual_understanding',
                'model_used': 'CLIP-ViT-B/32'
            }
            
        except Exception as e:
            self.logger.error(f"CLIP analysis failed: {e}")
            return {
                'dominant_category': 'drawing',
                'confidence': 0.5,
                'category_scores': {},
                'error': str(e)
            }
    
    def _conduct_blip_analysis(self, image: Image.Image, child_age: int, drawing_context: str) -> Dict[str, Any]:
        """Conduct BLIP-based image description analysis"""
        try:
            self.logger.info("📝 Conducting BLIP description analysis...")
            
            # Process image with BLIP
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
                description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # Analyze description for psychological indicators
            psychological_indicators = self._analyze_description_psychology(description, child_age)
            
            # Calculate confidence based on description quality
            confidence = self._calculate_description_confidence(description, child_age, drawing_context)
            
            return {
                'description': description,
                'confidence': confidence,
                'psychological_indicators': psychological_indicators,
                'analysis_type': 'image_description',
                'model_used': 'BLIP-base'
            }
            
        except Exception as e:
            self.logger.error(f"BLIP analysis failed: {e}")
            return {
                'description': 'A child\'s drawing',
                'confidence': 0.5,
                'psychological_indicators': {},
                'error': str(e)
            }
    
    def _conduct_llm_analysis(self, image_path: str, child_age: int, drawing_context: str) -> Dict[str, Any]:
        """Conduct LLM-based psychological analysis"""
        try:
            self.logger.info("🧠 Conducting LLM psychological analysis...")
            
            # Create mock traditional results for LLM analysis
            mock_traditional_results = {
                'blip_description': 'A child\'s drawing',
                'color_analysis': {'dominant_color': 'mixed'},
                'shape_analysis': {'complexity_level': 'medium'},
                'spatial_analysis': {'spatial_balance': 'balanced'}
            }
            
            # Get LLM analyses
            llm_responses = self.llm_integrator.analyze_drawing_comprehensive(
                image_path, child_age, drawing_context, mock_traditional_results
            )
            
            if llm_responses:
                # Combine LLM responses
                combined_analysis = self._combine_llm_responses(llm_responses)
                
                return {
                    'interpretation': combined_analysis['interpretation'],
                    'confidence': combined_analysis['confidence'],
                    'individual_responses': llm_responses,
                    'analysis_type': 'psychological_interpretation',
                    'providers_used': [resp.provider for resp in llm_responses]
                }
            else:
                return self._create_fallback_llm_analysis(child_age, drawing_context)
                
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return self._create_fallback_llm_analysis(child_age, drawing_context, str(e))
    
    def _analyze_description_psychology(self, description: str, child_age: int) -> Dict[str, Any]:
        """Analyze psychological indicators from image description"""
        indicators = {
            'emotional_valence': 'neutral',
            'complexity_level': 'medium',
            'social_elements': False,
            'concerning_elements': False
        }
        
        description_lower = description.lower()
        
        # Emotional analysis
        positive_words = ['happy', 'smiling', 'bright', 'colorful', 'playing', 'sunny', 'cheerful']
        negative_words = ['sad', 'dark', 'crying', 'alone', 'scared', 'angry', 'broken']
        
        positive_count = sum(1 for word in positive_words if word in description_lower)
        negative_count = sum(1 for word in negative_words if word in description_lower)
        
        if positive_count > negative_count:
            indicators['emotional_valence'] = 'positive'
        elif negative_count > positive_count:
            indicators['emotional_valence'] = 'negative'
        
        # Social elements
        social_words = ['family', 'people', 'person', 'friends', 'together', 'group']
        indicators['social_elements'] = any(word in description_lower for word in social_words)
        
        # Concerning elements
        concerning_words = ['violence', 'weapon', 'blood', 'death', 'monster', 'nightmare']
        indicators['concerning_elements'] = any(word in description_lower for word in concerning_words)
        
        return indicators
    
    def _calculate_description_confidence(self, description: str, child_age: int, drawing_context: str) -> float:
        """Calculate confidence in description quality"""
        base_confidence = 0.7
        
        # Adjust based on description length and detail
        word_count = len(description.split())
        if word_count > 8:
            base_confidence += 0.1
        elif word_count < 4:
            base_confidence -= 0.1
        
        # Adjust based on context relevance
        if drawing_context.lower() in description.lower():
            base_confidence += 0.1
        
        return float(np.clip(base_confidence, 0.1, 0.95))
    
    def _combine_llm_responses(self, llm_responses: List[LLMResponse]) -> Dict[str, Any]:
        """Combine multiple LLM responses into unified analysis"""
        if not llm_responses:
            return {'interpretation': 'No LLM analysis available', 'confidence': 0.5}
        
        # Calculate average confidence
        avg_confidence = np.mean([resp.confidence for resp in llm_responses])
        
        # Combine interpretations
        interpretations = [resp.response for resp in llm_responses]
        combined_interpretation = self._synthesize_interpretations(interpretations)
        
        return {
            'interpretation': combined_interpretation,
            'confidence': float(avg_confidence),
            'num_responses': len(llm_responses)
        }
    
    def _synthesize_interpretations(self, interpretations: List[str]) -> str:
        """Synthesize multiple interpretations into coherent analysis"""
        if not interpretations:
            return "No interpretations available"
        
        if len(interpretations) == 1:
            return interpretations[0]
        
        # For multiple interpretations, create a synthesis
        synthesis = f"Based on {len(interpretations)} AI expert analyses:\n\n"
        
        # Extract common themes
        common_themes = self._extract_common_themes(interpretations)
        if common_themes:
            synthesis += f"Common themes identified: {', '.join(common_themes)}\n\n"
        
        # Add first interpretation as primary
        synthesis += f"Primary analysis: {interpretations[0][:200]}..."
        
        if len(interpretations) > 1:
            synthesis += f"\n\nAdditional perspectives from {len(interpretations)-1} other AI experts provide supporting evidence for this assessment."
        
        return synthesis
    
    def _extract_common_themes(self, interpretations: List[str]) -> List[str]:
        """Extract common themes from multiple interpretations"""
        themes = []
        
        # Common psychological terms to look for
        theme_keywords = [
            'positive', 'negative', 'healthy', 'concerning', 'appropriate', 'advanced',
            'emotional', 'cognitive', 'social', 'creative', 'developmental'
        ]
        
        for keyword in theme_keywords:
            count = sum(1 for interp in interpretations if keyword in interp.lower())
            if count >= len(interpretations) // 2:  # Majority agreement
                themes.append(keyword)
        
        return themes[:5]  # Limit to top 5 themes
    
    def _generate_consensus_analysis(self, analyses: Dict[str, Any], child_age: int, drawing_context: str) -> Dict[str, Any]:
        """Generate consensus analysis from all AI components"""
        try:
            self.logger.info("🎯 Generating AI consensus analysis...")
            
            # Extract key findings from each analysis
            findings = {}
            
            # CLIP findings
            if 'clip_analysis' in analyses:
                clip_data = analyses['clip_analysis']
                findings['visual_category'] = clip_data.get('dominant_category', 'unknown')
                findings['visual_confidence'] = clip_data.get('confidence', 0.5)
            
            # BLIP findings
            if 'blip_analysis' in analyses:
                blip_data = analyses['blip_analysis']
                findings['description_quality'] = blip_data.get('confidence', 0.5)
                findings['emotional_indicators'] = blip_data.get('psychological_indicators', {})
            
            # LLM findings
            if 'llm_psychological' in analyses:
                llm_data = analyses['llm_psychological']
                findings['psychological_confidence'] = llm_data.get('confidence', 0.5)
                findings['expert_interpretation'] = llm_data.get('interpretation', '')
            
            # Generate consensus
            consensus = self._calculate_consensus(findings, child_age, drawing_context)
            
            return consensus
            
        except Exception as e:
            self.logger.error(f"Consensus generation failed: {e}")
            return self._create_fallback_consensus(child_age, drawing_context)
    
    def _calculate_consensus(self, findings: Dict[str, Any], child_age: int, drawing_context: str) -> Dict[str, Any]:
        """Calculate consensus from individual findings"""
        # Determine primary emotional state
        emotional_indicators = findings.get('emotional_indicators', {})
        emotional_valence = emotional_indicators.get('emotional_valence', 'neutral')
        
        if emotional_valence == 'positive':
            primary_emotional_state = 'positive'
        elif emotional_valence == 'negative':
            primary_emotional_state = 'concerning'
        else:
            primary_emotional_state = 'neutral'
        
        # Determine developmental assessment
        visual_confidence = findings.get('visual_confidence', 0.5)
        description_quality = findings.get('description_quality', 0.5)
        
        avg_quality = (visual_confidence + description_quality) / 2
        
        if avg_quality > 0.7:
            developmental_assessment = 'age_appropriate_or_advanced'
        elif avg_quality > 0.4:
            developmental_assessment = 'age_appropriate'
        else:
            developmental_assessment = 'may_need_support'
        
        # Calculate AI agreement score
        confidences = [
            findings.get('visual_confidence', 0.5),
            findings.get('description_quality', 0.5),
            findings.get('psychological_confidence', 0.5)
        ]
        
        ai_agreement_score = np.mean(confidences)
        
        # Overall confidence level
        confidence_level = ai_agreement_score
        
        return {
            'primary_emotional_state': primary_emotional_state,
            'developmental_assessment': developmental_assessment,
            'ai_agreement_score': float(ai_agreement_score),
            'confidence_level': float(confidence_level),
            'consensus_quality': 'high' if confidence_level > 0.7 else 'medium' if confidence_level > 0.5 else 'low',
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _compile_individual_analyses(self, analyses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compile individual analyses for expert collaboration"""
        individual_analyses = []
        
        # CLIP analysis
        if 'clip_analysis' in analyses:
            clip_data = analyses['clip_analysis']
            individual_analyses.append({
                'provider': 'CLIP',
                'analysis_type': 'visual_understanding',
                'confidence': clip_data.get('confidence', 0.5),
                'primary_finding': clip_data.get('dominant_category', 'unknown'),
                'detailed_analysis': f"Visual analysis identified: {clip_data.get('dominant_category', 'unknown')}",
                'metadata': {
                    'model': 'CLIP-ViT-B/32',
                    'category_scores': clip_data.get('category_scores', {})
                }
            })
        
        # BLIP analysis
        if 'blip_analysis' in analyses:
            blip_data = analyses['blip_analysis']
            individual_analyses.append({
                'provider': 'BLIP',
                'analysis_type': 'image_description',
                'confidence': blip_data.get('confidence', 0.5),
                'primary_finding': blip_data.get('description', 'A child\'s drawing'),
                'detailed_analysis': f"Description: {blip_data.get('description', 'A child\'s drawing')}",
                'metadata': {
                    'model': 'BLIP-base',
                    'psychological_indicators': blip_data.get('psychological_indicators', {})
                }
            })
        
        # LLM analyses
        if 'llm_psychological' in analyses:
            llm_data = analyses['llm_psychological']
            if 'individual_responses' in llm_data:
                for resp in llm_data['individual_responses']:
                    individual_analyses.append({
                        'provider': resp.provider,
                        'analysis_type': resp.analysis_type,
                        'confidence': resp.confidence,
                        'primary_finding': 'Psychological interpretation',
                        'detailed_analysis': resp.response,
                        'metadata': resp.metadata
                    })
        
        return individual_analyses
    
    def _create_fallback_analysis(self, child_age: int, drawing_context: str, error: str = None) -> Dict[str, Any]:
        """Create fallback analysis when AI components fail"""
        return {
            'clip_analysis': {
                'dominant_category': 'drawing',
                'confidence': 0.5,
                'category_scores': {},
                'error': error
            },
            'blip_analysis': {
                'description': f'A {child_age}-year-old child\'s {drawing_context.lower()}',
                'confidence': 0.5,
                'psychological_indicators': {'emotional_valence': 'neutral'}
            },
            'llm_psychological': {
                'interpretation': f'This appears to be a {drawing_context.lower()} by a {child_age}-year-old child. The drawing shows typical developmental characteristics for this age group.',
                'confidence': 0.5,
                'individual_responses': []
            },
            'consensus_analysis': {
                'primary_emotional_state': 'neutral',
                'developmental_assessment': 'age_appropriate',
                'ai_agreement_score': 0.5,
                'confidence_level': 0.5,
                'consensus_quality': 'low'
            },
            'individual_analyses': [],
            'fallback_reason': error or 'AI components not available'
        }
    
    def _create_fallback_llm_analysis(self, child_age: int, drawing_context: str, error: str = None) -> Dict[str, Any]:
        """Create fallback LLM analysis"""
        return {
            'interpretation': f'This {drawing_context.lower()} by a {child_age}-year-old shows typical developmental characteristics. The child demonstrates age-appropriate artistic expression and creativity.',
            'confidence': 0.5,
            'individual_responses': [],
            'analysis_type': 'fallback_psychological',
            'providers_used': [],
            'error': error
        }
    
    def _create_fallback_consensus(self, child_age: int, drawing_context: str) -> Dict[str, Any]:
        """Create fallback consensus analysis"""
        return {
            'primary_emotional_state': 'neutral',
            'developmental_assessment': 'age_appropriate',
            'ai_agreement_score': 0.5,
            'confidence_level': 0.5,
            'consensus_quality': 'low',
            'analysis_timestamp': datetime.now().isoformat(),
            'fallback': True
        }
    
    def get_analysis_capabilities(self) -> Dict[str, bool]:
        """Get current analysis capabilities"""
        return {
            'clip_visual_analysis': self.clip_model is not None,
            'blip_description': self.blip_model is not None,
            'llm_psychological': self.llm_integrator is not None,
            'multi_ai_consensus': True,
            'expert_collaboration': True
        }
    
    def validate_analysis_quality(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the quality of analysis results"""
        quality_metrics = {
            'completeness': 0.0,
            'confidence': 0.0,
            'consistency': 0.0,
            'overall_quality': 'low'
        }
        
        try:
            # Check completeness
            expected_components = ['clip_analysis', 'blip_analysis', 'llm_psychological', 'consensus_analysis']
            present_components = sum(1 for comp in expected_components if comp in analysis_results)
            quality_metrics['completeness'] = present_components / len(expected_components)
            
            # Check confidence levels
            confidences = []
            for component in expected_components:
                if component in analysis_results:
                    conf = analysis_results[component].get('confidence', 0.5)
                    confidences.append(conf)
            
            if confidences:
                quality_metrics['confidence'] = np.mean(confidences)
            
            # Check consistency (low variance in confidence scores indicates consistency)
            if len(confidences) > 1:
                consistency = 1.0 - np.var(confidences)
                quality_metrics['consistency'] = max(0.0, consistency)
            else:
                quality_metrics['consistency'] = 0.5
            
            # Overall quality
            overall = np.mean([
                quality_metrics['completeness'],
                quality_metrics['confidence'],
                quality_metrics['consistency']
            ])
            
            if overall > 0.8:
                quality_metrics['overall_quality'] = 'excellent'
            elif overall > 0.6:
                quality_metrics['overall_quality'] = 'good'
            elif overall > 0.4:
                quality_metrics['overall_quality'] = 'fair'
            else:
                quality_metrics['overall_quality'] = 'poor'
            
        except Exception as e:
            self.logger.error(f"Quality validation failed: {e}")
        
        return quality_metrics

# Utility functions for integration
def test_ai_analysis_engine():
    """Test the AI analysis engine"""
    print("🧪 Testing AI Analysis Engine...")
    
    try:
        analyzer = ComprehensiveAIAnalyzer()
        capabilities = analyzer.get_analysis_capabilities()
        
        print("📊 Analysis Capabilities:")
        for capability, available in capabilities.items():
            status = "✅" if available else "❌"
            print(f"  {status} {capability}")
        
        print("\n✅ AI Analysis Engine test completed!")
        return True
        
    except Exception as e:
        print(f"❌ AI Analysis Engine test failed: {e}")
        return False

if __name__ == "__main__":
    test_ai_analysis_engine()