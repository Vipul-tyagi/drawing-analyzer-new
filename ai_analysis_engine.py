import numpy as np
import cv2
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, pipeline, BlipProcessor, BlipForConditionalGeneration
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import logging
from scipy.stats import entropy
from llm_integrator import StreamlinedLLMIntegrator
import warnings
warnings.filterwarnings("ignore")

try:
    from llm_integrator import StreamlinedLLMIntegrator
    LLM_INTEGRATOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LLM Integrator not available: {e}")
    LLM_INTEGRATOR_AVAILABLE = False
    
    # Create a dummy class to prevent crashes
    class StreamlinedLLMIntegrator:
        def __init__(self):
            self.available_providers = []
        
        def analyze_drawing_comprehensive(self, *args, **kwargs):
            return []

class ComprehensiveAIAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = self._setup_logging()
        self.setup_ai_models()
        
        # ADD: Initialize LLM integrator
        self.llm_integrator = StreamlinedLLMIntegrator()
        
        self.psychological_frameworks = {
            'beck_depression': self._load_beck_framework(),
            'attachment_theory': self._load_attachment_framework(),
            'developmental_stages': self._load_developmental_framework(),
            'trauma_indicators': self._load_trauma_framework()
            }
    
    def llm_psychological_analysis(self, image_description: Dict, child_age: int, context: str, image_path: str = None) -> Dict:
        """Enhanced LLM analysis using multiple providers"""
        
        # Try advanced LLM analysis first
        if self.llm_integrator.available_providers and image_path:
            return self.advanced_llm_analysis(image_path, child_age, context, image_description)
        
        # Fallback to local LLM or rule-based
        elif self.llm_pipeline:
            return self.local_llm_analysis(image_description, child_age, context)
        else:
            return self.rule_based_psychological_analysis(image_description, child_age, context)
        
    def local_llm_analysis(self, ml_results: Dict, child_age: int, context: str) -> Dict:
        """Fallback local analysis when external LLMs are unavailable"""
    
        # Extract key information from ML results
        description = ml_results.get('primary_description', 'A child\'s drawing')
    
        # Generate analysis based on available data
        analysis = f"""
        PSYCHOLOGICAL ANALYSIS - {child_age}-year-old child, {context}

        VISUAL DESCRIPTION:
        {description}
    
        DEVELOPMENTAL ASSESSMENT:
        For a {child_age}-year-old, this drawing shows {'age-appropriate' if child_age >= 4 else 'early'} developmental markers.
    
        EMOTIONAL INDICATORS:
        The drawing suggests typical emotional expression for this age group.
    
        RECOMMENDATIONS:
        - Continue encouraging creative expression
        - Provide supportive environment for artistic development
        - Monitor emotional well-being through regular check-ins
        """

        return {
            'interpretation': analysis,
            'confidence': 0.6,
            'model': 'Local Rule-Based Analysis',
            'structured_analysis': {
                'visual_description': description,
                'developmental_assessment': f'Age-appropriate for {child_age} years',
                'emotional_indicators': 'Typical emotional expression',
                'recommendations': 'Continue supportive environment'
            },
            'clinical_indicators': {'depression_risk': 'low', 'anxiety_risk': 'low'},
            'risk_assessment': {'level': 'low', 'confidence': 0.6},
            'recommendations': [
            'Continue encouraging creative expression',
            'Provide supportive environment',
            'Monitor emotional well-being'
            ],
            'developmental_insights': {
                'developmental_level': 'age_appropriate',
                'cognitive_indicators': ['typical_development'],
                'motor_skills': 'typical'
            }
        }
    
    def advanced_llm_analysis(self, image_path: str, child_age: int, context: str, ml_results: Dict) -> Dict:
        """Use multiple LLM providers for comprehensive analysis"""
        
        try:
            # Check if we have any working providers
            if not hasattr(self.llm_integrator, 'available_providers') or not self.llm_integrator.available_providers:
                print("No LLM providers available, using local analysis")
                return self.local_llm_analysis(ml_results, child_age, context)
            
            # Load image for LLM analysis
            from PIL import Image
            image = Image.open(image_path)
            
            # Get analyses from all available LLM providers
            llm_responses = self.llm_integrator.analyze_drawing_comprehensive(
                image, child_age, context, ml_results
            )
            
            if not llm_responses:
                return self.rule_based_psychological_analysis(ml_results, child_age, context)
            
            # Combine all LLM responses
            combined_analysis = self._combine_llm_responses(llm_responses, child_age)
            
            # Process responses...
            combined_analysis = self._combine_llm_responses(llm_responses, child_age)
            
            return {
                'interpretation': combined_analysis['combined_interpretation'],
                'confidence': combined_analysis['average_confidence'],
                'model': 'Multi-LLM (OpenAI + DeepSeek + Perplexity)',
                'structured_analysis': combined_analysis['structured_analysis'],
                'clinical_indicators': self.extract_clinical_indicators_from_llm(combined_analysis['combined_interpretation']),
                'risk_assessment': self.assess_risk_from_llm(combined_analysis['combined_interpretation'], child_age),
                'recommendations': self.extract_recommendations_from_llm(combined_analysis['combined_interpretation']),
                'developmental_insights': self.extract_developmental_insights(combined_analysis['combined_interpretation'], child_age),
                'individual_llm_responses': llm_responses
            }
            
        except Exception as e:
            self.logger.error(f"Advanced LLM analysis failed: {e}")
            return self.rule_based_psychological_analysis(ml_results, child_age, context)
    
    def _combine_llm_responses(self, llm_responses: List, child_age: int) -> Dict:
        """Combine multiple LLM responses into a comprehensive analysis"""
        
        combined_text = ""
        confidences = []
        
        for response in llm_responses:
            combined_text += f"\n\n## Analysis from {response.provider}:\n"
            combined_text += response.response
            confidences.append(response.confidence)
        
        # Create structured analysis
        structured_analysis = {
            'visual_description': self._extract_section(combined_text, 'VISUAL DESCRIPTION'),
            'developmental_assessment': self._extract_section(combined_text, 'DEVELOPMENTAL ASSESSMENT'),
            'psychological_indicators': self._extract_section(combined_text, 'PSYCHOLOGICAL INDICATORS'),
            'color_psychology': self._extract_section(combined_text, 'COLOR PSYCHOLOGY'),
            'spatial_organization': self._extract_section(combined_text, 'SPATIAL ORGANIZATION'),
            'recommendations': self._extract_section(combined_text, 'RECOMMENDATIONS'),
            'overall_assessment': self._extract_section(combined_text, 'OVERALL ASSESSMENT')
        }
        
        return {
            'combined_interpretation': combined_text,
            'average_confidence': np.mean(confidences) if confidences else 0.7,
            'structured_analysis': structured_analysis,
            'provider_count': len(llm_responses)
        }
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract specific sections from LLM responses"""
        lines = text.split('\n')
        section_content = []
        in_section = False
        
        for line in lines:
            if section_name.upper() in line.upper():
                in_section = True
                continue
            elif line.startswith('##') and in_section:
                break
            elif in_section:
                section_content.append(line)
        
        return '\n'.join(section_content).strip()
    
    def _setup_logging(self):
        """Setup logging for AI analysis"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def setup_ai_models(self):
        """Setup all AI models for comprehensive analysis"""
        
        # CLIP for visual understanding
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            self.logger.info("✅ CLIP model loaded successfully")
        except Exception as e:
            self.logger.error(f"⚠️ CLIP model failed to load: {e}")
            self.clip_model = None
        
        # BLIP for image captioning
        try:
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model.to(self.device)
            self.logger.info("✅ BLIP model loaded successfully")
        except Exception as e:
            self.logger.error(f"⚠️ BLIP model failed to load: {e}")
            self.blip_model = None
        
        # LLM for psychological interpretation
        try:
            self.llm_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1,
                max_length=512
            )
            self.logger.info("✅ LLM model loaded successfully")
        except Exception as e:
            self.logger.error(f"⚠️ LLM model failed to load: {e}")
            self.llm_pipeline = None
        
        # Sentiment analysis for emotional assessment
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info("✅ Sentiment analysis model loaded successfully")
        except Exception as e:
            self.logger.error(f"⚠️ Sentiment analysis model failed to load: {e}")
            self.sentiment_pipeline = None
    
    def conduct_multi_ai_analysis(self, image_path: str, child_age: int, context: str) -> Dict:
        """Conduct comprehensive multi-AI analysis"""
        
        self.logger.info(f"🤖 Starting comprehensive AI analysis for age {child_age}, context: {context}")
        
        analyses = {}
        
        # 1. CLIP-based visual analysis
        self.logger.info("🔍 Conducting CLIP visual analysis...")
        analyses['clip_analysis'] = self.clip_visual_analysis(image_path, child_age, context)
        
        # 2. BLIP image captioning and description
        self.logger.info("📝 Generating BLIP image description...")
        analyses['blip_analysis'] = self.blip_image_description(image_path)
        
        # 3. LLM psychological interpretation
        self.logger.info("🧠 Conducting LLM psychological analysis...")
        analyses['llm_psychological'] = self.llm_psychological_analysis(
            analyses['blip_analysis'], child_age, context
        )
        
        # 4. Sentiment analysis of drawing content
        self.logger.info("😊 Analyzing emotional sentiment...")
        analyses['sentiment_analysis'] = self.sentiment_analysis(analyses['blip_analysis'])
        
        # 5. Multi-model consensus
        self.logger.info("🎯 Generating AI consensus...")
        analyses['consensus_analysis'] = self.generate_consensus(analyses, child_age, context)
        
        # 6. Individual analyses for expert collaboration
        analyses['individual_analyses'] = self.create_individual_analyses(analyses)
        
        # 7. Confidence and reliability metrics
        analyses['ai_confidence_metrics'] = self.calculate_ai_confidence_metrics(analyses)
        
        self.logger.info("✅ Comprehensive AI analysis completed")
        return analyses
    
    def clip_visual_analysis(self, image_path: str, child_age: int, context: str) -> Dict:
        """CLIP-based comprehensive visual analysis"""
        if not self.clip_model:
            return {'error': 'CLIP model not available'}
        
        try:
            image = Image.open(image_path)
            
            # Comprehensive psychological categories for CLIP analysis
            base_categories = [
                "a happy child's drawing",
                "a sad child's drawing", 
                "a detailed child's drawing",
                "a simple child's drawing",
                "a drawing showing family",
                "a drawing showing isolation",
                "a creative artistic drawing",
                "a drawing with bright colors",
                "a drawing with dark colors",
                "a drawing showing confidence",
                "a drawing showing anxiety",
                "a drawing of a person",
                "a drawing of a house",
                "a drawing of a tree",
                "a drawing of animals",
                "a drawing showing emotions",
                "a drawing with good proportions",
                "a drawing with poor proportions",
                "a drawing showing developmental maturity",
                "a drawing showing developmental delay"
            ]
            
            # Age-specific categories
            age_specific_categories = self._get_age_specific_categories(child_age)
            
            # Context-specific categories
            context_specific_categories = self._get_context_specific_categories(context)
            
            # Combine all categories
            all_categories = base_categories + age_specific_categories + context_specific_categories
            
            # Process with CLIP
            inputs = self.clip_processor(
                text=all_categories,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
            
            # Create analysis results
            clip_results = {}
            for i, category in enumerate(all_categories):
                clean_category = category.replace("a ", "").replace("child's ", "")
                clip_results[clean_category] = float(probs[0][i])
            
            # Extract psychological indicators
            psychological_indicators = self.extract_psychological_indicators_from_clip(clip_results)
            
            # Developmental assessment
            developmental_assessment = self.assess_development_from_clip(clip_results, child_age)
            
            # Emotional assessment
            emotional_assessment = self.assess_emotions_from_clip(clip_results)
            
            return {
                'category_scores': clip_results,
                'dominant_category': max(clip_results, key=clip_results.get),
                'confidence': float(max(probs[0])),
                'psychological_indicators': psychological_indicators,
                'developmental_assessment': developmental_assessment,
                'emotional_assessment': emotional_assessment,
                'age_appropriateness': self._assess_age_appropriateness_clip(clip_results, child_age),
                'clinical_flags': self._detect_clinical_flags_clip(clip_results)
            }
            
        except Exception as e:
            self.logger.error(f"CLIP analysis failed: {str(e)}")
            return {'error': f'CLIP analysis failed: {str(e)}'}
    
    def blip_image_description(self, image_path: str) -> Dict:
        """BLIP-based comprehensive image description"""
        if not self.blip_model:
            return {'description': 'Image captioning not available', 'confidence': 0.5}
        
        try:
            image = Image.open(image_path)
            
            # Generate multiple descriptions with different prompts
            descriptions = {}
            
            # Basic description
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
            basic_description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            descriptions['basic'] = basic_description
            
            # Emotional description
            emotional_prompt = "This drawing shows emotions of"
            inputs = self.blip_processor(image, emotional_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
            emotional_description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            descriptions['emotional'] = emotional_description
            
            # Detailed description
            detail_prompt = "This detailed drawing contains"
            inputs = self.blip_processor(image, detail_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
            detailed_description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            descriptions['detailed'] = detailed_description
            
            # Extract psychological elements
            psychological_elements = self.extract_psychological_elements_from_blip(descriptions)
            
            # Analyze narrative structure
            narrative_analysis = self.analyze_narrative_structure(descriptions)
            
            return {
                'descriptions': descriptions,
                'primary_description': basic_description,
                'confidence': 0.8,  # Default confidence for BLIP
                'model': 'BLIP',
                'psychological_elements': psychological_elements,
                'narrative_analysis': narrative_analysis,
                'complexity_assessment': self._assess_description_complexity(descriptions),
                'emotional_content': self._extract_emotional_content_blip(descriptions)
            }
            
        except Exception as e:
            self.logger.error(f"BLIP analysis failed: {str(e)}")
            return {'description': f'Error: {str(e)}', 'confidence': 0.0}
    
    def llm_psychological_analysis(self, image_description: Dict, child_age: int, context: str) -> Dict:
        """LLM-based comprehensive psychological analysis"""
        if not self.llm_pipeline:
            return self.rule_based_psychological_analysis(image_description, child_age, context)
        
        # Create comprehensive prompt for LLM
        primary_description = image_description.get('primary_description', 'A child\'s drawing')
        emotional_desc = image_description.get('descriptions', {}).get('emotional', '')
        detailed_desc = image_description.get('descriptions', {}).get('detailed', '')
        
        prompt = f"""
        As a licensed child psychologist with expertise in art therapy and developmental psychology, analyze this child's drawing:
        
        Basic Description: {primary_description}
        Emotional Content: {emotional_desc}
        Detailed Elements: {detailed_desc}
        
        Child Age: {child_age} years
        Drawing Context: {context}
        
        Provide a comprehensive psychological analysis addressing:
        
        1. EMOTIONAL WELLBEING:
        - Current emotional state indicators
        - Mood and affect assessment
        - Emotional regulation capabilities
        - Signs of emotional distress or resilience
        
        2. DEVELOPMENTAL ASSESSMENT:
        - Age-appropriateness of drawing skills
        - Cognitive development indicators
        - Fine motor skill development
        - Symbolic thinking progression
        
        3. SOCIAL AND FAMILY DYNAMICS:
        - Social connection indicators
        - Family relationship patterns
        - Attachment style indicators
        - Peer interaction capabilities
        
        4. POTENTIAL CONCERNS:
        - Any red flags requiring attention
        - Trauma indicators (if present)
        - Developmental delays (if evident)
        - Emotional support needs
        
        5. STRENGTHS AND RESOURCES:
        - Creative abilities and talents
        - Coping mechanisms
        - Resilience factors
        - Areas of healthy development
        
        6. RECOMMENDATIONS:
        - Immediate support strategies
        - Long-term developmental goals
        - Environmental modifications
        - Professional referrals (if needed)
        
        Analysis:
        """
        
        try:
            response = self.llm_pipeline(
                prompt,
                max_length=800,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id
            )
            
            interpretation = response[0]['generated_text'].replace(prompt, '').strip()
            
            # Structure the analysis
            structured_analysis = self.structure_llm_analysis(interpretation)
            
            # Extract specific indicators
            clinical_indicators = self.extract_clinical_indicators_from_llm(interpretation)
            
            # Risk assessment
            risk_assessment = self.assess_risk_from_llm(interpretation, child_age)
            
            return {
                'interpretation': interpretation,
                'confidence': 0.75,
                'model': 'LLM',
                'structured_analysis': structured_analysis,
                'clinical_indicators': clinical_indicators,
                'risk_assessment': risk_assessment,
                'recommendations': self.extract_recommendations_from_llm(interpretation),
                'developmental_insights': self.extract_developmental_insights(interpretation, child_age)
            }
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {str(e)}")
            return self.rule_based_psychological_analysis(image_description, child_age, context)
    
    def sentiment_analysis(self, image_description: Dict) -> Dict:
        """Analyze emotional sentiment from image descriptions"""
        if not self.sentiment_pipeline:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        try:
            descriptions = image_description.get('descriptions', {})
            sentiment_results = {}
            
            for desc_type, description in descriptions.items():
                if description and len(description.strip()) > 0:
                    result = self.sentiment_pipeline(description)
                    sentiment_results[desc_type] = {
                        'label': result[0]['label'],
                        'score': result[0]['score']
                    }
            
            # Calculate overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(sentiment_results)
            
            # Map to psychological constructs
            psychological_mapping = self._map_sentiment_to_psychology(overall_sentiment)
            
            return {
                'individual_sentiments': sentiment_results,
                'overall_sentiment': overall_sentiment,
                'psychological_mapping': psychological_mapping,
                'emotional_valence': self._calculate_emotional_valence(sentiment_results),
                'emotional_intensity': self._calculate_emotional_intensity(sentiment_results)
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {str(e)}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'error': str(e)}
    
    def generate_consensus(self, analyses: Dict, child_age: int, context: str) -> Dict:
        """Generate consensus from multiple AI analyses"""
        
        # Extract key insights from each model
        clip_analysis = analyses.get('clip_analysis', {})
        blip_analysis = analyses.get('blip_analysis', {})
        llm_analysis = analyses.get('llm_psychological', {})
        sentiment_analysis = analyses.get('sentiment_analysis', {})
        
        # Determine primary emotional state
        primary_emotional_state = self._determine_consensus_emotional_state(
            clip_analysis, blip_analysis, llm_analysis, sentiment_analysis
        )
        
        # Developmental assessment consensus
        developmental_assessment = self._determine_consensus_developmental_level(
            clip_analysis, llm_analysis, child_age
        )
        
        # Risk assessment consensus
        risk_assessment = self._determine_consensus_risk_level(
            clip_analysis, llm_analysis, sentiment_analysis
        )
        
        # Strengths identification
        strengths_assessment = self._identify_consensus_strengths(
            clip_analysis, llm_analysis
        )
        
        # Calculate confidence scores
        confidence_scores = self._calculate_consensus_confidence(analyses)
        
        # AI agreement score
        ai_agreement_score = self.calculate_ai_agreement(analyses)
        
        # Generate recommendations
        consensus_recommendations = self._generate_consensus_recommendations(
            primary_emotional_state, developmental_assessment, risk_assessment, child_age
        )
        
        return {
            'primary_emotional_state': primary_emotional_state,
            'developmental_assessment': developmental_assessment,
            'risk_assessment': risk_assessment,
            'strengths_assessment': strengths_assessment,
            'confidence_level': float(np.mean(list(confidence_scores.values()))),
            'confidence_breakdown': confidence_scores,
            'ai_agreement_score': float(ai_agreement_score),
            'consensus_timestamp': datetime.now().isoformat(),
            'models_used': self._get_models_used(analyses),
            'consensus_strength': self._categorize_consensus_strength(ai_agreement_score),
            'recommendations': consensus_recommendations,
            'clinical_significance': self._assess_clinical_significance(risk_assessment),
            'follow_up_needed': self._determine_follow_up_needs(risk_assessment, developmental_assessment)
        }
    
    def create_individual_analyses(self, analyses: Dict) -> List[Dict]:
        """Create individual analyses for expert collaboration"""
        individual_analyses = []
        
        # CLIP analysis
        if 'clip_analysis' in analyses and analyses['clip_analysis']:
            clip = analyses['clip_analysis']
            individual_analyses.append({
                'model': 'CLIP',
                'model_type': 'vision_language',
                'confidence': clip.get('confidence', 0.5),
                'primary_finding': clip.get('dominant_category', 'Unknown'),
                'detailed_scores': clip.get('category_scores', {}),
                'psychological_indicators': clip.get('psychological_indicators', {}),
                'clinical_flags': clip.get('clinical_flags', []),
                'timestamp': datetime.now().isoformat(),
                'strengths': ['Visual pattern recognition', 'Semantic understanding'],
                'limitations': ['Limited contextual understanding', 'Potential cultural bias']
            })
        
        # BLIP analysis
        if 'blip_analysis' in analyses and analyses['blip_analysis']:
            blip = analyses['blip_analysis']
            individual_analyses.append({
                'model': 'BLIP',
                'model_type': 'image_captioning',
                'confidence': blip.get('confidence', 0.5),
                'primary_finding': blip.get('primary_description', 'No description'),
                'descriptions': blip.get('descriptions', {}),
                'psychological_elements': blip.get('psychological_elements', []),
                'narrative_analysis': blip.get('narrative_analysis', {}),
                'timestamp': datetime.now().isoformat(),
                'strengths': ['Detailed description', 'Multiple perspectives'],
                'limitations': ['May miss subtle details', 'Limited psychological training']
            })
        
        # LLM analysis
        if 'llm_psychological' in analyses and analyses['llm_psychological']:
            llm = analyses['llm_psychological']
            individual_analyses.append({
                'model': 'LLM',
                'model_type': 'language_model',
                'confidence': llm.get('confidence', 0.5),
                'primary_finding': llm.get('interpretation', 'No interpretation')[:200] + '...',
                'full_interpretation': llm.get('interpretation', ''),
                'structured_analysis': llm.get('structured_analysis', {}),
                'clinical_indicators': llm.get('clinical_indicators', {}),
                'risk_assessment': llm.get('risk_assessment', {}),
                'timestamp': datetime.now().isoformat(),
                'strengths': ['Comprehensive analysis', 'Clinical reasoning'],
                'limitations': ['May hallucinate', 'Training data limitations']
            })
        
        # Sentiment analysis
        if 'sentiment_analysis' in analyses and analyses['sentiment_analysis']:
            sentiment = analyses['sentiment_analysis']
            individual_analyses.append({
                'model': 'Sentiment_Analyzer',
                'model_type': 'sentiment_analysis',
                'confidence': 0.7,  # Default for sentiment analysis
                'primary_finding': sentiment.get('overall_sentiment', {}).get('label', 'neutral'),
                'sentiment_breakdown': sentiment.get('individual_sentiments', {}),
                'emotional_valence': sentiment.get('emotional_valence', 0.5),
                'psychological_mapping': sentiment.get('psychological_mapping', {}),
                'timestamp': datetime.now().isoformat(),
                'strengths': ['Emotional detection', 'Quantified sentiment'],
                'limitations': ['Limited to text analysis', 'May miss visual emotions']
            })
        
        return individual_analyses
    
    def calculate_ai_confidence_metrics(self, analyses: Dict) -> Dict:
        """Calculate comprehensive AI confidence metrics"""
        
        confidence_metrics = {}
        
        # Individual model confidences
        clip_confidence = analyses.get('clip_analysis', {}).get('confidence', 0.5)
        blip_confidence = analyses.get('blip_analysis', {}).get('confidence', 0.5)
        llm_confidence = analyses.get('llm_psychological', {}).get('confidence', 0.5)
        
        confidence_metrics['individual_confidences'] = {
            'clip': float(clip_confidence),
            'blip': float(blip_confidence),
            'llm': float(llm_confidence)
        }
        
        # Overall confidence
        confidence_metrics['overall_confidence'] = float(np.mean([
            clip_confidence, blip_confidence, llm_confidence
        ]))
        
        # Confidence variance (lower = more agreement)
        confidence_variance = np.var([clip_confidence, blip_confidence, llm_confidence])
        confidence_metrics['confidence_variance'] = float(confidence_variance)
        
        # Agreement-based confidence
        agreement_score = self.calculate_ai_agreement(analyses)
        confidence_metrics['agreement_based_confidence'] = float(agreement_score)
        
        # Reliability estimate
        reliability_estimate = self._estimate_reliability(analyses)
        confidence_metrics['reliability_estimate'] = float(reliability_estimate)
        
        # Uncertainty quantification
        uncertainty_metrics = self._quantify_uncertainty(analyses)
        confidence_metrics['uncertainty_metrics'] = uncertainty_metrics
        
        return confidence_metrics
    
    # Helper methods for psychological frameworks
    def _load_beck_framework(self) -> Dict:
        """Load Beck's depression framework for analysis"""
        return {
            'cognitive_triad': ['negative_self', 'negative_world', 'negative_future'],
            'indicators': ['hopelessness', 'worthlessness', 'helplessness'],
            'visual_markers': ['dark_colors', 'small_size', 'isolation', 'incomplete_figures']
        }
    
    def _load_attachment_framework(self) -> Dict:
        """Load attachment theory framework"""
        return {
            'secure': ['proximity_seeking', 'safe_haven', 'secure_base'],
            'anxious': ['hyperactivation', 'proximity_seeking', 'fear_abandonment'],
            'avoidant': ['deactivation', 'self_reliance', 'emotional_distance'],
            'disorganized': ['approach_avoidance', 'fear_caregiver', 'dissociation']
        }
    
    def _load_developmental_framework(self) -> Dict:
        """Load developmental stages framework"""
        return {
            'scribbling': (1, 3),
            'pre_schematic': (4, 6),
            'schematic': (7, 9),
            'dawning_realism': (10, 12),
            'pseudo_naturalistic': (13, 17)
        }
    
    def _load_trauma_framework(self) -> Dict:
        """Load trauma indicators framework"""
        return {
            'fragmentation': ['broken_lines', 'disconnected_parts', 'incomplete_figures'],
            'dissociation': ['floating_elements', 'surreal_combinations', 'lack_coherence'],
            'hypervigilance': ['excessive_detail', 'multiple_eyes', 'defensive_postures'],
            'regression': ['age_inappropriate_simplicity', 'primitive_representations']
        }
    
    def _get_age_specific_categories(self, child_age: int) -> List[str]:
        """Get age-specific CLIP categories"""
        if child_age < 4:
            return [
                "a toddler's scribbling",
                "early mark-making",
                "basic circular shapes"
            ]
        elif child_age < 7:
            return [
                "a preschooler's drawing",
                "basic human figures",
                "simple house drawings"
            ]
        elif child_age < 10:
            return [
                "a school-age child's drawing",
                "detailed human figures",
                "realistic proportions"
            ]
        else:
            return [
                "an adolescent's drawing",
                "sophisticated artistic expression",
                "realistic perspective drawing"
            ]
    
    def _get_context_specific_categories(self, context: str) -> List[str]:
        """Get context-specific CLIP categories"""
        context_categories = {
            'family_drawing': [
                "a family portrait",
                "family members together",
                "family home scene"
            ],
            'house_tree_person': [
                "a house drawing",
                "a tree drawing",
                "a person drawing"
            ],
            'self_portrait': [
                "a self-portrait",
                "drawing of oneself",
                "personal representation"
            ],
            'free_drawing': [
                "creative expression",
                "imaginative drawing",
                "artistic creation"
            ]
        }
        return context_categories.get(context, [])
    
    def extract_psychological_indicators_from_clip(self, clip_results: Dict) -> Dict:
        """Extract psychological indicators from CLIP results"""
        indicators = {
            'emotional_wellbeing': 0.5,
            'developmental_level': 0.5,
            'social_connection': 0.5,
            'creativity_level': 0.5,
            'confidence_level': 0.5,
            'anxiety_indicators': 0.5
        }
        
        # Emotional wellbeing
        positive_scores = (
            clip_results.get('happy drawing', 0) + 
            clip_results.get('drawing with bright colors', 0) +
            clip_results.get('drawing showing confidence', 0)
        ) / 3
        
        negative_scores = (
            clip_results.get('sad drawing', 0) + 
            clip_results.get('drawing with dark colors', 0) +
            clip_results.get('drawing showing anxiety', 0)
        ) / 3
        
        indicators['emotional_wellbeing'] = max(0, min(1, 0.5 + (positive_scores - negative_scores)))
        
        # Developmental level
        detail_score = clip_results.get('detailed drawing', 0)
        maturity_score = clip_results.get('drawing showing developmental maturity', 0)
        delay_score = clip_results.get('drawing showing developmental delay', 0)
        
        indicators['developmental_level'] = max(0, min(1, (detail_score + maturity_score - delay_score)))
        
        # Social connection
        family_score = clip_results.get('drawing showing family', 0)
        isolation_score = clip_results.get('drawing showing isolation', 0)
        
        indicators['social_connection'] = max(0, min(1, 0.5 + (family_score - isolation_score)))
        
        # Creativity level
        indicators['creativity_level'] = clip_results.get('creative artistic drawing', 0)
        
        # Confidence level
        confidence_score = clip_results.get('drawing showing confidence', 0)
        proportion_score = clip_results.get('drawing with good proportions', 0)
        
        indicators['confidence_level'] = (confidence_score + proportion_score) / 2
        
        # Anxiety indicators
        indicators['anxiety_indicators'] = clip_results.get('drawing showing anxiety', 0)
        
        return indicators
    
    def assess_development_from_clip(self, clip_results: Dict, child_age: int) -> Dict:
        """Assess developmental level from CLIP results"""
        
        # Get expected developmental markers for age
        expected_complexity = self._get_expected_complexity_for_age(child_age)
        
        # Assess actual complexity from CLIP
        actual_complexity = (
            clip_results.get('detailed drawing', 0) +
            clip_results.get('drawing with good proportions', 0) +
            clip_results.get('drawing showing developmental maturity', 0)
        ) / 3
        
        # Compare with expectations
        if actual_complexity > expected_complexity * 1.2:
            level = 'advanced'
        elif actual_complexity > expected_complexity * 0.8:
            level = 'age_appropriate'
        else:
            level = 'needs_support'
        
        return {
            'developmental_level': level,
            'complexity_score': float(actual_complexity),
            'expected_complexity': float(expected_complexity),
            'developmental_quotient': float(actual_complexity / expected_complexity) if expected_complexity > 0 else 1.0,
            'specific_strengths': self._identify_developmental_strengths_clip(clip_results),
            'areas_for_growth': self._identify_growth_areas_clip(clip_results, child_age)
        }
    
    def assess_emotions_from_clip(self, clip_results: Dict) -> Dict:
        """Assess emotional state from CLIP results"""
        
        emotion_scores = {
            'happiness': clip_results.get('happy drawing', 0),
            'sadness': clip_results.get('sad drawing', 0),
            'anxiety': clip_results.get('drawing showing anxiety', 0),
            'confidence': clip_results.get('drawing showing confidence', 0)
        }
        
        # Determine dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        
        # Calculate emotional valence
        positive_emotions = emotion_scores['happiness'] + emotion_scores['confidence']
        negative_emotions = emotion_scores['sadness'] + emotion_scores['anxiety']
        emotional_valence = (positive_emotions - negative_emotions + 2) / 4  # Normalize to 0-1
        
        return {
            'emotion_scores': emotion_scores,
            'dominant_emotion': dominant_emotion,
            'emotional_valence': float(emotional_valence),
            'emotional_intensity': float(max(emotion_scores.values())),
            'emotional_balance': float(1 - abs(positive_emotions - negative_emotions)),
            'clinical_significance': self._assess_emotional_clinical_significance(emotion_scores)
        }
    
    def extract_psychological_elements_from_blip(self, descriptions: Dict) -> List[str]:
        """Extract psychological elements from BLIP descriptions"""
        elements = []
        
        for desc_type, description in descriptions.items():
            description_lower = description.lower()
            
            # Emotional elements
            if any(word in description_lower for word in ['happy', 'smiling', 'joyful', 'cheerful']):
                elements.append('positive_emotion')
            if any(word in description_lower for word in ['sad', 'crying', 'frowning', 'upset']):
                elements.append('negative_emotion')
            if any(word in description_lower for word in ['angry', 'mad', 'frustrated']):
                elements.append('anger_emotion')
            
            # Social elements
            if any(word in description_lower for word in ['family', 'people', 'group', 'together']):
                elements.append('social_content')
            if any(word in description_lower for word in ['alone', 'isolated', 'single', 'by themselves']):
                elements.append('isolation_content')
            
            # Environmental elements
            if any(word in description_lower for word in ['house', 'home', 'building']):
                elements.append('shelter_content')
            if any(word in description_lower for word in ['tree', 'nature', 'outdoor', 'garden']):
                elements.append('nature_content')
            
            # Developmental elements
            if any(word in description_lower for word in ['detailed', 'complex', 'realistic', 'sophisticated']):
                elements.append('advanced_development')
            if any(word in description_lower for word in ['simple', 'basic', 'primitive', 'childlike']):
                elements.append('simple_development')
            
            # Creative elements
            if any(word in description_lower for word in ['colorful', 'creative', 'artistic', 'imaginative']):
                elements.append('creative_expression')
        
        return list(set(elements))  # Remove duplicates
    
    def analyze_narrative_structure(self, descriptions: Dict) -> Dict:
        """Analyze narrative structure in descriptions"""
        narrative_elements = {
            'has_characters': False,
            'has_setting': False,
            'has_action': False,
            'narrative_complexity': 0.0
        }
        
        all_text = ' '.join(descriptions.values()).lower()
        
        # Check for characters
        if any(word in all_text for word in ['person', 'people', 'child', 'man', 'woman', 'figure']):
            narrative_elements['has_characters'] = True
        
        # Check for setting
        if any(word in all_text for word in ['house', 'tree', 'garden', 'room', 'outside', 'inside']):
            narrative_elements['has_setting'] = True
        
        # Check for action
        if any(word in all_text for word in ['playing', 'running', 'sitting', 'standing', 'holding']):
            narrative_elements['has_action'] = True
        
        # Calculate complexity
        complexity_score = sum([
            narrative_elements['has_characters'],
            narrative_elements['has_setting'],
            narrative_elements['has_action']
        ]) / 3
        
        narrative_elements['narrative_complexity'] = complexity_score
        
        return narrative_elements
    
    def rule_based_psychological_analysis(self, image_description: Dict, child_age: int, context: str) -> Dict:
        """Fallback rule-based analysis when LLM is not available"""
        
        description = image_description.get('primary_description', '').lower()
        descriptions = image_description.get('descriptions', {})
        
        # Emotional indicators
        emotional_indicators = []
        if any(word in description for word in ['bright', 'colorful', 'happy', 'smiling']):
            emotional_indicators.append('positive_mood')
        if any(word in description for word in ['dark', 'black', 'sad', 'crying']):
            emotional_indicators.append('possible_negative_mood')
        if any(word in description for word in ['angry', 'red', 'scribbled']):
            emotional_indicators.append('possible_anger')
        
        # Developmental indicators
        developmental_indicators = []
        if any(word in description for word in ['detailed', 'complex', 'realistic']):
            developmental_indicators.append('age_appropriate_detail')
        if 'simple' in description and child_age > 8:
            developmental_indicators.append('possible_developmental_consideration')
        if any(word in description for word in ['proportional', 'perspective', 'dimensional']):
            developmental_indicators.append('advanced_spatial_skills')
        
        # Social indicators
        social_indicators = []
        if any(word in description for word in ['family', 'people', 'person', 'figures']):
            social_indicators.append('social_awareness')
        if any(word in description for word in ['house', 'home', 'building']):
            social_indicators.append('environmental_awareness')
        if 'alone' in description or 'isolated' in description:
            social_indicators.append('possible_isolation_theme')
        
        # Risk assessment
        risk_indicators = []
        if len(emotional_indicators) > 0 and 'negative' in str(emotional_indicators):
            risk_indicators.append('emotional_monitoring_recommended')
        if 'developmental_consideration' in str(developmental_indicators):
            risk_indicators.append('developmental_assessment_recommended')
        
        # Generate interpretation
        interpretation = self.generate_rule_based_interpretation(
            emotional_indicators, developmental_indicators, social_indicators, child_age, context
        )
        
        return {
            'emotional_indicators': emotional_indicators,
            'developmental_indicators': developmental_indicators,
            'social_indicators': social_indicators,
            'risk_indicators': risk_indicators,
            'confidence': 0.6,
            'model': 'Rule-based',
            'interpretation': interpretation,
            'structured_analysis': {
                'emotional_assessment': f"Emotional indicators: {', '.join(emotional_indicators)}",
                'developmental_assessment': f"Developmental markers: {', '.join(developmental_indicators)}",
                'social_assessment': f"Social elements: {', '.join(social_indicators)}",
                'recommendations': self._generate_rule_based_recommendations(risk_indicators, child_age)
            }
        }
    
    def calculate_ai_agreement(self, analyses: Dict) -> float:
        """Calculate agreement between different AI analyses"""
        
        # Extract comparable metrics from different analyses
        clip_emotional = analyses.get('clip_analysis', {}).get('emotional_assessment', {}).get('emotional_valence', 0.5)
        sentiment_emotional = analyses.get('sentiment_analysis', {}).get('emotional_valence', 0.5)
        
        # Calculate agreement on emotional assessment
        emotional_agreement = 1 - abs(clip_emotional - sentiment_emotional)
        
        # Extract confidence scores
        confidences = []
        for analysis_type in ['clip_analysis', 'blip_analysis', 'llm_psychological']:
            if analysis_type in analyses:
                conf = analyses[analysis_type].get('confidence', 0.5)
                confidences.append(conf)
        
        # Agreement based on confidence consistency
        if len(confidences) > 1:
            confidence_agreement = 1 - (np.std(confidences) / np.mean(confidences)) if np.mean(confidences) > 0 else 0.5
        else:
            confidence_agreement = 0.5
        
        # Overall agreement
        overall_agreement = (emotional_agreement + confidence_agreement) / 2
        
        return float(np.clip(overall_agreement, 0, 1))
    
    def structure_llm_analysis(self, interpretation: str) -> Dict:
        """Structure LLM analysis into categories"""
        
        structured = {
            'emotional_assessment': '',
            'developmental_assessment': '',
            'social_assessment': '',
            'risk_assessment': '',
            'strengths_assessment': '',
            'recommendations': ''
        }
        
        # Simple keyword-based structuring
        lines = interpretation.split('\n')
        current_category = 'emotional_assessment'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Determine category based on keywords
            if any(word in line.lower() for word in ['emotional', 'feeling', 'mood', 'affect']):
                current_category = 'emotional_assessment'
            elif any(word in line.lower() for word in ['developmental', 'development', 'age', 'cognitive']):
                current_category = 'developmental_assessment'
            elif any(word in line.lower() for word in ['social', 'family', 'relationship', 'peer']):
                current_category = 'social_assessment'
            elif any(word in line.lower() for word in ['risk', 'concern', 'worry', 'flag']):
                current_category = 'risk_assessment'
            elif any(word in line.lower() for word in ['strength', 'positive', 'talent', 'ability']):
                current_category = 'strengths_assessment'
            elif any(word in line.lower() for word in ['recommend', 'suggest', 'should', 'consider']):
                current_category = 'recommendations'
            
            structured[current_category] += line + ' '
        
        # Clean up empty categories
        for key in structured:
            structured[key] = structured[key].strip()
        
        return structured
    
    def extract_clinical_indicators_from_llm(self, interpretation: str) -> Dict:
        """Extract clinical indicators from LLM interpretation"""
        
        clinical_indicators = {
            'depression_risk': 'low',
            'anxiety_risk': 'low',
            'trauma_indicators': 'none',
            'attachment_concerns': 'none',
            'developmental_concerns': 'none'
        }
        
        interpretation_lower = interpretation.lower()
        
        # Depression indicators
        if any(word in interpretation_lower for word in ['depression', 'depressed', 'hopeless', 'worthless']):
            clinical_indicators['depression_risk'] = 'moderate'
        if any(word in interpretation_lower for word in ['severe depression', 'major depression', 'suicidal']):
            clinical_indicators['depression_risk'] = 'high'
        
        # Anxiety indicators
        if any(word in interpretation_lower for word in ['anxiety', 'anxious', 'worried', 'fearful']):
            clinical_indicators['anxiety_risk'] = 'moderate'
        if any(word in interpretation_lower for word in ['severe anxiety', 'panic', 'phobia']):
            clinical_indicators['anxiety_risk'] = 'high'
        
        # Trauma indicators
        if any(word in interpretation_lower for word in ['trauma', 'traumatic', 'abuse', 'neglect']):
            clinical_indicators['trauma_indicators'] = 'present'
        
        # Attachment concerns
        if any(word in interpretation_lower for word in ['attachment', 'insecure', 'avoidant', 'disorganized']):
            clinical_indicators['attachment_concerns'] = 'present'
        
        # Developmental concerns
        if any(word in interpretation_lower for word in ['delay', 'behind', 'immature', 'regression']):
            clinical_indicators['developmental_concerns'] = 'present'
        
        return clinical_indicators
    
    def assess_risk_from_llm(self, interpretation: str, child_age: int) -> Dict:
        """Assess risk level from LLM interpretation"""
        
        risk_assessment = {
            'overall_risk': 'low',
            'immediate_concerns': False,
            'follow_up_needed': False,
            'professional_referral': False,
            'risk_factors': []
        }
        
        interpretation_lower = interpretation.lower()
        
        # High-risk indicators
        high_risk_terms = ['severe', 'critical', 'urgent', 'immediate', 'crisis', 'suicidal', 'self-harm']
        if any(term in interpretation_lower for term in high_risk_terms):
            risk_assessment['overall_risk'] = 'high'
            risk_assessment['immediate_concerns'] = True
            risk_assessment['professional_referral'] = True
        else:
        # Moderate-risk indicators
            moderate_risk_terms = ['concerning', 'worrying', 'significant', 'notable', 'marked']
            if any(term in interpretation_lower for term in moderate_risk_terms):
                risk_assessment['overall_risk'] = 'moderate'
                risk_assessment['follow_up_needed'] = True
        
        # Extract specific risk factors
        risk_factors = []
        if 'depression' in interpretation_lower:
            risk_factors.append('depression_indicators')
        if 'anxiety' in interpretation_lower:
            risk_factors.append('anxiety_indicators')
        if 'trauma' in interpretation_lower:
            risk_factors.append('trauma_indicators')
        if 'isolation' in interpretation_lower:
            risk_factors.append('social_isolation')
        
        risk_assessment['risk_factors'] = risk_factors
        
        return risk_assessment
    
    def extract_recommendations_from_llm(self, interpretation: str) -> List[str]:
        """Extract recommendations from LLM interpretation"""
        
        recommendations = []
        
        # Split into sentences and look for recommendation patterns
        sentences = interpretation.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(word in sentence.lower() for word in ['recommend', 'suggest', 'should', 'consider', 'encourage']):
                # Clean up the recommendation
                if sentence and len(sentence) > 10:
                    recommendations.append(sentence)
        
        # Add default recommendations if none found
        if not recommendations:
            recommendations = [
                "Continue monitoring child's emotional and developmental progress",
                "Encourage continued artistic expression and creativity",
                "Provide supportive and nurturing environment"
            ]
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def extract_developmental_insights(self, interpretation: str, child_age: int) -> Dict:
        """Extract developmental insights from LLM interpretation"""
        
        insights = {
            'developmental_level': 'age_appropriate',
            'cognitive_indicators': [],
            'motor_skills': 'typical',
            'creative_development': 'typical'
        }
        
        interpretation_lower = interpretation.lower()
        
        # Developmental level assessment
        if any(word in interpretation_lower for word in ['advanced', 'above average', 'gifted', 'precocious']):
            insights['developmental_level'] = 'advanced'
        elif any(word in interpretation_lower for word in ['delayed', 'behind', 'immature', 'below average']):
            insights['developmental_level'] = 'delayed'
        
        # Cognitive indicators
        if 'problem solving' in interpretation_lower:
            insights['cognitive_indicators'].append('problem_solving_skills')
        if 'attention' in interpretation_lower:
            insights['cognitive_indicators'].append('attention_skills')
        if 'memory' in interpretation_lower:
            insights['cognitive_indicators'].append('memory_skills')
        
        # Motor skills
        if any(word in interpretation_lower for word in ['fine motor', 'coordination', 'dexterity']):
            if 'good' in interpretation_lower or 'strong' in interpretation_lower:
                insights['motor_skills'] = 'strong'
            elif 'poor' in interpretation_lower or 'weak' in interpretation_lower:
                insights['motor_skills'] = 'needs_support'
        
        # Creative development
        if any(word in interpretation_lower for word in ['creative', 'imaginative', 'artistic']):
            if 'highly' in interpretation_lower or 'very' in interpretation_lower:
                insights['creative_development'] = 'exceptional'
        
        return insights
    
    def generate_rule_based_interpretation(self, emotional_indicators, developmental_indicators, social_indicators, child_age, context) -> str:
        """Generate rule-based interpretation"""
        
        interpretation = f"Analysis of {context} drawing by {child_age}-year-old child:\n\n"
        
        if emotional_indicators:
            interpretation += f"Emotional indicators suggest: {', '.join(emotional_indicators)}.\n"
        
        if developmental_indicators:
            interpretation += f"Developmental markers show: {', '.join(developmental_indicators)}.\n"
        
        if social_indicators:
            interpretation += f"Social elements include: {', '.join(social_indicators)}.\n"
        
        interpretation += "\nOverall, this drawing reflects typical childhood expression with individual characteristics."
        
        return interpretation
    
    def _calculate_overall_sentiment(self, sentiment_results: Dict) -> Dict:
        """Calculate overall sentiment from individual results"""
        if not sentiment_results:
            return {'label': 'neutral', 'score': 0.5}
    
        positive_scores = []
        negative_scores = []
    
        for result in sentiment_results.values():
            # ADD TYPE CHECKING:
            if isinstance(result, dict) and 'label' in result and 'score' in result:
                if result['label'] in ['POSITIVE', 'positive']:
                    positive_scores.append(result['score'])
                elif result['label'] in ['NEGATIVE', 'negative']:
                    negative_scores.append(result['score'])
    
        avg_positive = np.mean(positive_scores) if positive_scores else 0
        avg_negative = np.mean(negative_scores) if negative_scores else 0
    
        if avg_positive > avg_negative:
            return {'label': 'positive', 'score': avg_positive}
        elif avg_negative > avg_positive:
            return {'label': 'negative', 'score': avg_negative}
        else:
            return {'label': 'neutral', 'score': 0.5}

    
    def _map_sentiment_to_psychology(self, overall_sentiment: Dict) -> Dict:
        """Map sentiment analysis to psychological constructs"""
        
        sentiment_label = overall_sentiment.get('label', 'neutral')
        sentiment_score = overall_sentiment.get('score', 0.5)
        
        psychological_mapping = {
            'emotional_valence': 0.5,
            'mood_indicators': 'neutral',
            'affect_quality': 'balanced'
        }
        
        if sentiment_label == 'positive':
            psychological_mapping['emotional_valence'] = 0.5 + (sentiment_score * 0.5)
            psychological_mapping['mood_indicators'] = 'positive'
            psychological_mapping['affect_quality'] = 'elevated'
        elif sentiment_label == 'negative':
            psychological_mapping['emotional_valence'] = 0.5 - (sentiment_score * 0.5)
            psychological_mapping['mood_indicators'] = 'concerning'
            psychological_mapping['affect_quality'] = 'depressed'
        
        return psychological_mapping
    
    def _calculate_emotional_valence(self, sentiment_results: Dict) -> float:
        """Calculate emotional valence score"""
        overall_sentiment = self._calculate_overall_sentiment(sentiment_results)
        
        if overall_sentiment['label'] == 'positive':
            return 0.5 + (overall_sentiment['score'] * 0.5)
        elif overall_sentiment['label'] == 'negative':
            return 0.5 - (overall_sentiment['score'] * 0.5)
        else:
            return 0.5
    
    def _calculate_emotional_intensity(self, sentiment_results: Dict) -> float:
        """Calculate emotional intensity score"""
        if not sentiment_results:
            return 0.5
        
        scores = [result['score'] for result in sentiment_results.values()]
        return float(np.mean(scores))
    
    def _determine_consensus_emotional_state(self, clip_analysis, blip_analysis, llm_analysis, sentiment_analysis) -> str:
        """Determine consensus emotional state from all analyses"""
        
        emotional_indicators = []
        
        # From CLIP
        if clip_analysis and 'emotional_assessment' in clip_analysis:
            dominant_emotion = clip_analysis['emotional_assessment'].get('dominant_emotion', 'neutral')
            emotional_indicators.append(dominant_emotion)
        
        # From sentiment analysis
        if sentiment_analysis and 'overall_sentiment' in sentiment_analysis:
            sentiment_label = sentiment_analysis['overall_sentiment'].get('label', 'neutral')
            emotional_indicators.append(sentiment_label)
        
        # From LLM
        if llm_analysis and 'structured_analysis' in llm_analysis:
            emotional_assessment = llm_analysis['structured_analysis'].get('emotional_assessment', '')
            if 'positive' in emotional_assessment.lower():
                emotional_indicators.append('positive')
            elif 'negative' in emotional_assessment.lower() or 'concern' in emotional_assessment.lower():
                emotional_indicators.append('concerning')
            else:
                emotional_indicators.append('neutral')
        
        # Determine consensus
        if not emotional_indicators:
            return 'neutral'
        
        # Count occurrences
        positive_count = sum(1 for indicator in emotional_indicators if indicator in ['happy', 'positive', 'POSITIVE'])
        negative_count = sum(1 for indicator in emotional_indicators if indicator in ['sad', 'negative', 'concerning', 'NEGATIVE'])
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'concerning'
        else:
            return 'neutral'
    
    def _determine_consensus_developmental_level(self, clip_analysis, llm_analysis, child_age) -> str:
        """Determine consensus developmental level"""
        
        developmental_indicators = []
        
        # From CLIP
        if clip_analysis and 'developmental_assessment' in clip_analysis:
            clip_level = clip_analysis['developmental_assessment'].get('developmental_level', 'age_appropriate')
            developmental_indicators.append(clip_level)
        
        # From LLM
        if llm_analysis and 'developmental_insights' in llm_analysis:
            llm_level = llm_analysis['developmental_insights'].get('developmental_level', 'age_appropriate')
            developmental_indicators.append(llm_level)
        
        # Determine consensus
        if not developmental_indicators:
            return 'age_appropriate'
        
        # Count occurrences
        advanced_count = developmental_indicators.count('advanced')
        delayed_count = developmental_indicators.count('delayed') + developmental_indicators.count('needs_support')
        appropriate_count = developmental_indicators.count('age_appropriate')
        
        if advanced_count > max(delayed_count, appropriate_count):
            return 'advanced'
        elif delayed_count > max(advanced_count, appropriate_count):
            return 'needs_support'
        else:
            return 'age_appropriate'
    
    def _determine_consensus_risk_level(self, clip_analysis, llm_analysis, sentiment_analysis) -> Dict:
        """Determine consensus risk level"""
        
        risk_indicators = []
        
        # From CLIP
        if clip_analysis and 'clinical_flags' in clip_analysis:
            if clip_analysis['clinical_flags']:
                risk_indicators.append('moderate')
            else:
                risk_indicators.append('low')
        
        # From LLM
        if llm_analysis and 'risk_assessment' in llm_analysis:
            llm_risk = llm_analysis['risk_assessment'].get('overall_risk', 'low')
            risk_indicators.append(llm_risk)
        
        # From sentiment
        if sentiment_analysis and 'overall_sentiment' in sentiment_analysis:
            sentiment_label = sentiment_analysis['overall_sentiment'].get('label', 'neutral')
            if sentiment_label == 'negative':
                risk_indicators.append('moderate')
            else:
                risk_indicators.append('low')
        
        # Determine consensus
        if not risk_indicators:
            return {'level': 'low', 'confidence': 0.5}
        
        # Take the highest risk level
        if 'high' in risk_indicators:
            return {'level': 'high', 'confidence': 0.8}
        elif 'moderate' in risk_indicators:
            return {'level': 'moderate', 'confidence': 0.7}
        else:
            return {'level': 'low', 'confidence': 0.6}
    
    def _identify_consensus_strengths(self, clip_analysis, llm_analysis) -> List[str]:
        """Identify consensus strengths"""
        
        strengths = []
        
        # From CLIP
        if clip_analysis and 'psychological_indicators' in clip_analysis:
            indicators = clip_analysis['psychological_indicators']
            if indicators.get('creativity_level', 0) > 0.7:
                strengths.append('high_creativity')
            if indicators.get('confidence_level', 0) > 0.7:
                strengths.append('self_confidence')
            if indicators.get('emotional_wellbeing', 0) > 0.7:
                strengths.append('emotional_resilience')
        
        # From LLM
        if llm_analysis and 'structured_analysis' in llm_analysis:
            strengths_text = llm_analysis['structured_analysis'].get('strengths_assessment', '')
            if 'creative' in strengths_text.lower():
                strengths.append('creative_expression')
            if 'confident' in strengths_text.lower():
                strengths.append('self_assurance')
            if 'social' in strengths_text.lower():
                strengths.append('social_skills')
        
        return list(set(strengths))  # Remove duplicates
    
    def _calculate_consensus_confidence(self, analyses: Dict) -> Dict:
        """Calculate consensus confidence scores"""
        
        confidence_scores = {}
        
        # Extract individual confidences
        for analysis_type, analysis_data in analyses.items():
            if isinstance(analysis_data, dict) and 'confidence' in analysis_data:
                confidence_scores[analysis_type] = analysis_data['confidence']
        
        return confidence_scores
    
    def _get_models_used(self, analyses: Dict) -> List[str]:
        """Get list of models used in analysis"""
        models = []
        
        if 'clip_analysis' in analyses and analyses['clip_analysis']:
            models.append('CLIP')
        if 'blip_analysis' in analyses and analyses['blip_analysis']:
            models.append('BLIP')
        if 'llm_psychological' in analyses and analyses['llm_psychological']:
            models.append('LLM')
        if 'sentiment_analysis' in analyses and analyses['sentiment_analysis']:
            models.append('Sentiment_Analyzer')
        
        return models
    
    def _categorize_consensus_strength(self, agreement_score: float) -> str:
        """Categorize consensus strength"""
        if agreement_score > 0.8:
            return 'strong'
        elif agreement_score > 0.6:
            return 'moderate'
        else:
            return 'weak'
    
    def _generate_consensus_recommendations(self, emotional_state, developmental_assessment, risk_assessment, child_age) -> List[str]:
        """Generate consensus-based recommendations"""
        
        recommendations = []
        
        # Emotional recommendations
        if emotional_state == 'concerning':
            recommendations.extend([
                "Monitor emotional well-being closely",
                "Provide additional emotional support and validation",
                "Consider consultation with school counselor or child psychologist"
            ])
        elif emotional_state == 'positive':
            recommendations.extend([
                "Continue fostering positive emotional expression",
                "Encourage creative activities and artistic expression"
            ])
        
        # Developmental recommendations
        if developmental_assessment == 'advanced':
            recommendations.extend([
                "Provide enrichment opportunities to challenge abilities",
                "Consider advanced art or creative programs"
            ])
        elif developmental_assessment == 'needs_support':
            recommendations.extend([
                "Provide additional developmental support",
                "Consider occupational therapy assessment for fine motor skills"
            ])
        
        # Age-specific recommendations
        if child_age < 6:
            recommendations.append("Focus on play-based learning and exploration")
        elif child_age < 12:
            recommendations.append("Encourage structured creative activities")
        else:
            recommendations.append("Support identity development through artistic expression")
        
        return recommendations
    
    def _assess_clinical_significance(self, risk_assessment: Dict) -> str:
        """Assess clinical significance of findings"""
        risk_level = risk_assessment.get('level', 'low')
        
        if risk_level == 'high':
            return 'clinically_significant'
        elif risk_level == 'moderate':
            return 'subclinical'
        else:
            return 'not_significant'
    
    def _determine_follow_up_needs(self, risk_assessment: Dict, developmental_assessment: str) -> bool:
        """Determine if follow-up is needed"""
        risk_level = risk_assessment.get('level', 'low')
        
        return (risk_level in ['high', 'moderate'] or 
                developmental_assessment in ['needs_support', 'delayed'])
    
    # Placeholder methods for remaining functionality
    def _assess_age_appropriateness_clip(self, clip_results: Dict, child_age: int) -> str:
        """Assess age appropriateness from CLIP results"""
        return 'age_appropriate'
    
    def _detect_clinical_flags_clip(self, clip_results: Dict) -> List[str]:
        """Detect clinical flags from CLIP results"""
        flags = []
        
        # Check for concerning patterns
        if clip_results.get('drawing showing anxiety', 0) > 0.7:
            flags.append('high_anxiety_indicators')
        if clip_results.get('sad drawing', 0) > 0.7:
            flags.append('depressive_indicators')
        if clip_results.get('drawing showing isolation', 0) > 0.6:
            flags.append('social_isolation_concerns')
        
        return flags
    
    def _get_expected_complexity_for_age(self, child_age: int) -> float:
        """Get expected complexity for child's age"""
        return min(child_age / 10.0, 1.0)
    
    def _identify_developmental_strengths_clip(self, clip_results: Dict) -> List[str]:
        """Identify developmental strengths from CLIP"""
        strengths = []
        
        if clip_results.get('detailed drawing', 0) > 0.7:
            strengths.append('high_detail_orientation')
        if clip_results.get('drawing with good proportions', 0) > 0.6:
            strengths.append('spatial_awareness')
        if clip_results.get('creative artistic drawing', 0) > 0.6:
            strengths.append('creative_expression')
        if clip_results.get('drawing showing developmental maturity', 0) > 0.5:
            strengths.append('age_appropriate_development')
        
        return strengths
    
    def _identify_growth_areas_clip(self, clip_results: Dict, child_age: int) -> List[str]:
        """Identify areas for growth from CLIP"""
        growth_areas = []
        
        if clip_results.get('simple drawing', 0) > 0.7 and child_age > 8:
            growth_areas.append('complexity_development')
        if clip_results.get('drawing with poor proportions', 0) > 0.6:
            growth_areas.append('spatial_skills')
        if clip_results.get('drawing showing developmental delay', 0) > 0.5:
            growth_areas.append('developmental_support')
        
        return growth_areas
    
    def _assess_emotional_clinical_significance(self, emotion_scores: Dict) -> str:
        """Assess clinical significance of emotional indicators"""
        max_negative = max(emotion_scores.get('sadness', 0), emotion_scores.get('anxiety', 0))
        max_positive = max(emotion_scores.get('happiness', 0), emotion_scores.get('confidence', 0))
        
        if max_negative > 0.8:
            return 'clinically_significant'
        elif max_negative > 0.6:
            return 'moderate_concern'
        elif max_positive > 0.7:
            return 'positive_indicators'
        else:
            return 'within_normal_range'
    
    def _assess_description_complexity(self, descriptions: Dict) -> Dict:
        """Assess complexity of BLIP descriptions"""
        complexity_metrics = {
            'word_count': 0,
            'unique_words': 0,
            'complexity_score': 0.0,
            'descriptive_richness': 0.0
        }
        
        all_text = ' '.join(descriptions.values())
        words = all_text.split()
        
        complexity_metrics['word_count'] = len(words)
        complexity_metrics['unique_words'] = len(set(words))
        
        # Calculate complexity score
        if len(words) > 0:
            complexity_metrics['complexity_score'] = len(set(words)) / len(words)
        
        # Descriptive richness based on adjectives and descriptive words
        descriptive_words = ['detailed', 'colorful', 'bright', 'large', 'small', 'beautiful', 'creative']
        descriptive_count = sum(1 for word in words if word.lower() in descriptive_words)
        
        if len(words) > 0:
            complexity_metrics['descriptive_richness'] = descriptive_count / len(words)
        
        return complexity_metrics
    
    def _extract_emotional_content_blip(self, descriptions: Dict) -> Dict:
        """Extract emotional content from BLIP descriptions"""
        emotional_content = {
            'positive_emotions': [],
            'negative_emotions': [],
            'neutral_emotions': [],
            'emotional_intensity': 0.0
        }
        
        positive_words = ['happy', 'joyful', 'cheerful', 'bright', 'colorful', 'smiling', 'playful']
        negative_words = ['sad', 'dark', 'crying', 'angry', 'lonely', 'scared', 'worried']
        neutral_words = ['calm', 'peaceful', 'quiet', 'simple', 'plain']
        
        all_text = ' '.join(descriptions.values()).lower()
        words = all_text.split()
        
        for word in words:
            if word in positive_words:
                emotional_content['positive_emotions'].append(word)
            elif word in negative_words:
                emotional_content['negative_emotions'].append(word)
            elif word in neutral_words:
                emotional_content['neutral_emotions'].append(word)
        
        # Calculate emotional intensity
        total_emotional_words = (
            len(emotional_content['positive_emotions']) + 
            len(emotional_content['negative_emotions'])
        )
        
        if len(words) > 0:
            emotional_content['emotional_intensity'] = total_emotional_words / len(words)
        
        return emotional_content
    
    def _generate_rule_based_recommendations(self, risk_indicators: List[str], child_age: int) -> List[str]:
        """Generate rule-based recommendations"""
        recommendations = []
        
        if 'emotional_monitoring_recommended' in risk_indicators:
            recommendations.extend([
                "Monitor emotional well-being and provide supportive environment",
                "Encourage emotional expression through art and play",
                "Consider consultation with school counselor if concerns persist"
            ])
        
        if 'developmental_assessment_recommended' in risk_indicators:
            recommendations.extend([
                "Consider developmental assessment with qualified professional",
                "Provide age-appropriate developmental activities",
                "Monitor progress and adjust support as needed"
            ])
        
        # Age-specific recommendations
        if child_age < 5:
            recommendations.append("Focus on sensory play and exploration activities")
        elif child_age < 10:
            recommendations.append("Encourage structured creative activities and skill building")
        else:
            recommendations.append("Support identity development and self-expression")
        
        if not risk_indicators:
            recommendations.extend([
                "Continue encouraging creative expression and artistic development",
                "Maintain supportive and nurturing environment",
                "Celebrate child's unique artistic voice and creativity"
            ])
        
        return recommendations
    
    def _estimate_reliability(self, analyses: Dict) -> float:
        """Estimate reliability of AI analyses"""
        reliability_factors = []
        
        # Model availability factor
        available_models = len([a for a in analyses.values() if a and not a.get('error')])
        total_models = len(analyses)
        model_availability = available_models / total_models if total_models > 0 else 0
        reliability_factors.append(model_availability)
        
        # Confidence consistency factor
        confidences = []
        for analysis in analyses.values():
            if isinstance(analysis, dict) and 'confidence' in analysis:
                confidences.append(analysis['confidence'])
        
        if len(confidences) > 1:
            confidence_std = np.std(confidences)
            confidence_consistency = 1 - min(confidence_std, 1.0)  # Normalize to 0-1
            reliability_factors.append(confidence_consistency)
        
        # Agreement factor
        agreement_score = self.calculate_ai_agreement(analyses)
        reliability_factors.append(agreement_score)
        
        # Overall reliability
        overall_reliability = np.mean(reliability_factors) if reliability_factors else 0.5
        
        return float(overall_reliability)
    
    def _quantify_uncertainty(self, analyses: Dict) -> Dict:
        """Quantify uncertainty in AI analyses"""
        uncertainty_metrics = {
            'model_disagreement': 0.0,
            'confidence_variance': 0.0,
            'missing_data_penalty': 0.0,
            'overall_uncertainty': 0.0
        }
        
        # Model disagreement
        agreement_score = self.calculate_ai_agreement(analyses)
        uncertainty_metrics['model_disagreement'] = 1 - agreement_score
        
        # Confidence variance
        confidences = []
        for analysis in analyses.values():
            if isinstance(analysis, dict) and 'confidence' in analysis:
                confidences.append(analysis['confidence'])
        
        if confidences:
            uncertainty_metrics['confidence_variance'] = float(np.var(confidences))
        
        # Missing data penalty
        available_analyses = len([a for a in analyses.values() if a and not a.get('error')])
        total_expected = 4  # CLIP, BLIP, LLM, Sentiment
        missing_penalty = (total_expected - available_analyses) / total_expected
        uncertainty_metrics['missing_data_penalty'] = missing_penalty
        
        # Overall uncertainty
        uncertainty_metrics['overall_uncertainty'] = np.mean([
            uncertainty_metrics['model_disagreement'],
            uncertainty_metrics['confidence_variance'],
            uncertainty_metrics['missing_data_penalty']
        ])
        
        return uncertainty_metrics

# Additional utility functions for the AI analysis engine

def create_ai_analysis_report(analyses: Dict, child_info: Dict) -> Dict:
    """Create a comprehensive AI analysis report"""
    
    report = {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(),
            'child_age': child_info.get('age', 'Unknown'),
            'drawing_context': child_info.get('context', 'Unknown'),
            'analysis_version': '1.0.0'
        },
        'executive_summary': {},
        'detailed_findings': analyses,
        'recommendations': [],
        'confidence_assessment': {},
        'follow_up_actions': []
    }
    
    # Generate executive summary
    consensus = analyses.get('consensus_analysis', {})
    if consensus:
        report['executive_summary'] = {
            'primary_emotional_state': consensus.get('primary_emotional_state', 'Unknown'),
            'developmental_level': consensus.get('developmental_assessment', 'Unknown'),
            'risk_level': consensus.get('risk_assessment', {}).get('level', 'Unknown'),
            'overall_confidence': consensus.get('confidence_level', 0.0),
            'key_strengths': consensus.get('strengths_assessment', []),
            'areas_of_concern': []
        }
        
        # Extract areas of concern
        if consensus.get('risk_assessment', {}).get('level') in ['moderate', 'high']:
            report['executive_summary']['areas_of_concern'].append('Elevated risk indicators detected')
        
        if consensus.get('developmental_assessment') == 'needs_support':
            report['executive_summary']['areas_of_concern'].append('Developmental support recommended')
    
    # Compile recommendations
    if consensus:
        report['recommendations'] = consensus.get('recommendations', [])
    
    # Confidence assessment
    confidence_metrics = analyses.get('ai_confidence_metrics', {})
    if confidence_metrics:
        report['confidence_assessment'] = {
            'overall_confidence': confidence_metrics.get('overall_confidence', 0.0),
            'model_agreement': confidence_metrics.get('agreement_based_confidence', 0.0),
            'reliability_estimate': confidence_metrics.get('reliability_estimate', 0.0),
            'uncertainty_level': confidence_metrics.get('uncertainty_metrics', {}).get('overall_uncertainty', 0.0)
        }
    
    # Follow-up actions
    if consensus and consensus.get('follow_up_needed', False):
        report['follow_up_actions'].extend([
            'Schedule follow-up assessment in 3-6 months',
            'Monitor progress and development',
            'Implement recommended interventions'
        ])
        
        if consensus.get('risk_assessment', {}).get('level') == 'high':
            report['follow_up_actions'].insert(0, 'Immediate professional consultation recommended')
    
    return report

def validate_ai_analysis_input(image_path: str, child_age: int, context: str) -> Dict:
    """Validate input parameters for AI analysis"""
    
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate image path
    if not image_path or not os.path.exists(image_path):
        validation_result['is_valid'] = False
        validation_result['errors'].append('Invalid or missing image path')
    
    # Validate child age
    if not isinstance(child_age, int) or child_age < 1 or child_age > 18:
        validation_result['is_valid'] = False
        validation_result['errors'].append('Child age must be between 1 and 18 years')
    
    # Validate context
    valid_contexts = [
        'free_drawing', 'family_drawing', 'house_tree_person', 
        'self_portrait', 'emotional_expression', 'story_illustration', 
        'school_assignment'
    ]
    
    if context not in valid_contexts:
        validation_result['warnings'].append(f'Unknown context: {context}. Using default analysis.')
    
    # Age-context compatibility warnings
    if child_age < 4 and context in ['house_tree_person', 'self_portrait']:
        validation_result['warnings'].append(
            f'Context "{context}" may not be appropriate for age {child_age}'
        )
    
    return validation_result

# Export the main class and utility functions
__all__ = [
    'ComprehensiveAIAnalyzer',
    'create_ai_analysis_report', 
    'validate_ai_analysis_input'
]
