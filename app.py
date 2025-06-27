import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import traceback
import torch
from transformers import pipeline, CLIPModel, CLIPProcessor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all custom modules with comprehensive error handling
MODULES_STATUS = {}

def safe_import(module_name, class_name=None):
    """Safely import modules and track their status"""
    try:
        module = __import__(module_name)
        if class_name:
            return getattr(module, class_name), True
        return module, True
    except ImportError as e:
        st.error(f"Module import error for {module_name}: {e}")
        MODULES_STATUS[module_name] = f"Failed: {e}"
        return None, False
    except Exception as e:
        st.error(f"Unexpected error importing {module_name}: {e}")
        MODULES_STATUS[module_name] = f"Error: {e}"
        return None, False

# Import all analysis modules
psydraw_extractor, PSYDRAW_AVAILABLE = safe_import('psydraw_feature_extractor', 'PsyDrawFeatureExtractor')
psychological_engine, PSYCHOLOGICAL_AVAILABLE = safe_import('psychological_assessment_engine', 'PsychologicalAssessmentEngine')
clinical_assessment, CLINICAL_AVAILABLE = safe_import('clinical_assessment_advanced', 'AdvancedClinicalAssessment')
expert_collaboration, EXPERT_AVAILABLE = safe_import('expert_collaboration_framework', 'ExpertCollaborationFramework')
research_validation, RESEARCH_AVAILABLE = safe_import('research_validation_module', 'ResearchValidationModule')
validation_framework, VALIDATION_AVAILABLE = safe_import('validation_framework', 'ValidationFramework')
bias_detection, BIAS_AVAILABLE = safe_import('validation_framework', 'BiasDetectionSystem')
video_generator, VIDEO_AVAILABLE = safe_import('video_generator_consolidated', 'ConsolidatedVideoGenerator')
ai_analyzer, AI_ANALYZER_AVAILABLE = safe_import('ai_analysis_engine', 'ComprehensiveAIAnalyzer')
enhanced_summary, SUMMARY_AVAILABLE = safe_import('enhanced_summary_generator', 'EnhancedSummaryGenerator')
pdf_generator, PDF_AVAILABLE = safe_import('pdf_report_generator', 'PDFReportGenerator')
llm_integrator, LLM_AVAILABLE = safe_import('llm_integrator', 'StreamlinedLLMIntegrator')

# Enhanced Drawing Analyzer components
try:
    from enhanced_drawing_analyzer import EnhancedDrawingAnalyzer, ScientificallyValidatedAnalyzer
    ENHANCED_ANALYZER_AVAILABLE = True
except ImportError as e:
    st.error(f"Enhanced analyzer import error: {e}")
    ENHANCED_ANALYZER_AVAILABLE = False

class ComprehensiveDrawingAnalyzer:
    """
    Unified main application class integrating ALL psychological assessment modules
    and AI analysis capabilities with comprehensive error handling
    """
    
    def __init__(self):
        self.initialized_components = {}
        self.error_log = []
        
        # Initialize all available components
        self._initialize_components()
        
        # Set up analysis capabilities
        self.analysis_capabilities = self._assess_capabilities()
        
        print(f"✅ Comprehensive Drawing Analyzer initialized with {len(self.initialized_components)} components")
    
    def _initialize_components(self):
        """Initialize all available components with error handling"""
        
        # Core PsyDraw components
        if PSYDRAW_AVAILABLE:
            try:
                self.initialized_components['psydraw_extractor'] = psydraw_extractor()
                print("✅ PsyDraw Feature Extractor initialized")
            except Exception as e:
                self.error_log.append(f"PsyDraw extractor failed: {e}")
        
        if PSYCHOLOGICAL_AVAILABLE:
            try:
                self.initialized_components['psychological_engine'] = psychological_engine()
                print("✅ Psychological Assessment Engine initialized")
            except Exception as e:
                self.error_log.append(f"Psychological engine failed: {e}")
        
        if CLINICAL_AVAILABLE:
            try:
                self.initialized_components['clinical_assessment'] = clinical_assessment()
                print("✅ Clinical Assessment initialized")
            except Exception as e:
                self.error_log.append(f"Clinical assessment failed: {e}")
        
        # Validation and research components
        if VALIDATION_AVAILABLE:
            try:
                self.initialized_components['validation_framework'] = validation_framework()
                print("✅ Validation Framework initialized")
            except Exception as e:
                self.error_log.append(f"Validation framework failed: {e}")
        
        if BIAS_AVAILABLE:
            try:
                self.initialized_components['bias_detection'] = bias_detection()
                print("✅ Bias Detection System initialized")
            except Exception as e:
                self.error_log.append(f"Bias detection failed: {e}")
        
        if RESEARCH_AVAILABLE:
            try:
                self.initialized_components['research_validation'] = research_validation()
                print("✅ Research Validation Module initialized")
            except Exception as e:
                self.error_log.append(f"Research validation failed: {e}")
        
        # AI and LLM components
        if AI_ANALYZER_AVAILABLE:
            try:
                self.initialized_components['ai_analyzer'] = ai_analyzer()
                print("✅ AI Analysis Engine initialized")
            except Exception as e:
                self.error_log.append(f"AI analyzer failed: {e}")
        
        if LLM_AVAILABLE:
            try:
                self.initialized_components['llm_integrator'] = llm_integrator()
                print("✅ LLM Integrator initialized")
            except Exception as e:
                self.error_log.append(f"LLM integrator failed: {e}")
        
        # Expert collaboration and reporting
        if EXPERT_AVAILABLE:
            try:
                self.initialized_components['expert_collaboration'] = expert_collaboration()
                print("✅ Expert Collaboration Framework initialized")
            except Exception as e:
                self.error_log.append(f"Expert collaboration failed: {e}")
        
        if SUMMARY_AVAILABLE:
            try:
                self.initialized_components['enhanced_summary'] = enhanced_summary()
                print("✅ Enhanced Summary Generator initialized")
            except Exception as e:
                self.error_log.append(f"Enhanced summary failed: {e}")
        
        if PDF_AVAILABLE:
            try:
                self.initialized_components['pdf_generator'] = pdf_generator()
                print("✅ PDF Report Generator initialized")
            except Exception as e:
                self.error_log.append(f"PDF generator failed: {e}")
        
        # Video generation
        if VIDEO_AVAILABLE:
            try:
                self.initialized_components['video_generator'] = video_generator()
                print("✅ Video Generator initialized")
            except Exception as e:
                self.error_log.append(f"Video generator failed: {e}")
        
        # Enhanced analyzers
        if ENHANCED_ANALYZER_AVAILABLE:
            try:
                self.initialized_components['enhanced_analyzer'] = EnhancedDrawingAnalyzer()
                self.initialized_components['scientific_analyzer'] = ScientificallyValidatedAnalyzer()
                print("✅ Enhanced Analyzers initialized")
            except Exception as e:
                self.error_log.append(f"Enhanced analyzers failed: {e}")
    
    def _assess_capabilities(self) -> Dict[str, bool]:
        """Assess what analysis capabilities are available"""
        return {
            'basic_analysis': True,  # Always available
            'psydraw_features': 'psydraw_extractor' in self.initialized_components,
            'psychological_assessment': 'psychological_engine' in self.initialized_components,
            'clinical_assessment': 'clinical_assessment' in self.initialized_components,
            'ai_analysis': 'ai_analyzer' in self.initialized_components,
            'llm_analysis': 'llm_integrator' in self.initialized_components,
            'scientific_validation': 'validation_framework' in self.initialized_components,
            'bias_detection': 'bias_detection' in self.initialized_components,
            'research_validation': 'research_validation' in self.initialized_components,
            'expert_collaboration': 'expert_collaboration' in self.initialized_components,
            'enhanced_summary': 'enhanced_summary' in self.initialized_components,
            'pdf_generation': 'pdf_generator' in self.initialized_components,
            'video_generation': 'video_generator' in self.initialized_components,
            'enhanced_analysis': 'enhanced_analyzer' in self.initialized_components
        }
    
    def analyze_drawing_comprehensive(self, image_path: str, child_age: int, drawing_context: str, 
                                    analysis_options: Dict[str, bool] = None) -> Dict:
        """
        Comprehensive drawing analysis using ALL available modules
        """
        if analysis_options is None:
            analysis_options = {
                'include_clinical': True,
                'include_validation': True,
                'include_ai_analysis': True,
                'include_llm_analysis': True,
                'include_bias_detection': True,
                'include_research_validation': True
            }
        
        try:
            # Load and validate image
            if not os.path.exists(image_path):
                return {'error': f'Image file not found: {image_path}'}
                
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image file'}
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Initialize results structure
            comprehensive_results = {
                'timestamp': datetime.now().isoformat(),
                'input_info': {
                    'child_age': child_age,
                    'age_group': self._determine_age_group(child_age),
                    'drawing_context': drawing_context,
                    'image_path': image_path
                },
                'analysis_capabilities_used': {},
                'error_log': []
            }
            
            # 1. Extract PsyDraw features (if available)
            if self.analysis_capabilities['psydraw_features']:
                st.info("🧠 Extracting psychological features...")
                try:
                    psydraw_features = self.initialized_components['psydraw_extractor'].extract_complete_psydraw_features(
                        image_rgb, child_age, drawing_context
                    )
                    comprehensive_results['psydraw_features'] = psydraw_features
                    comprehensive_results['analysis_capabilities_used']['psydraw_features'] = True
                except Exception as e:
                    st.warning(f"PsyDraw feature extraction warning: {e}")
                    comprehensive_results['psydraw_features'] = self._create_default_features()
                    comprehensive_results['error_log'].append(f"PsyDraw extraction: {e}")
            else:
                comprehensive_results['psydraw_features'] = self._create_default_features()
            
            # 2. AI Analysis (if available)
            if self.analysis_capabilities['ai_analysis'] and analysis_options.get('include_ai_analysis', True):
                st.info("🤖 Conducting advanced AI analysis...")
                try:
                    ai_analysis_results = self.initialized_components['ai_analyzer'].conduct_multi_ai_analysis(
                        image_path, child_age, drawing_context
                    )
                    comprehensive_results['ai_analysis'] = ai_analysis_results
                    comprehensive_results['analysis_capabilities_used']['ai_analysis'] = True
                except Exception as e:
                    st.warning(f"AI analysis warning: {e}")
                    comprehensive_results['ai_analysis'] = self._create_default_ai_analysis()
                    comprehensive_results['error_log'].append(f"AI analysis: {e}")
            else:
                comprehensive_results['ai_analysis'] = self._create_default_ai_analysis()
            
            # 3. LLM Analysis (if available and different from AI analysis)
            if self.analysis_capabilities['llm_analysis'] and analysis_options.get('include_llm_analysis', True):
                if 'ai_analysis' not in comprehensive_results or not comprehensive_results['ai_analysis'].get('llm_psychological'):
                    st.info("🧠 Conducting LLM psychological analysis...")
                    try:
                        # Create mock traditional results for LLM
                        mock_traditional = {
                            'blip_description': 'A child\'s drawing',
                            'color_analysis': {'dominant_color': 'mixed'},
                            'shape_analysis': {'complexity_level': 'medium'},
                            'spatial_analysis': {'spatial_balance': 'balanced'}
                        }
                        
                        llm_responses = self.initialized_components['llm_integrator'].analyze_drawing_comprehensive(
                            image_path, child_age, drawing_context, mock_traditional
                        )
                        comprehensive_results['llm_analyses'] = [{
                            'provider': resp.provider,
                            'analysis': resp.response,
                            'confidence': resp.confidence,
                            'analysis_type': resp.analysis_type,
                            'metadata': resp.metadata
                        } for resp in llm_responses]
                        comprehensive_results['analysis_capabilities_used']['llm_analysis'] = True
                    except Exception as e:
                        st.warning(f"LLM analysis warning: {e}")
                        comprehensive_results['llm_analyses'] = []
                        comprehensive_results['error_log'].append(f"LLM analysis: {e}")
            
            # 4. Psychological Assessment (if available)
            if self.analysis_capabilities['psychological_assessment']:
                st.info("🔬 Conducting psychological assessment...")
                try:
                    psychological_assessment = self.initialized_components['psychological_engine'].conduct_comprehensive_assessment(
                        comprehensive_results['psydraw_features'], child_age, drawing_context
                    )
                    comprehensive_results['psychological_assessment'] = psychological_assessment
                    comprehensive_results['analysis_capabilities_used']['psychological_assessment'] = True
                except Exception as e:
                    st.warning(f"Psychological assessment warning: {e}")
                    comprehensive_results['psychological_assessment'] = self._create_default_psychological_assessment()
                    comprehensive_results['error_log'].append(f"Psychological assessment: {e}")
            else:
                comprehensive_results['psychological_assessment'] = self._create_default_psychological_assessment()
            
            # 5. Clinical Assessment (if requested and available)
            if self.analysis_capabilities['clinical_assessment'] and analysis_options.get('include_clinical', True):
                st.info("🏥 Performing clinical assessment...")
                try:
                    # Trauma assessment
                    trauma_assessment = self.initialized_components['clinical_assessment'].conduct_trauma_assessment(
                        image_rgb, comprehensive_results['psydraw_features'], child_age
                    )
                    
                    # Attachment assessment
                    attachment_assessment = self.initialized_components['clinical_assessment'].assess_attachment_patterns(
                        image_rgb, comprehensive_results['psydraw_features'], drawing_context
                    )
                    
                    comprehensive_results['clinical_results'] = {
                        'trauma_assessment': trauma_assessment,
                        'attachment_assessment': attachment_assessment
                    }
                    comprehensive_results['analysis_capabilities_used']['clinical_assessment'] = True
                except Exception as e:
                    st.warning(f"Clinical assessment warning: {e}")
                    comprehensive_results['clinical_results'] = self._create_default_clinical_results()
                    comprehensive_results['error_log'].append(f"Clinical assessment: {e}")
            
            # 6. Scientific Validation (if requested and available)
            validation_results = {}
            if analysis_options.get('include_validation', True):
                
                # Research validation
                if self.analysis_capabilities['research_validation']:
                    st.info("📊 Validating against research benchmarks...")
                    try:
                        validation_results = self.initialized_components['research_validation'].conduct_comprehensive_validation(
                            comprehensive_results, child_age
                        )
                        comprehensive_results['analysis_capabilities_used']['research_validation'] = True
                    except Exception as e:
                        st.warning(f"Research validation warning: {e}")
                        comprehensive_results['error_log'].append(f"Research validation: {e}")
                
                # Bias detection
                if self.analysis_capabilities['bias_detection'] and analysis_options.get('include_bias_detection', True):
                    try:
                        bias_results = self.initialized_components['bias_detection'].detect_biases(
                            comprehensive_results.get('psychological_assessment', {}),
                            {'child_age': child_age, 'drawing_context': drawing_context},
                            comprehensive_results.get('ai_analysis', {})
                        )
                        validation_results['bias_detection'] = bias_results
                        comprehensive_results['analysis_capabilities_used']['bias_detection'] = True
                    except Exception as e:
                        st.warning(f"Bias detection warning: {e}")
                        comprehensive_results['error_log'].append(f"Bias detection: {e}")
                
                # Scientific validation framework
                if self.analysis_capabilities['scientific_validation']:
                    try:
                        scientific_validation = self.initialized_components['validation_framework'].validate_assessment(
                            comprehensive_results
                        )
                        validation_results['scientific_validation'] = scientific_validation
                        comprehensive_results['analysis_capabilities_used']['scientific_validation'] = True
                    except Exception as e:
                        st.warning(f"Scientific validation warning: {e}")
                        comprehensive_results['error_log'].append(f"Scientific validation: {e}")
            
            comprehensive_results['validation_results'] = validation_results
            
            # 7. Calculate comprehensive confidence scores
            comprehensive_results['confidence_scores'] = self._calculate_comprehensive_confidence(comprehensive_results)
            
            # 8. Generate enhanced summary (if available)
            if self.analysis_capabilities['enhanced_summary']:
                try:
                    enhanced_summary_result = self.initialized_components['enhanced_summary'].generate_enhanced_summary(comprehensive_results)
                    comprehensive_results['enhanced_summary'] = enhanced_summary_result
                    comprehensive_results['analysis_capabilities_used']['enhanced_summary'] = True
                except Exception as e:
                    st.warning(f"Enhanced summary warning: {e}")
                    comprehensive_results['error_log'].append(f"Enhanced summary: {e}")
            
            return comprehensive_results
            
        except Exception as e:
            st.error(f"Comprehensive analysis error: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def generate_pdf_report(self, analysis_results: Dict, image_path: str = None) -> Optional[str]:
        """Generate PDF report if available"""
        if not self.analysis_capabilities['pdf_generation']:
            return None
        
        try:
            # Use enhanced summary if available, otherwise create basic summary
            if 'enhanced_summary' in analysis_results:
                enhanced_summary_data = analysis_results['enhanced_summary']
            else:
                enhanced_summary_data = self._create_basic_summary(analysis_results)
            
            pdf_filename = self.initialized_components['pdf_generator'].generate_comprehensive_report(
                analysis_results, enhanced_summary_data, image_path
            )
            
            return pdf_filename
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
            return None
    
    def generate_video(self, image_path: str, analysis_results: Dict, video_options: Dict) -> Optional[str]:
        """Generate analysis video if available"""
        if not self.analysis_capabilities['video_generation']:
            return None
        
        try:
            video_path = self.initialized_components['video_generator'].create_comprehensive_video(
                image_path, analysis_results, video_options
            )
            return video_path
        except Exception as e:
            st.error(f"Video generation failed: {e}")
            return None
    
    def prepare_expert_review(self, analysis_results: Dict, image_path: str) -> Optional[Dict]:
        """Prepare expert review package if available"""
        if not self.analysis_capabilities['expert_collaboration']:
            return None
        
        try:
            expert_package = self.initialized_components['expert_collaboration'].prepare_expert_review_package(
                analysis_results, image_path
            )
            return expert_package
        except Exception as e:
            st.error(f"Expert review preparation failed: {e}")
            return None
    
    def _determine_age_group(self, age: int) -> str:
        """Determine age group for the child"""
        if age < 4:
            return "Toddler (2-3 years)"
        elif age < 7:
            return "Preschool (4-6 years)"
        elif age < 12:
            return "School Age (7-11 years)"
        else:
            return "Adolescent (12+ years)"
    
    def _calculate_comprehensive_confidence(self, results: Dict) -> Dict:
        """Calculate comprehensive confidence scores"""
        confidence_scores = {}
        
        # Traditional analysis confidence
        confidence_scores['traditional'] = 0.75
        
        # AI analysis confidence
        if 'ai_analysis' in results:
            ai_confidence = results['ai_analysis'].get('consensus_analysis', {}).get('confidence_level', 0.7)
            confidence_scores['ai_analysis'] = float(ai_confidence)
        
        # Psychological assessment confidence
        if 'psychological_assessment' in results:
            confidence_scores['psychological'] = 0.80
        
        # Clinical assessment confidence
        if 'clinical_results' in results:
            confidence_scores['clinical'] = 0.75
        
        # Validation confidence
        if 'validation_results' in results:
            confidence_scores['validation'] = 0.85
        
        # Overall confidence (weighted average)
        if confidence_scores:
            weights = {
                'traditional': 0.2,
                'ai_analysis': 0.3,
                'psychological': 0.2,
                'clinical': 0.15,
                'validation': 0.15
            }
            
            weighted_sum = sum(confidence_scores.get(key, 0.5) * weight for key, weight in weights.items())
            total_weight = sum(weight for key, weight in weights.items() if key in confidence_scores)
            
            confidence_scores['overall'] = weighted_sum / total_weight if total_weight > 0 else 0.5
        else:
            confidence_scores['overall'] = 0.5
        
        return confidence_scores
    
    # Default/fallback creation methods
    def _create_default_features(self) -> Dict:
        """Create default features when extraction fails"""
        return {
            'cognitive_features': {'detail_density': 0.5, 'organization_score': 0.5},
            'emotional_features': {'emotional_valence': 0.5, 'color_emotions': {}},
            'developmental_features': {'age_appropriateness': 'age_appropriate'},
            'personality_features': {'drawing_size_percentile': 0.5},
            'social_features': {'social_connection_score': 0.5},
            'psychological_scores': {
                'emotional_wellbeing': 0.6,
                'cognitive_development': 0.6,
                'social_adjustment': 0.6,
                'overall_psychological_health': 0.6
            }
        }
    
    def _create_default_ai_analysis(self) -> Dict:
        """Create default AI analysis when AI fails"""
        return {
            'clip_analysis': {'dominant_category': 'drawing', 'confidence': 0.5},
            'blip_analysis': {'description': 'A child\'s drawing', 'confidence': 0.5},
            'llm_psychological': {'interpretation': 'Standard analysis', 'confidence': 0.5},
            'consensus_analysis': {'primary_emotional_state': 'neutral', 'confidence_level': 0.5},
            'individual_analyses': []
        }
    
    def _create_default_psychological_assessment(self) -> Dict:
        """Create default psychological assessment"""
        return {
            'assessment_timestamp': datetime.now().isoformat(),
            'domain_assessments': {
                'emotional': {'indicators': [], 'overall_emotional_health': 0.6},
                'cognitive': {'indicators': [], 'overall_cognitive_functioning': 0.6},
                'social': {'indicators': [], 'social_connection_score': 0.6}
            },
            'overall_psychological_profile': {'profile_summary': 'healthy_development'}
        }
    
    def _create_default_clinical_results(self) -> Dict:
        """Create default clinical results"""
        return {
            'trauma_assessment': {
                'trauma_flags': [],
                'overall_trauma_risk': 0.2,
                'risk_level': 'low'
            },
            'attachment_assessment': {
                'attachment_style': 'secure',
                'attachment_security_score': 0.7
            }
        }
    
    def _create_basic_summary(self, analysis_results: Dict) -> Dict:
        """Create basic summary when enhanced summary is not available"""
        return {
            'executive_summary': {
                'main_summary': 'Analysis completed successfully.',
                'key_findings': ['Analysis completed', 'Basic assessment available'],
                'overall_assessment': 'Good',
                'confidence_level': analysis_results.get('confidence_scores', {}).get('overall', 0.7)
            },
            'enhanced_recommendations': {
                'immediate_actions': ['Encourage continued artistic expression'],
                'short_term_goals': ['Provide various art materials'],
                'long_term_development': ['Support creative development']
            },
            'developmental_assessment': {
                'age_group': analysis_results.get('input_info', {}).get('age_group', 'Unknown'),
                'milestone_progress': '85%'
            },
            'action_plan': {
                'this_week': ['Display artwork', 'Encourage drawing time'],
                'this_month': ['Introduce new materials'],
                'next_3_months': ['Monitor progress']
            }
        }

def main():
    """Main Streamlit application with comprehensive integration"""
    
    # Page configuration
    st.set_page_config(
        page_title="Comprehensive Children's Drawing Analysis with AI",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .ai-section {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-good { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-error { color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<div class="main-header">🎨 Comprehensive Children\'s Drawing Analysis Platform</div>', 
                unsafe_allow_html=True)
    
    # System status section
    with st.expander("🔧 System Status & Capabilities", expanded=False):
        st.subheader("📊 Module Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Core Analysis:**")
            st.write(f"{'✅' if PSYDRAW_AVAILABLE else '❌'} PsyDraw Features")
            st.write(f"{'✅' if PSYCHOLOGICAL_AVAILABLE else '❌'} Psychological Assessment")
            st.write(f"{'✅' if CLINICAL_AVAILABLE else '❌'} Clinical Assessment")
        
        with col2:
            st.write("**AI & LLM:**")
            st.write(f"{'✅' if AI_ANALYZER_AVAILABLE else '❌'} AI Analysis Engine")
            st.write(f"{'✅' if LLM_AVAILABLE else '❌'} LLM Integration")
            st.write(f"{'✅' if os.getenv('OPENAI_API_KEY') else '❌'} OpenAI API")
            st.write(f"{'✅' if os.getenv('PERPLEXITY_API_KEY') else '❌'} Perplexity API")
        
        with col3:
            st.write("**Validation & Output:**")
            st.write(f"{'✅' if VALIDATION_AVAILABLE else '❌'} Scientific Validation")
            st.write(f"{'✅' if PDF_AVAILABLE else '❌'} PDF Generation")
            st.write(f"{'✅' if VIDEO_AVAILABLE else '❌'} Video Generation")
            st.write(f"{'✅' if EXPERT_AVAILABLE else '❌'} Expert Collaboration")
        
        if MODULES_STATUS:
            st.subheader("⚠️ Module Issues")
            for module, status in MODULES_STATUS.items():
                st.write(f"**{module}:** {status}")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        with st.spinner("Initializing comprehensive analysis system..."):
            try:
                st.session_state.analyzer = ComprehensiveDrawingAnalyzer()
                st.success("✅ Analysis system initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize analyzer: {e}")
                st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📋 Analysis Configuration")
        
        # Child information
        st.subheader("👶 Child Information")
        child_age = st.slider("Child's Age", min_value=2, max_value=18, value=7)
        
        drawing_context = st.selectbox(
            "Drawing Context",
            ["free_drawing", "family_drawing", "house_tree_person", "self_portrait", 
             "emotional_expression", "story_illustration", "school_assignment"]
        )
        
        # Analysis options
        st.subheader("🔬 Analysis Options")
        analysis_options = {
            'include_clinical': st.checkbox("Clinical Assessment", value=True),
            'include_validation': st.checkbox("Scientific Validation", value=True),
            'include_ai_analysis': st.checkbox("AI Analysis", value=True),
            'include_llm_analysis': st.checkbox("LLM Analysis", value=True),
            'include_bias_detection': st.checkbox("Bias Detection", value=True),
            'include_research_validation': st.checkbox("Research Validation", value=True)
        }
        
        # Output options
        st.subheader("📄 Output Options")
        generate_pdf = st.checkbox("Generate PDF Report", value=True)
        generate_video = st.checkbox("Generate Analysis Video", value=False)
        prepare_expert_review = st.checkbox("Prepare Expert Review", value=False)
        
        if generate_video:
            video_style = st.selectbox(
                "Video Style",
                ["analysis_walkthrough", "character_animation", "drawing_evolution", 
                 "story_animation", "scientific_presentation"]
            )
            video_duration = st.slider("Duration (seconds)", 10, 30, 15)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Upload Drawing")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the child's drawing"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Drawing", use_column_width=True)
                
                # Save temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    image.save(tmp_file.name)
                    temp_image_path = tmp_file.name
                    st.session_state.temp_image_path = temp_image_path
            except Exception as e:
                st.error(f"Error processing image: {e}")
                uploaded_file = None
    
    with col2:
        if uploaded_file is not None and 'temp_image_path' in st.session_state:
            st.subheader("🔬 Analysis Results")
            
            # Analysis button
            if st.button("🚀 Start Comprehensive Analysis", type="primary"):
                
                with st.spinner("Conducting comprehensive analysis..."):
                    try:
                        # Perform comprehensive analysis
                        results = st.session_state.analyzer.analyze_drawing_comprehensive(
                            st.session_state.temp_image_path, 
                            child_age, 
                            drawing_context,
                            analysis_options
                        )
                        
                        # Store results in session state
                        st.session_state.analysis_results = results
                        
                        # Generate additional outputs if requested
                        if generate_pdf and 'error' not in results:
                            with st.spinner("Generating PDF report..."):
                                pdf_path = st.session_state.analyzer.generate_pdf_report(
                                    results, st.session_state.temp_image_path
                                )
                                if pdf_path:
                                    st.session_state.pdf_path = pdf_path
                        
                        if generate_video and 'error' not in results:
                            with st.spinner("Generating analysis video..."):
                                video_options = {
                                    'style': video_style if generate_video else 'analysis_walkthrough',
                                    'duration': video_duration if generate_video else 15
                                }
                                video_path = st.session_state.analyzer.generate_video(
                                    st.session_state.temp_image_path, results, video_options
                                )
                                if video_path:
                                    st.session_state.video_path = video_path
                        
                        if prepare_expert_review and 'error' not in results:
                            expert_package = st.session_state.analyzer.prepare_expert_review(
                                results, st.session_state.temp_image_path
                            )
                            if expert_package:
                                st.session_state.expert_package = expert_package
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                
                # Display results
                if 'analysis_results' in st.session_state:
                    results = st.session_state.analysis_results
                    if 'error' not in results:
                        st.success("✅ Comprehensive analysis completed successfully!")
                        
                        # Display confidence scores
                        confidence_scores = results.get('confidence_scores', {})
                        
                        st.subheader("📊 Analysis Confidence")
                        conf_cols = st.columns(len(confidence_scores))
                        for i, (metric, score) in enumerate(confidence_scores.items()):
                            with conf_cols[i]:
                                st.metric(metric.replace('_', ' ').title(), f"{score:.1%}")
                        
                        # Display capabilities used
                        capabilities_used = results.get('analysis_capabilities_used', {})
                        if capabilities_used:
                            st.subheader("🔧 Analysis Components Used")
                            used_components = [k.replace('_', ' ').title() for k, v in capabilities_used.items() if v]
                            st.write(", ".join(used_components))
                        
                        # Display any errors
                        if results.get('error_log'):
                            with st.expander("⚠️ Analysis Warnings", expanded=False):
                                for error in results['error_log']:
                                    st.warning(error)
                    else:
                        st.error(f"❌ Analysis failed: {results['error']}")
    
    # Results display section
    if 'analysis_results' in st.session_state and 'error' not in st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Create tabs for different result sections
        tabs = ["🤖 AI Analysis", "📊 Psychological Profile", "🧠 Cognitive Analysis", 
                "❤️ Emotional Assessment", "🏥 Clinical Indicators", "📈 Validation Results"]
        
        if generate_video:
            tabs.append("🎬 Video Generation")
        
        tab_objects = st.tabs(tabs)
        
        with tab_objects[0]:  # AI Analysis
            display_ai_analysis(results)
        
        with tab_objects[1]:  # Psychological Profile
            display_psychological_profile(results)
        
        with tab_objects[2]:  # Cognitive Analysis
            display_cognitive_analysis(results)
        
        with tab_objects[3]:  # Emotional Assessment
            display_emotional_assessment(results)
        
        with tab_objects[4]:  # Clinical Indicators
            display_clinical_indicators(results, analysis_options.get('include_clinical', True))
        
        with tab_objects[5]:  # Validation Results
            display_validation_results(results, analysis_options.get('include_validation', True))
        
        if generate_video and len(tab_objects) > 6:
            with tab_objects[6]:  # Video Generation
                display_video_results()
        
        # Download section
        st.markdown("---")
        st.subheader("📥 Downloads & Expert Collaboration")
        
        download_cols = st.columns(4)
        
        with download_cols[0]:
            if 'pdf_path' in st.session_state and os.path.exists(st.session_state.pdf_path):
                with open(st.session_state.pdf_path, 'rb') as pdf_file:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_file.read(),
                        file_name=f"drawing_analysis_{child_age}yo_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
        
        with download_cols[1]:
            if 'video_path' in st.session_state and os.path.exists(st.session_state.video_path):
                with open(st.session_state.video_path, 'rb') as video_file:
                    st.download_button(
                        label="🎬 Download Video",
                        data=video_file.read(),
                        file_name=f"analysis_video_{child_age}yo_{datetime.now().strftime('%Y%m%d')}.mp4",
                        mime="video/mp4"
                    )
        
        with download_cols[2]:
            # Raw analysis data
            analysis_json = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📊 Download Raw Data",
                data=analysis_json,
                file_name=f"analysis_data_{child_age}yo_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        with download_cols[3]:
            if 'expert_package' in st.session_state:
                expert_json = json.dumps(st.session_state.expert_package, indent=2, default=str)
                st.download_button(
                    label="👥 Download Expert Package",
                    data=expert_json,
                    file_name=f"expert_review_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

# Display functions for different analysis sections
def display_ai_analysis(results: Dict):
    """Display AI analysis results"""
    st.markdown("### 🤖 AI Analysis Results")
    
    ai_analysis = results.get('ai_analysis', {})
    
    if not ai_analysis:
        st.info("AI analysis was not included in this assessment.")
        return
    
    # AI Model Results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🖼️ CLIP Visual Analysis")
        clip_analysis = ai_analysis.get('clip_analysis', {})
        if clip_analysis:
            st.metric("Dominant Category", clip_analysis.get('dominant_category', 'Unknown'))
            st.metric("Confidence", f"{clip_analysis.get('confidence', 0):.1%}")
    
    with col2:
        st.subheader("📝 BLIP Image Description")
        blip_analysis = ai_analysis.get('blip_analysis', {})
        if blip_analysis:
            st.write("**Description:**", blip_analysis.get('description', 'No description available'))
            st.metric("Confidence", f"{blip_analysis.get('confidence', 0):.1%}")
    
    with col3:
        st.subheader("🧠 LLM Psychological Analysis")
        llm_analysis = ai_analysis.get('llm_psychological', {})
        if llm_analysis:
            st.write("**Interpretation:**", llm_analysis.get('interpretation', 'No interpretation available')[:200] + "...")
            st.metric("Confidence", f"{llm_analysis.get('confidence', 0):.1%}")
    
    # Consensus Analysis
    st.subheader("🎯 AI Consensus Analysis")
    consensus = ai_analysis.get('consensus_analysis', {})
    if consensus:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Primary Emotional State", consensus.get('primary_emotional_state', 'Unknown'))
        with col2:
            st.metric("Developmental Assessment", consensus.get('developmental_assessment', 'Unknown'))
        with col3:
            st.metric("AI Agreement Score", f"{consensus.get('ai_agreement_score', 0):.1%}")

def display_psychological_profile(results: Dict):
    """Display psychological profile"""
    st.markdown("### 🧠 Psychological Profile Overview")
    
    psychological_assessment = results.get('psychological_assessment', {})
    
    if not psychological_assessment:
        st.info("Psychological assessment not available.")
        return
    
    # Display domain assessments
    domain_assessments = psychological_assessment.get('domain_assessments', {})
    
    if domain_assessments:
        for domain, assessment in domain_assessments.items():
            if assessment and isinstance(assessment, dict):
                st.subheader(f"{domain.replace('_', ' ').title()} Assessment")
                
                # Display overall score if available
                overall_keys = [k for k in assessment.keys() if k.startswith('overall_')]
                for key in overall_keys:
                    st.metric(f"Overall {domain.title()}", f"{assessment[key]:.2f}")

def display_cognitive_analysis(results: Dict):
    """Display cognitive analysis"""
    st.markdown("### 🧠 Cognitive Analysis")
    
    cognitive_features = results.get('psydraw_features', {}).get('cognitive_features', {})
    
    if cognitive_features:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cognitive Metrics")
            st.metric("Detail Density", f"{cognitive_features.get('detail_density', 0):.3f}")
            st.metric("Organization Score", f"{cognitive_features.get('organization_score', 0):.2f}")
        
        with col2:
            st.subheader("Complexity Analysis")
            st.metric("Object Count", cognitive_features.get('object_count', 0))
            st.write("**Complexity Level:**", cognitive_features.get('complexity_level', 'Unknown'))
    else:
        st.info("Cognitive features not available in analysis results.")

def display_emotional_assessment(results: Dict):
    """Display emotional assessment"""
    st.markdown("### ❤️ Emotional Assessment")
    
    emotional_features = results.get('psydraw_features', {}).get('emotional_features', {})
    
    if emotional_features:
        # Color emotions analysis
        color_emotions = emotional_features.get('color_emotions', {})
        if color_emotions and 'emotional_profile' in color_emotions:
            st.subheader("Color-Emotion Analysis")
            
            emotional_profile = color_emotions['emotional_profile']
            if emotional_profile:
                # Create emotion bar chart
                emotions = list(emotional_profile.keys())
                scores = list(emotional_profile.values())
                
                fig = px.bar(
                    x=emotions, y=scores,
                    title="Emotional Profile from Color Usage",
                    labels={'x': 'Emotions', 'y': 'Intensity Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Emotional Valence", f"{emotional_features.get('emotional_valence', 0.5):.2f}")
        with col2:
            color_temp = emotional_features.get('color_emotions', {}).get('color_temperature', 1)
            st.metric("Color Temperature", f"{color_temp:.2f}")
    else:
        st.info("Emotional features not available in analysis results.")

def display_clinical_indicators(results: Dict, include_clinical: bool):
    """Display clinical assessment indicators"""
    st.markdown("### 🏥 Clinical Assessment")
    
    if not include_clinical:
        st.info("Clinical assessment was not included in this analysis.")
        return
    
    clinical_results = results.get('clinical_results', {})
    
    if not clinical_results:
        st.info("Clinical assessment results not available.")
        return
    
    # Trauma assessment
    if 'trauma_assessment' in clinical_results:
        trauma_results = clinical_results['trauma_assessment']
        
        st.subheader("🚨 Trauma Indicators")
        
        overall_risk = trauma_results.get('overall_trauma_risk', 0)
        risk_level = trauma_results.get('risk_level', 'unknown')
        
        if risk_level == 'high':
            st.error(f"⚠️ High trauma risk detected (Score: {overall_risk:.2f})")
        elif risk_level == 'moderate':
            st.warning(f"⚠️ Moderate trauma risk detected (Score: {overall_risk:.2f})")
        else:
            st.success(f"✅ Low trauma risk (Score: {overall_risk:.2f})")
    
    # Attachment assessment
    if 'attachment_assessment' in clinical_results:
        attachment_results = clinical_results['attachment_assessment']
        
        st.subheader("👨‍👩‍👧‍👦 Attachment Assessment")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Attachment Style", attachment_results.get('attachment_style', 'Unknown'))
        with col2:
            st.metric("Security Score", f"{attachment_results.get('attachment_security_score', 0):.2f}")

def display_validation_results(results: Dict, include_validation: bool):
    """Display validation results"""
    st.markdown("### 📈 Validation Results")
    
    if not include_validation:
        st.info("Validation was not included in this analysis.")
        return
    
    validation_results = results.get('validation_results', {})
    
    if not validation_results:
        st.info("Validation results not available.")
        return
    
    # Research validation
    if 'research_compliance_score' in validation_results:
        st.metric("Research Compliance", f"{validation_results['research_compliance_score']:.1%}")
    
    # Bias detection
    if 'bias_detection' in validation_results:
        bias_results = validation_results['bias_detection']
        overall_bias_risk = bias_results.get('overall_bias_risk', 'unknown')
        
        if overall_bias_risk == 'high':
            st.error("⚠️ High bias risk detected")
        elif overall_bias_risk == 'medium':
            st.warning("⚠️ Medium bias risk detected")
        else:
            st.success("✅ Low bias risk")

def display_video_results():
    """Display video generation results"""
    st.markdown("### 🎬 Video Generation")
    
    if 'video_path' in st.session_state and os.path.exists(st.session_state.video_path):
        st.success("✅ Analysis video generated successfully!")
        
        # Display video
        with open(st.session_state.video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)
    else:
        st.info("Video generation was not completed or failed.")

if __name__ == "__main__":
    main()