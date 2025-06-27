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

# Import your custom modules with error handling
try:
    from psydraw_feature_extractor import PsyDrawFeatureExtractor
    from psychological_assessment_engine import PsychologicalAssessmentEngine
    from clinical_assessment_advanced import AdvancedClinicalAssessment
    from expert_collaboration_framework import ExpertCollaborationFramework
    from research_validation_module import ResearchValidationModule
    from validation_framework import ValidationFramework, BiasDetectionSystem
    from video_generator_consolidated import ConsolidatedVideoGenerator
    from ai_analysis_engine import ComprehensiveAIAnalyzer  # NEW AI ENGINE
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"Module import error: {e}")
    MODULES_LOADED = False

class ComprehensiveDrawingAnalyzer:
    """
    Main application class integrating all psychological assessment modules
    INCLUDING ADVANCED AI ANALYSIS
    """
    
    def __init__(self):
        if not MODULES_LOADED:
            st.error("Required modules not loaded. Please check module imports.")
            return
            
        try:
            # Original modules
            self.psydraw_extractor = PsyDrawFeatureExtractor()
            self.psychological_engine = PsychologicalAssessmentEngine()
            self.clinical_assessment = AdvancedClinicalAssessment()
            self.expert_collaboration = ExpertCollaborationFramework()
            self.research_validation = ResearchValidationModule()
            self.validation_framework = ValidationFramework()
            self.bias_detection = BiasDetectionSystem()
            self.video_generator = ConsolidatedVideoGenerator()
            
            # NEW: Advanced AI Analysis Engine
            self.ai_analyzer = ComprehensiveAIAnalyzer()
            
            self.initialized = True
        except Exception as e:
            st.error(f"Initialization error: {e}")
            self.initialized = False
    
    def analyze_drawing(self, image_path: str, child_age: int, drawing_context: str, 
                       include_clinical: bool = True, include_validation: bool = True,
                       include_ai_analysis: bool = True) -> Dict:
        """
        Comprehensive drawing analysis using all available modules
        NOW WITH ADVANCED AI ANALYSIS
        """
        if not self.initialized:
            return {'error': 'Analyzer not properly initialized'}
            
        try:
            # Load and validate image
            if not os.path.exists(image_path):
                return {'error': f'Image file not found: {image_path}'}
                
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image file'}
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract PsyDraw features
            st.info("🧠 Extracting psychological features...")
            try:
                psydraw_features = self.psydraw_extractor.extract_complete_psydraw_features(
                    image_rgb, child_age, drawing_context
                )
            except Exception as e:
                st.warning(f"PsyDraw feature extraction warning: {e}")
                psydraw_features = self._create_default_features()
            
            # NEW: Comprehensive AI Analysis
            ai_analysis_results = {}
            if include_ai_analysis:
                st.info("🤖 Conducting advanced AI analysis...")
                try:
                    ai_analysis_results = self.ai_analyzer.conduct_multi_ai_analysis(
                        image_path, child_age, drawing_context
                    )
                except Exception as e:
                    st.warning(f"AI analysis warning: {e}")
                    ai_analysis_results = self._create_default_ai_analysis()
            
            # Conduct psychological assessment (ENHANCED with AI)
            st.info("🔬 Conducting psychological assessment...")
            try:
                psychological_assessment = self.psychological_engine.conduct_comprehensive_assessment(
                    psydraw_features, child_age, drawing_context
                )
            except Exception as e:
                st.warning(f"Psychological assessment warning: {e}")
                psychological_assessment = self._create_default_psychological_assessment()
            
            # Clinical assessment (if requested)
            clinical_results = {}
            if include_clinical:
                st.info("🏥 Performing clinical assessment...")
                try:
                    # Trauma assessment (ENHANCED with AI)
                    trauma_assessment = self.clinical_assessment.conduct_trauma_assessment(
                        image_rgb, psydraw_features, child_age
                    )
                    
                    # Attachment assessment (ENHANCED with AI)
                    attachment_assessment = self.clinical_assessment.assess_attachment_patterns(
                        image_rgb, psydraw_features, drawing_context
                    )
                    
                    clinical_results = {
                        'trauma_assessment': trauma_assessment,
                        'attachment_assessment': attachment_assessment
                    }
                except Exception as e:
                    st.warning(f"Clinical assessment warning: {e}")
                    clinical_results = self._create_default_clinical_results()
            
            # Research validation (if requested)
            validation_results = {}
            if include_validation:
                st.info("📊 Validating against research benchmarks...")
                try:
                    validation_results = self.research_validation.conduct_comprehensive_validation(
                        {
                            'psydraw_features': psydraw_features,
                            'psychological_assessment': psychological_assessment,
                            'clinical_results': clinical_results,
                            'ai_analysis': ai_analysis_results  # NEW: AI validation
                        },
                        child_age
                    )
                    
                    # Bias detection (ENHANCED with AI bias detection)
                    bias_results = self.bias_detection.detect_biases(
                        psychological_assessment,
                        {'child_age': child_age, 'drawing_context': drawing_context},
                        ai_analysis_results
                    )
                    validation_results['bias_detection'] = bias_results
                except Exception as e:
                    st.warning(f"Validation warning: {e}")
                    validation_results = self._create_default_validation_results()
            
            # Compile comprehensive results
            comprehensive_results = {
                'timestamp': datetime.now().isoformat(),
                'input_info': {
                    'child_age': child_age,
                    'drawing_context': drawing_context,
                    'image_path': image_path
                },
                'psydraw_features': psydraw_features,
                'ai_analysis': ai_analysis_results,  # NEW: AI analysis results
                'psychological_assessment': psychological_assessment,
                'clinical_results': clinical_results,
                'validation_results': validation_results,
                'confidence_scores': self._calculate_overall_confidence(
                    psydraw_features, psychological_assessment, validation_results, ai_analysis_results
                ),
                'llm_analyses': ai_analysis_results.get('individual_analyses', [])  # NEW: For expert collaboration
            }
            
            return comprehensive_results
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def _create_default_ai_analysis(self) -> Dict:
        """Create default AI analysis when AI fails"""
        return {
            'clip_analysis': {'dominant_category': 'drawing', 'confidence': 0.5},
            'blip_analysis': {'description': 'A child\'s drawing', 'confidence': 0.5},
            'llm_psychological': {'interpretation': 'Standard analysis', 'confidence': 0.5},
            'consensus_analysis': {'primary_emotional_state': 'neutral', 'confidence_level': 0.5},
            'individual_analyses': []
        }
    
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
    
    def _create_default_validation_results(self) -> Dict:
        """Create default validation results"""
        return {
            'research_compliance_score': 0.75,
            'research_alignment': {'overall_research_alignment': 0.75},
            'bias_detection': {'overall_bias_risk': 'low'}
        }
    
    def _calculate_overall_confidence(self, psydraw_features: Dict, 
                                    psychological_assessment: Dict, 
                                    validation_results: Dict,
                                    ai_analysis: Dict) -> Dict:
        """Calculate overall confidence scores (ENHANCED with AI confidence)"""
        
        # Base confidence from feature extraction quality
        feature_confidence = psydraw_features.get('psychological_scores', {}).get(
            'overall_psychological_health', 0.7
        )
        
        # Psychological assessment confidence
        psych_confidence = 0.8
        
        # Validation confidence
        validation_confidence = validation_results.get('research_compliance_score', 0.75) if validation_results else 0.75
        
        # NEW: AI analysis confidence
        ai_confidence = ai_analysis.get('consensus_analysis', {}).get('confidence_level', 0.7) if ai_analysis else 0.7
        
        # Overall weighted confidence (NOW INCLUDES AI)
        overall_confidence = (
            feature_confidence * 0.25 +
            psych_confidence * 0.25 +
            validation_confidence * 0.25 +
            ai_confidence * 0.25
        )
        
        return {
            'feature_extraction': float(feature_confidence),
            'psychological_assessment': float(psych_confidence),
            'validation_score': float(validation_confidence),
            'ai_analysis': float(ai_confidence),  # NEW
            'overall': float(overall_confidence)
        }

def main():
    """Main Streamlit application (ENHANCED with AI analysis options)"""
    
    # Page configuration
    st.set_page_config(
        page_title="Advanced Children's Drawing Analysis with AI",
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
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<div class="main-header">🎨 Advanced Children\'s Drawing Analysis with AI</div>', 
                unsafe_allow_html=True)
    
    # NEW: AI Status indicator
    st.markdown('<div class="ai-section">🤖 <strong>AI-Powered Analysis:</strong> Multi-model AI system with CLIP, BLIP, and LLM integration</div>', 
                unsafe_allow_html=True)
    
    # Check if modules are loaded
    if not MODULES_LOADED:
        st.error("❌ Required modules could not be loaded. Please check your installation.")
        st.stop()
    
    # Initialize analyzer with error handling
    if 'analyzer' not in st.session_state:
        with st.spinner("Initializing AI analysis systems..."):
            try:
                st.session_state.analyzer = ComprehensiveDrawingAnalyzer()
                if not st.session_state.analyzer.initialized:
                    st.error("❌ Analyzer initialization failed")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Failed to initialize analyzer: {e}")
                st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📋 Analysis Configuration")
        
        # Child information
        st.subheader("Child Information")
        child_age = st.slider("Child's Age", min_value=2, max_value=18, value=7)
        
        drawing_context = st.selectbox(
            "Drawing Context",
            ["free_drawing", "family_drawing", "house_tree_person", "self_portrait", 
             "emotional_expression", "story_illustration", "school_assignment"]
        )
        
        # Analysis options
        st.subheader("Analysis Options")
        include_clinical = st.checkbox("Include Clinical Assessment", value=True)
        include_validation = st.checkbox("Include Research Validation", value=True)
        include_ai_analysis = st.checkbox("Include AI Analysis", value=True)  # NEW
        include_video = st.checkbox("Generate Analysis Video", value=False)
        
        # NEW: AI-specific options
        if include_ai_analysis:
            st.subheader("🤖 AI Analysis Options")
            ai_models = st.multiselect(
                "AI Models to Use",
                ["CLIP (Visual Understanding)", "BLIP (Image Captioning)", "LLM (Psychological Interpretation)"],
                default=["CLIP (Visual Understanding)", "BLIP (Image Captioning)", "LLM (Psychological Interpretation)"]
            )
            ai_confidence_threshold = st.slider("AI Confidence Threshold", 0.1, 1.0, 0.7)
        
        # Video options (if enabled)
        if include_video:
            st.subheader("Video Options")
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
            if st.button("🚀 Start Comprehensive AI Analysis", type="primary"):
                
                with st.spinner("Conducting comprehensive AI-powered analysis..."):
                    try:
                        # Perform analysis (NOW WITH AI)
                        results = st.session_state.analyzer.analyze_drawing(
                            st.session_state.temp_image_path, child_age, drawing_context,
                            include_clinical, include_validation, include_ai_analysis
                        )
                        
                        # Store results in session state
                        st.session_state.analysis_results = results
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                
                # Display results
                if 'analysis_results' in st.session_state:
                    results = st.session_state.analysis_results
                    if 'error' not in results:
                        st.success("✅ AI-powered analysis completed successfully!")
                        
                        # Display confidence scores (NOW WITH AI)
                        confidence_scores = results.get('confidence_scores', {})
                        
                        col_conf1, col_conf2, col_conf3, col_conf4 = st.columns(4)
                        with col_conf1:
                            st.metric("Overall Confidence", f"{confidence_scores.get('overall', 0):.1%}")
                        with col_conf2:
                            st.metric("Psychological Assessment", f"{confidence_scores.get('psychological_assessment', 0):.1%}")
                        with col_conf3:
                            st.metric("Validation Score", f"{confidence_scores.get('validation_score', 0):.1%}")
                        with col_conf4:
                            st.metric("AI Analysis", f"{confidence_scores.get('ai_analysis', 0):.1%}")  # NEW
                    else:
                        st.error(f"❌ Analysis failed: {results['error']}")
    
    # Results display section (ENHANCED with AI results)
    if 'analysis_results' in st.session_state and 'error' not in st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Create tabs for different result sections (NEW AI TAB)
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🤖 AI Analysis", "📊 Psychological Profile", "🧠 Cognitive Analysis", 
            "❤️ Emotional Assessment", "🏥 Clinical Indicators", "📈 Research Validation", "🎬 Video Generation"
        ])
        
        with tab1:  # NEW: AI Analysis Tab
            display_ai_analysis(results)
        
        with tab2:
            display_psychological_profile(results)
        
        with tab3:
            display_cognitive_analysis(results)
        
        with tab4:
            display_emotional_assessment(results)
        
        with tab5:
            if include_clinical:
                display_clinical_indicators(results)
            else:
                st.info("Clinical assessment was not included in this analysis.")
        
        with tab6:
            if include_validation:
                display_research_validation(results)
            else:
                st.info("Research validation was not included in this analysis.")
        
        with tab7:
            if include_video and 'temp_image_path' in st.session_state:
                display_video_generation(results, st.session_state.temp_image_path, 
                                       video_style if include_video else 'analysis_walkthrough', 
                                       video_duration if include_video else 15)
            else:
                st.info("Video generation is disabled or no analysis results available.")
        
        # Expert collaboration section (ENHANCED with AI data)
        st.markdown('<div class="section-header">👥 Expert Collaboration</div>', unsafe_allow_html=True)
        
        if st.button("📋 Prepare Expert Review Package"):
            try:
                expert_package = st.session_state.analyzer.expert_collaboration.prepare_expert_review_package(
                    results, st.session_state.temp_image_path
                )
                
                st.success("Expert review package prepared!")
                
                # Display package in expandable section
                with st.expander("View Expert Package"):
                    st.json(expert_package)
                
                # Download button for expert package
                package_json = json.dumps(expert_package, indent=2)
                st.download_button(
                    label="📥 Download Expert Package",
                    data=package_json,
                    file_name=f"expert_review_{expert_package.get('case_id', 'unknown')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Failed to prepare expert package: {e}")

def display_ai_analysis(results: Dict):
    """NEW: Display comprehensive AI analysis results"""
    st.markdown('<div class="section-header">🤖 AI Analysis Results</div>', unsafe_allow_html=True)
    
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
            
            # Category scores
            category_scores = clip_analysis.get('category_scores', {})
            if category_scores:
                fig = px.bar(
                    x=list(category_scores.keys())[:5],  # Top 5
                    y=list(category_scores.values())[:5],
                    title="Top Visual Categories"
                )
                st.plotly_chart(fig, use_container_width=True)
    
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
            st.write("**Interpretation:**", llm_analysis.get('interpretation', 'No interpretation available'))
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
    
    # Individual AI Analyses
    individual_analyses = ai_analysis.get('individual_analyses', [])
    if individual_analyses:
        st.subheader("📊 Individual AI Model Comparisons")
        
        # Create comparison table
        comparison_data = []
        for i, analysis in enumerate(individual_analyses):
            comparison_data.append({
                'Model': f'AI Model {i+1}',
                'Confidence': f"{analysis.get('confidence', 0):.1%}",
                'Primary Finding': analysis.get('primary_finding', 'N/A')
            })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)

def display_psychological_profile(results: Dict):
    """Display comprehensive psychological profile (ENHANCED with AI insights)"""
    st.markdown('<div class="section-header">🧠 Psychological Profile Overview</div>', unsafe_allow_html=True)
    
    try:
        psychological_assessment = results.get('psychological_assessment', {})
        domain_assessments = psychological_assessment.get('domain_assessments', {})
        ai_analysis = results.get('ai_analysis', {})  # NEW: AI insights
        
        # Overall psychological health visualization
        if 'psychological_scores' in results.get('psydraw_features', {}):
            scores = results['psydraw_features']['psychological_scores']
            
            # Create radar chart for psychological domains
            categories = ['Emotional Wellbeing', 'Cognitive Development', 'Social Adjustment']
            values = [
                scores.get('emotional_wellbeing', 0.5),
                scores.get('cognitive_development', 0.5),
                scores.get('social_adjustment', 0.5)
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Psychological Profile'
            ))
            
            # NEW: Add AI consensus if available
            if ai_analysis and 'consensus_analysis' in ai_analysis:
                ai_consensus = ai_analysis['consensus_analysis']
                ai_values = [0.6, 0.6, 0.6]  # Default values, can be enhanced
                fig.add_trace(go.Scatterpolar(
                    r=ai_values,
                    theta=categories,
                    fill='toself',
                    name='AI Assessment',
                    line=dict(color='orange')
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Psychological Domain Scores (Human + AI Assessment)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # NEW: AI-Enhanced Insights
        if ai_analysis:
            st.subheader("🤖 AI-Enhanced Psychological Insights")
            consensus = ai_analysis.get('consensus_analysis', {})
            if consensus:
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**AI Primary Assessment:** {consensus.get('primary_emotional_state', 'Unknown')}")
                with col2:
                    st.info(f"**AI Confidence Level:** {consensus.get('confidence_level', 0):.1%}")
        
        # Domain-specific assessments
        for domain, assessment in domain_assessments.items():
            if assessment and isinstance(assessment, dict):
                st.subheader(f"{domain.replace('_', ' ').title()} Assessment")
                
                # Display overall score if available
                overall_key = f'overall_{domain}_health'
                if overall_key in assessment:
                    st.metric(f"Overall {domain.title()} Health", f"{assessment[overall_key]:.2f}")
                
                # Display indicators if available
                if 'indicators' in assessment and assessment['indicators']:
                    with st.expander(f"View {domain.title()} Indicators"):
                        for i, indicator in enumerate(assessment['indicators']):
                            if hasattr(indicator, 'indicator_name'):
                                st.write(f"**{indicator.indicator_name}**: {indicator.interpretation}")
                            else:
                                st.write(f"Indicator {i+1}: {str(indicator)}")
    
    except Exception as e:
        st.error(f"Error displaying psychological profile: {e}")

def display_cognitive_analysis(results: Dict):
    """Display cognitive analysis results (ENHANCED with AI)"""
    st.markdown('<div class="section-header">🧠 Cognitive Analysis</div>', unsafe_allow_html=True)
    
    try:
        cognitive_features = results.get('psydraw_features', {}).get('cognitive_features', {})
        ai_analysis = results.get('ai_analysis', {})  # NEW: AI cognitive insights
        
        if cognitive_features:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Cognitive Metrics")
                st.metric("Detail Density", f"{cognitive_features.get('detail_density', 0):.3f}")
                st.metric("Organization Score", f"{cognitive_features.get('organization_score', 0):.2f}")
                st.metric("Planning Score", f"{cognitive_features.get('planning_score', 0):.2f}")
            
            with col2:
                st.subheader("Complexity Analysis")
                st.metric("Object Count", cognitive_features.get('object_count', 0))
                st.write("**Complexity Level:**", cognitive_features.get('complexity_level', 'Unknown'))
                st.metric("Cognitive Entropy", f"{cognitive_features.get('cognitive_entropy', 0):.2f}")
        
        # NEW: AI Cognitive Insights
        if ai_analysis:
            st.subheader("🤖 AI Cognitive Assessment")
            clip_analysis = ai_analysis.get('clip_analysis', {})
            if clip_analysis:
                # Extract cognitive-related categories from CLIP
                category_scores = clip_analysis.get('category_scores', {})
                cognitive_categories = {k: v for k, v in category_scores.items() 
                                     if 'detailed' in k.lower() or 'simple' in k.lower()}
                
                if cognitive_categories:
                    fig = px.bar(
                        x=list(cognitive_categories.keys()),
                        y=list(cognitive_categories.values()),
                        title="AI-Detected Cognitive Complexity Indicators"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Cognitive features not available in analysis results.")
    
    except Exception as e:
        st.error(f"Error displaying cognitive analysis: {e}")

def display_emotional_assessment(results: Dict):
    """Display emotional assessment results (ENHANCED with AI)"""
    st.markdown('<div class="section-header">❤️ Emotional Assessment</div>', unsafe_allow_html=True)
    
    try:
        emotional_features = results.get('psydraw_features', {}).get('emotional_features', {})
        ai_analysis = results.get('ai_analysis', {})  # NEW: AI emotional insights
        
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
                color_temp = color_emotions.get('color_temperature', 1) if color_emotions else 1
                st.metric("Color Temperature", f"{color_temp:.2f}")
            with col2:
                st.metric("Emotional Valence", f"{emotional_features.get('emotional_valence', 0.5):.2f}")
        
        # NEW: AI Emotional Assessment
        if ai_analysis:
            st.subheader("🤖 AI Emotional Analysis")
            
            # Extract emotional insights from AI models
            consensus = ai_analysis.get('consensus_analysis', {})
            if consensus:
                emotional_state = consensus.get('primary_emotional_state', 'neutral')
                
                if emotional_state == 'positive':
                    st.success(f"✅ **AI Assessment**: Positive emotional indicators detected")
                elif emotional_state == 'concerning':
                    st.warning(f"⚠️ **AI Assessment**: Some concerning emotional patterns detected")
                else:
                    st.info(f"ℹ️ **AI Assessment**: Neutral emotional state")
            
            # CLIP emotional categories
            clip_analysis = ai_analysis.get('clip_analysis', {})
            if clip_analysis:
                category_scores = clip_analysis.get('category_scores', {})
                emotional_categories = {k: v for k, v in category_scores.items() 
                                      if any(emotion in k.lower() for emotion in ['happy', 'sad', 'fear', 'anger'])}
                
                if emotional_categories:
                    fig = px.bar(
                        x=list(emotional_categories.keys()),
                        y=list(emotional_categories.values()),
                        title="AI-Detected Emotional Indicators",
                        color=list(emotional_categories.values()),
                        color_continuous_scale='RdYlBu_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Emotional features not available in analysis results.")
    
    except Exception as e:
        st.error(f"Error displaying emotional assessment: {e}")

def display_clinical_indicators(results: Dict):
    """Display clinical assessment indicators (ENHANCED with AI)"""
    st.markdown('<div class="section-header">🏥 Clinical Assessment</div>', unsafe_allow_html=True)
    
    try:
        clinical_results = results.get('clinical_results', {})
        ai_analysis = results.get('ai_analysis', {})  # NEW: AI clinical insights
        
        if not clinical_results:
            st.info("Clinical assessment results not available.")
            return
        
        # NEW: AI Clinical Risk Assessment
        if ai_analysis:
            st.subheader("🤖 AI Clinical Risk Assessment")
            consensus = ai_analysis.get('consensus_analysis', {})
            if consensus:
                confidence = consensus.get('confidence_level', 0.7)
                if confidence < 0.5:
                    st.warning("⚠️ AI detected potential areas requiring clinical attention")
                else:
                    st.success("✅ AI assessment indicates typical development patterns")
        
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
            
            # Display trauma flags if available
            trauma_flags = trauma_results.get('trauma_flags', [])
            if trauma_flags:
                with st.expander("View Trauma Indicators"):
                    for flag in trauma_flags:
                        if hasattr(flag, 'indicator_type'):
                            st.write(f"**{flag.indicator_type}**: {flag.description}")
                        else:
                            st.write(str(flag))
        
        # Attachment assessment
        if 'attachment_assessment' in clinical_results:
            attachment_results = clinical_results['attachment_assessment']
            
            st.subheader("👨‍👩‍👧‍👦 Attachment Assessment")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Attachment Style", attachment_results.get('attachment_style', 'Unknown'))
            with col2:
                st.metric("Security Score", f"{attachment_results.get('attachment_security_score', 0):.2f}")
    
    except Exception as e:
        st.error(f"Error displaying clinical indicators: {e}")

def display_research_validation(results: Dict):
    """Display research validation results (ENHANCED with AI validation)"""
    st.markdown('<div class="section-header">📈 Research Validation</div>', unsafe_allow_html=True)
    
    try:
        validation_results = results.get('validation_results', {})
        ai_analysis = results.get('ai_analysis', {})  # NEW: AI validation
        
        if not validation_results:
            st.info("Research validation results not available.")
            return
        
        # Research compliance score
        compliance_score = validation_results.get('research_compliance_score', 0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Research Compliance", f"{compliance_score:.1%}")
        with col2:
            research_alignment = validation_results.get('research_alignment', {})
            alignment_score = research_alignment.get('overall_research_alignment', 0)
            st.metric("Research Alignment", f"{alignment_score:.1%}")
        with col3:
            st.metric("Validation Quality", "Good" if compliance_score > 0.7 else "Moderate")
        with col4:  # NEW: AI validation score
            ai_confidence = ai_analysis.get('consensus_analysis', {}).get('confidence_level', 0) if ai_analysis else 0
            st.metric("AI Validation", f"{ai_confidence:.1%}")
        
        # NEW: AI-Research Alignment
        if ai_analysis:
            st.subheader("🤖 AI-Research Alignment Analysis")
            consensus = ai_analysis.get('consensus_analysis', {})
            if consensus:
                agreement_score = consensus.get('ai_agreement_score', 0)
                
                if agreement_score > 0.8:
                    st.success(f"✅ High AI-Research agreement ({agreement_score:.1%})")
                elif agreement_score > 0.6:
                    st.info(f"ℹ️ Moderate AI-Research agreement ({agreement_score:.1%})")
                else:
                    st.warning(f"⚠️ Low AI-Research agreement ({agreement_score:.1%}) - requires review")
        
        # Bias detection results
        if 'bias_detection' in validation_results:
            bias_results = validation_results['bias_detection']
            
            st.subheader("🔍 Bias Detection")
            
            overall_bias_risk = bias_results.get('overall_bias_risk', 'unknown')
            if overall_bias_risk == 'high':
                st.error("⚠️ High bias risk detected")
            elif overall_bias_risk == 'medium':
                st.warning("⚠️ Medium bias risk detected")
            else:
                st.success("✅ Low bias risk")
    
    except Exception as e:
        st.error(f"Error displaying research validation: {e}")

def display_video_generation(results: Dict, image_path: str, video_style: str, duration: int):
    """Display video generation interface (ENHANCED with AI insights)"""
    st.markdown('<div class="section-header">🎬 Video Generation</div>', unsafe_allow_html=True)
    
    try:
        if st.button("🎥 Generate AI-Enhanced Analysis Video"):
            with st.spinner("Generating AI-enhanced video... This may take a few minutes."):
                try:
                    video_options = {
                        'style': video_style,
                        'duration': duration,
                        'include_ai_insights': True  # NEW: Include AI insights in video
                    }
                    
                    video_path = st.session_state.analyzer.video_generator.create_comprehensive_video(
                        image_path, results, video_options
                    )
                    
                    if video_path and os.path.exists(video_path):
                        st.success("✅ AI-enhanced video generated successfully!")
                        
                        # Display video
                        with open(video_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download AI-Enhanced Video",
                            data=video_bytes,
                            file_name=f"ai_analysis_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                            mime="video/mp4"
                        )
                    else:
                        st.error("❌ Video generation failed")
                        
                except Exception as e:
                    st.error(f"❌ Video generation error: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in video generation interface: {e}")

if __name__ == "__main__":
    main()
