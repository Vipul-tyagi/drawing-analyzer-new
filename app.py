import streamlit as st
import os
import sys
import tempfile
import base64
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import json

# Configure page
st.set_page_config(
    page_title="Children's Drawing Analysis",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .analysis-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Import analysis components with error handling
@st.cache_resource
def load_analysis_components():
    """Load all analysis components with proper error handling"""
    components = {}
    
    try:
        from enhanced_drawing_analyzer import EnhancedDrawingAnalyzer, ScientificallyValidatedAnalyzer
        components['enhanced_analyzer'] = EnhancedDrawingAnalyzer()
        components['validated_analyzer'] = ScientificallyValidatedAnalyzer()
        st.success("✅ Enhanced Drawing Analyzer loaded successfully!")
    except ImportError as e:
        st.warning(f"⚠️ Enhanced Drawing Analyzer not available: {e}")
        components['enhanced_analyzer'] = None
        components['validated_analyzer'] = None
    
    try:
        from video_generator import VideoGenerator
        components['video_generator'] = VideoGenerator()
        st.success("✅ Video Generator loaded successfully!")
    except ImportError as e:
        st.warning(f"⚠️ Video Generator not available: {e}")
        components['video_generator'] = None
    
    try:
        from ai_analysis_engine import ComprehensiveAIAnalyzer
        components['ai_analyzer'] = ComprehensiveAIAnalyzer()
        st.success("✅ AI Analysis Engine loaded successfully!")
    except ImportError as e:
        st.warning(f"⚠️ AI Analysis Engine not available: {e}")
        components['ai_analyzer'] = None
    
    try:
        from clinical_assessment_advanced import AdvancedClinicalAssessment
        components['clinical_assessment'] = AdvancedClinicalAssessment()
        st.success("✅ Clinical Assessment loaded successfully!")
    except ImportError as e:
        st.warning(f"⚠️ Clinical Assessment not available: {e}")
        components['clinical_assessment'] = None
    
    try:
        from expert_collaboration_framework import ExpertCollaborationFramework
        components['expert_framework'] = ExpertCollaborationFramework()
        st.success("✅ Expert Collaboration Framework loaded successfully!")
    except ImportError as e:
        st.warning(f"⚠️ Expert Collaboration Framework not available: {e}")
        components['expert_framework'] = None
    
    return components

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎨 Children's Drawing Analysis System</h1>
        <p>Advanced AI-powered psychological assessment of children's drawings</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load components
    with st.spinner("Loading analysis components..."):
        components = load_analysis_components()
    
    # Sidebar configuration
    st.sidebar.title("🔧 Analysis Configuration")
    
    # Child information
    st.sidebar.subheader("👶 Child Information")
    child_age = st.sidebar.slider("Child's Age", min_value=2, max_value=18, value=6)
    drawing_context = st.sidebar.selectbox(
        "Drawing Context",
        ["Free Drawing", "House Drawing", "Family Drawing", "Tree Drawing", 
         "Person Drawing", "Animal Drawing", "School Assignment", "Therapeutic Session"]
    )
    
    # Analysis options
    st.sidebar.subheader("🔬 Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Basic Analysis", "Enhanced Analysis", "Scientific Validation", 
         "Clinical Assessment", "AI Multi-Model", "Complete Analysis"]
    )
    
    generate_pdf = st.sidebar.checkbox("📄 Generate PDF Report", value=True)
    generate_video = st.sidebar.checkbox("🎬 Generate Memory Video", value=False)
    
    if generate_video:
        video_style = st.sidebar.selectbox(
            "Video Animation Style",
            ["intelligent", "elements", "particle", "floating", "animated"]
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Drawing")
        uploaded_file = st.file_uploader(
            "Choose a drawing image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a clear image of the child's drawing"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Drawing", use_column_width=True)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                image.save(tmp_file.name)
                temp_image_path = tmp_file.name
            
            # Analysis button
            if st.button("🚀 Start Analysis", type="primary"):
                analyze_drawing(
                    temp_image_path, 
                    child_age, 
                    drawing_context, 
                    analysis_type,
                    components,
                    generate_pdf,
                    generate_video,
                    video_style if generate_video else None
                )
    
    with col2:
        st.subheader("ℹ️ About This System")
        st.markdown("""
        <div class="analysis-card">
            <h4>🎯 What We Analyze</h4>
            <ul>
                <li><strong>Developmental Assessment:</strong> Age-appropriate skills and milestones</li>
                <li><strong>Emotional Indicators:</strong> Mood, feelings, and emotional well-being</li>
                <li><strong>Cognitive Markers:</strong> Problem-solving and thinking patterns</li>
                <li><strong>Social Elements:</strong> Relationships and social awareness</li>
                <li><strong>Creative Expression:</strong> Artistic development and creativity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="analysis-card">
            <h4>🤖 AI Technologies Used</h4>
            <ul>
                <li><strong>Computer Vision:</strong> BLIP, CLIP, OpenCV</li>
                <li><strong>Language Models:</strong> GPT-4, Perplexity AI</li>
                <li><strong>Segmentation:</strong> SAM (Segment Anything Model)</li>
                <li><strong>Classification:</strong> Custom AI element classifier</li>
                <li><strong>Animation:</strong> Smart context-aware animations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # System status
        st.subheader("🔍 System Status")
        display_system_status(components)

def analyze_drawing(image_path, child_age, drawing_context, analysis_type, components, 
                   generate_pdf=True, generate_video=False, video_style=None):
    """Perform comprehensive drawing analysis"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Basic Analysis
        status_text.text("🔍 Starting basic analysis...")
        progress_bar.progress(10)
        
        results = None
        
        if analysis_type == "Basic Analysis":
            if components['enhanced_analyzer']:
                results = components['enhanced_analyzer'].analyze_drawing_comprehensive(
                    image_path, child_age, drawing_context
                )
            else:
                st.error("Enhanced analyzer not available for basic analysis")
                return
        
        elif analysis_type == "Enhanced Analysis":
            status_text.text("🧠 Performing enhanced analysis...")
            progress_bar.progress(20)
            
            if components['enhanced_analyzer']:
                results = components['enhanced_analyzer'].analyze_drawing_with_pdf_report(
                    image_path, child_age, drawing_context, generate_pdf
                )
            else:
                st.error("Enhanced analyzer not available")
                return
        
        elif analysis_type == "Scientific Validation":
            status_text.text("📊 Conducting scientific validation...")
            progress_bar.progress(30)
            
            if components['validated_analyzer']:
                results = components['validated_analyzer'].analyze_drawing_with_validation(
                    image_path, child_age, drawing_context
                )
            else:
                st.error("Scientific validation not available")
                return
        
        elif analysis_type == "Clinical Assessment":
            status_text.text("🏥 Performing clinical assessment...")
            progress_bar.progress(25)
            
            if components['clinical_assessment'] and components['enhanced_analyzer']:
                # First get basic analysis
                basic_results = components['enhanced_analyzer'].analyze_drawing_comprehensive(
                    image_path, child_age, drawing_context
                )
                
                # Then add clinical assessment
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                clinical_results = components['clinical_assessment'].conduct_trauma_assessment(
                    image_rgb, basic_results.get('traditional_analysis', {}), child_age
                )
                
                attachment_results = components['clinical_assessment'].assess_attachment_patterns(
                    image_rgb, basic_results.get('traditional_analysis', {}), drawing_context
                )
                
                # Combine results
                results = basic_results
                results['clinical_assessment'] = {
                    'trauma_assessment': clinical_results,
                    'attachment_assessment': attachment_results
                }
            else:
                st.error("Clinical assessment components not available")
                return
        
        elif analysis_type == "AI Multi-Model":
            status_text.text("🤖 Running multi-AI analysis...")
            progress_bar.progress(35)
            
            if components['ai_analyzer']:
                ai_results = components['ai_analyzer'].conduct_multi_ai_analysis(
                    image_path, child_age, drawing_context
                )
                
                # Also get enhanced analysis for comparison
                if components['enhanced_analyzer']:
                    enhanced_results = components['enhanced_analyzer'].analyze_drawing_comprehensive(
                        image_path, child_age, drawing_context
                    )
                    
                    results = enhanced_results
                    results['ai_multi_analysis'] = ai_results
                else:
                    results = {'ai_multi_analysis': ai_results}
            else:
                st.error("AI analyzer not available")
                return
        
        elif analysis_type == "Complete Analysis":
            status_text.text("🔬 Performing complete comprehensive analysis...")
            progress_bar.progress(40)
            
            # Run all available analyses
            if components['validated_analyzer']:
                results = components['validated_analyzer'].analyze_drawing_with_psydraw_validation(
                    image_path, child_age, drawing_context
                )
            elif components['enhanced_analyzer']:
                results = components['enhanced_analyzer'].analyze_drawing_with_pdf_report(
                    image_path, child_age, drawing_context, generate_pdf
                )
            else:
                st.error("No analysis components available")
                return
        
        progress_bar.progress(60)
        
        # Step 2: Generate Video if requested
        video_path = None
        if generate_video and components['video_generator'] and results:
            status_text.text("🎬 Generating memory video...")
            progress_bar.progress(70)
            
            try:
                video_result = components['video_generator'].generate_memory_video(
                    image_path,
                    results,
                    f"Watch this amazing {child_age}-year-old's {drawing_context.lower()} come to life!",
                    animation_style=video_style or 'intelligent'
                )
                
                if 'video_path' in video_result:
                    video_path = video_result['video_path']
                    st.success(f"✅ Video generated: {video_result['generation_method']}")
                else:
                    st.warning(f"⚠️ Video generation failed: {video_result.get('error', 'Unknown error')}")
            
            except Exception as e:
                st.warning(f"⚠️ Video generation failed: {str(e)}")
        
        progress_bar.progress(90)
        
        # Step 3: Display Results
        status_text.text("📊 Displaying results...")
        display_analysis_results(results, video_path, child_age, drawing_context)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Clean up temporary file
        if os.path.exists(image_path):
            os.unlink(image_path)
    
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)

def display_analysis_results(results, video_path=None, child_age=None, drawing_context=None):
    """Display comprehensive analysis results"""
    
    if not results or 'error' in results:
        st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error') if results else 'No results'}")
        return
    
    st.success("✅ Analysis completed successfully!")
    
    # Create tabs for different result sections
    tabs = st.tabs([
        "📊 Overview", 
        "🧠 Detailed Analysis", 
        "📈 Metrics", 
        "💡 Recommendations",
        "📄 Reports",
        "🎬 Media"
    ])
    
    with tabs[0]:  # Overview
        display_overview(results, child_age, drawing_context)
    
    with tabs[1]:  # Detailed Analysis
        display_detailed_analysis(results)
    
    with tabs[2]:  # Metrics
        display_metrics(results)
    
    with tabs[3]:  # Recommendations
        display_recommendations(results)
    
    with tabs[4]:  # Reports
        display_reports(results)
    
    with tabs[5]:  # Media
        display_media(results, video_path)

def display_overview(results, child_age, drawing_context):
    """Display analysis overview"""
    
    st.subheader("📊 Analysis Overview")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>👶</h3>
            <h4>Child Age</h4>
            <p>{} years</p>
        </div>
        """.format(child_age), unsafe_allow_html=True)
    
    with col2:
        confidence = results.get('confidence_scores', {}).get('overall', 0)
        st.markdown("""
        <div class="metric-card">
            <h3>🎯</h3>
            <h4>Confidence</h4>
            <p>{:.1%}</p>
        </div>
        """.format(confidence), unsafe_allow_html=True)
    
    with col3:
        analysis_count = len(results.get('llm_analyses', [])) + 1
        st.markdown("""
        <div class="metric-card">
            <h3>🤖</h3>
            <h4>AI Models</h4>
            <p>{} analyses</p>
        </div>
        """.format(analysis_count), unsafe_allow_html=True)
    
    with col4:
        quality = results.get('summary', {}).get('analysis_quality', 'Good')
        st.markdown("""
        <div class="metric-card">
            <h3>⭐</h3>
            <h4>Quality</h4>
            <p>{}</p>
        </div>
        """.format(quality), unsafe_allow_html=True)
    
    # AI Description
    if 'traditional_analysis' in results:
        ai_description = results['traditional_analysis'].get('blip_description', 'No description available')
        st.markdown(f"""
        <div class="analysis-card">
            <h4>🤖 AI Description</h4>
            <p>"{ai_description}"</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Developmental Assessment
    if 'traditional_analysis' in results and 'developmental_assessment' in results['traditional_analysis']:
        dev_assessment = results['traditional_analysis']['developmental_assessment']
        level = dev_assessment.get('level', 'unknown').replace('_', ' ').title()
        
        color = "green" if level == "Age Appropriate" else "orange" if level == "Above Expected" else "red"
        
        st.markdown(f"""
        <div class="analysis-card">
            <h4>📈 Developmental Level</h4>
            <p style="color: {color}; font-weight: bold; font-size: 1.2em;">{level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Emotional Indicators
    if 'traditional_analysis' in results and 'emotional_indicators' in results['traditional_analysis']:
        emotional = results['traditional_analysis']['emotional_indicators']
        mood = emotional.get('overall_mood', 'neutral').title()
        
        mood_color = "green" if mood == "Positive" else "red" if mood == "Concerning" else "blue"
        
        st.markdown(f"""
        <div class="analysis-card">
            <h4>😊 Emotional Mood</h4>
            <p style="color: {mood_color}; font-weight: bold; font-size: 1.2em;">{mood}</p>
        </div>
        """, unsafe_allow_html=True)

def display_detailed_analysis(results):
    """Display detailed analysis results"""
    
    st.subheader("🧠 Detailed Analysis")
    
    # Traditional Analysis
    if 'traditional_analysis' in results:
        with st.expander("🔍 Traditional Computer Vision Analysis", expanded=True):
            traditional = results['traditional_analysis']
            
            # Color Analysis
            if 'color_analysis' in traditional:
                st.write("**🎨 Color Analysis:**")
                color_data = traditional['color_analysis']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Dominant Color", color_data.get('dominant_color', 'Unknown'))
                    st.metric("Color Diversity", color_data.get('color_diversity', 0))
                with col2:
                    st.metric("Brightness Level", f"{color_data.get('brightness_level', 0):.0f}/255")
                    st.metric("Color Richness", color_data.get('color_richness', 'Unknown'))
            
            # Shape Analysis
            if 'shape_analysis' in traditional:
                st.write("**🔷 Shape Analysis:**")
                shape_data = traditional['shape_analysis']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Shapes", shape_data.get('total_shapes', 0))
                    st.metric("Complexity Level", shape_data.get('complexity_level', 'Unknown'))
                with col2:
                    st.metric("Drawing Coverage", f"{shape_data.get('drawing_coverage', 0):.1%}")
                    st.metric("Detail Level", shape_data.get('detail_level', 'Unknown'))
            
            # Spatial Analysis
            if 'spatial_analysis' in traditional:
                st.write("**📐 Spatial Analysis:**")
                spatial_data = traditional['spatial_analysis']
                
                st.metric("Spatial Balance", spatial_data.get('spatial_balance', 'Unknown'))
                st.metric("Drawing Style", spatial_data.get('drawing_style', 'Unknown'))
    
    # LLM Analyses
    if 'llm_analyses' in results and results['llm_analyses']:
        with st.expander("🤖 AI Expert Analyses", expanded=True):
            for i, analysis in enumerate(results['llm_analyses']):
                st.write(f"**{analysis['provider'].title()} Analysis:**")
                st.write(f"*Confidence: {analysis['confidence']:.1%}*")
                st.write(analysis['analysis'])
                
                if i < len(results['llm_analyses']) - 1:
                    st.divider()
    
    # Scientific Validation
    if 'scientific_validation' in results:
        with st.expander("📊 Scientific Validation", expanded=False):
            validation = results['scientific_validation']
            
            if 'validation_metrics' in validation:
                metrics = validation['validation_metrics']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Reliability", f"{metrics.get('reliability', 0):.1%}")
                with col2:
                    st.metric("Validity", f"{metrics.get('validity', 0):.1%}")
                with col3:
                    st.metric("Research Alignment", f"{validation.get('research_alignment', {}).get('overall_research_alignment', 0):.1%}")
            
            if 'bias_analysis' in validation:
                bias = validation['bias_analysis']
                st.write(f"**Bias Risk Level:** {bias.get('overall_bias_risk', 'Unknown')}")
    
    # Clinical Assessment
    if 'clinical_assessment' in results:
        with st.expander("🏥 Clinical Assessment", expanded=False):
            clinical = results['clinical_assessment']
            
            if 'trauma_assessment' in clinical:
                trauma = clinical['trauma_assessment']
                st.write("**Trauma Risk Assessment:**")
                st.write(f"Risk Level: {trauma.get('risk_level', 'Unknown')}")
                
                if 'trauma_flags' in trauma and trauma['trauma_flags']:
                    st.write("**Clinical Flags:**")
                    for flag in trauma['trauma_flags']:
                        st.warning(f"⚠️ {flag.indicator_type}: {flag.description}")
            
            if 'attachment_assessment' in clinical:
                attachment = clinical['attachment_assessment']
                st.write("**Attachment Assessment:**")
                st.write(f"Attachment Style: {attachment.get('attachment_style', 'Unknown')}")

def display_metrics(results):
    """Display analysis metrics and statistics"""
    
    st.subheader("📈 Analysis Metrics")
    
    # Confidence Scores
    if 'confidence_scores' in results:
        st.write("**🎯 Confidence Scores:**")
        confidence = results['confidence_scores']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Traditional ML", f"{confidence.get('traditional_ml', 0):.1%}")
        with col2:
            st.metric("LLM Average", f"{confidence.get('llm_average', 0):.1%}")
        with col3:
            st.metric("Overall", f"{confidence.get('overall', 0):.1%}")
    
    # Analysis Quality Indicators
    if 'summary' in results:
        summary = results['summary']
        
        st.write("**📊 Analysis Quality:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analysis Quality", summary.get('analysis_quality', 'Unknown'))
            st.metric("Total Analyses", summary.get('total_analyses', 0))
        with col2:
            st.metric("Available Providers", len(summary.get('available_providers', [])))
    
    # Enhanced Summary Metrics
    if 'enhanced_summary' in results and 'confidence_indicators' in results['enhanced_summary']:
        indicators = results['enhanced_summary']['confidence_indicators']
        
        st.write("**🔍 Quality Indicators:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Quality", indicators.get('data_quality', 'Unknown'))
        with col2:
            st.metric("Analysis Depth", indicators.get('analysis_depth', 'Unknown'))
        with col3:
            st.metric("Reliability Score", f"{indicators.get('reliability_score', 0):.1%}")

def display_recommendations(results):
    """Display recommendations and action plans"""
    
    st.subheader("💡 Recommendations & Action Plan")
    
    # Enhanced Summary Recommendations
    if 'enhanced_summary' in results and 'enhanced_recommendations' in results['enhanced_summary']:
        recommendations = results['enhanced_summary']['enhanced_recommendations']
        
        # Immediate Actions
        if 'immediate_actions' in recommendations and recommendations['immediate_actions']:
            st.write("**🚨 Immediate Actions:**")
            for action in recommendations['immediate_actions']:
                st.info(f"• {action}")
        
        # Short-term Goals
        if 'short_term_goals' in recommendations and recommendations['short_term_goals']:
            st.write("**📅 Short-term Goals (1-3 months):**")
            for goal in recommendations['short_term_goals']:
                st.success(f"• {goal}")
        
        # Long-term Development
        if 'long_term_development' in recommendations and recommendations['long_term_development']:
            st.write("**🎯 Long-term Development (3-12 months):**")
            for goal in recommendations['long_term_development']:
                st.info(f"• {goal}")
        
        # Materials and Activities
        if 'materials_and_activities' in recommendations:
            materials = recommendations['materials_and_activities']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'recommended_materials' in materials:
                    st.write("**🎨 Recommended Materials:**")
                    for material in materials['recommended_materials']:
                        st.write(f"• {material}")
            
            with col2:
                if 'suggested_activities' in materials:
                    st.write("**🎮 Suggested Activities:**")
                    for activity in materials['suggested_activities']:
                        st.write(f"• {activity}")
        
        # When to Seek Help
        if 'when_to_seek_help' in recommendations and recommendations['when_to_seek_help']:
            st.write("**⚠️ When to Seek Professional Help:**")
            for indicator in recommendations['when_to_seek_help']:
                st.warning(f"• {indicator}")
    
    # Action Plan
    if 'enhanced_summary' in results and 'action_plan' in results['enhanced_summary']:
        action_plan = results['enhanced_summary']['action_plan']
        
        st.write("**📋 Specific Action Plan:**")
        
        plan_tabs = st.tabs(["This Week", "This Month", "Next 3 Months", "Ongoing"])
        
        with plan_tabs[0]:
            if 'this_week' in action_plan:
                for action in action_plan['this_week']:
                    st.checkbox(action, key=f"week_{hash(action)}")
        
        with plan_tabs[1]:
            if 'this_month' in action_plan:
                for action in action_plan['this_month']:
                    st.checkbox(action, key=f"month_{hash(action)}")
        
        with plan_tabs[2]:
            if 'next_3_months' in action_plan:
                for action in action_plan['next_3_months']:
                    st.checkbox(action, key=f"quarter_{hash(action)}")
        
        with plan_tabs[3]:
            if 'ongoing_support' in action_plan:
                for action in action_plan['ongoing_support']:
                    st.checkbox(action, key=f"ongoing_{hash(action)}")

def display_reports(results):
    """Display and download reports"""
    
    st.subheader("📄 Reports & Downloads")
    
    # PDF Report
    if 'pdf_report_filename' in results and results['pdf_report_filename']:
        pdf_file = results['pdf_report_filename']
        
        if os.path.exists(pdf_file):
            st.success("✅ PDF Report Generated Successfully!")
            
            # Read PDF file for download
            with open(pdf_file, "rb") as file:
                pdf_data = file.read()
            
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_data,
                file_name=f"drawing_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
            
            st.info(f"📁 Report saved as: {pdf_file}")
        else:
            st.warning("⚠️ PDF report was generated but file not found")
    
    # JSON Data Export
    if st.button("📊 Export Analysis Data (JSON)"):
        # Create exportable data (remove non-serializable items)
        export_data = {}
        for key, value in results.items():
            try:
                json.dumps(value)  # Test if serializable
                export_data[key] = value
            except (TypeError, ValueError):
                export_data[key] = str(value)  # Convert to string if not serializable
        
        json_data = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="📊 Download Analysis Data",
            data=json_data,
            file_name=f"drawing_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Raw Results (for debugging)
    with st.expander("🔧 Raw Analysis Results (Debug)", expanded=False):
        st.json(results)

def display_media(results, video_path=None):
    """Display media content (videos, animations)"""
    
    st.subheader("🎬 Generated Media")
    
    # Memory Video
    if video_path and os.path.exists(video_path):
        st.success("✅ Memory Video Generated Successfully!")
        
        # Display video
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
        
        st.video(video_bytes)
        
        # Download button
        st.download_button(
            label="🎬 Download Memory Video",
            data=video_bytes,
            file_name=f"memory_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
            mime="video/mp4"
        )
        
        st.info(f"📁 Video saved as: {video_path}")
    
    elif video_path:
        st.warning("⚠️ Video was generated but file not found")
    
    else:
        st.info("ℹ️ No video generated. Enable video generation in the sidebar to create animated memories!")
    
    # Video Generation Options
    st.write("**🎨 Available Animation Styles:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("• **Intelligent:** AI-powered component animation")
        st.write("• **Elements:** Individual element animations")
        st.write("• **Particle:** Particle effect animations")
    
    with col2:
        st.write("• **Floating:** Floating and orbiting effects")
        st.write("• **Animated:** Standard animation effects")

def display_system_status(components):
    """Display system component status"""
    
    status_items = [
        ("Enhanced Analyzer", components.get('enhanced_analyzer') is not None),
        ("Video Generator", components.get('video_generator') is not None),
        ("AI Analysis Engine", components.get('ai_analyzer') is not None),
        ("Clinical Assessment", components.get('clinical_assessment') is not None),
        ("Expert Framework", components.get('expert_framework') is not None),
    ]
    
    for name, available in status_items:
        if available:
            st.success(f"✅ {name}")
        else:
            st.error(f"❌ {name}")

if __name__ == "__main__":
    main()