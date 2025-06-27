import streamlit as st
import os
from dotenv import load_dotenv
from enhanced_drawing_analyzer import ScientificallyValidatedAnalyzer
import tempfile
from PIL import Image

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="AI-Enhanced Children's Drawing Analysis",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 AI-Enhanced Children's Drawing Analysis")
    st.markdown("**Powered by OpenAI, Perplexity, and Advanced Psychology AI**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Analysis Configuration")
        
        # API Status Check
        st.subheader("🔌 API Status")
        openai_status = "✅ Connected" if os.getenv('OPENAI_API_KEY') else "❌ Not configured"
        perplexity_status = "✅ Connected" if os.getenv('PERPLEXITY_API_KEY') else "❌ Not configured"
        
        st.write(f"OpenAI: {openai_status}")
        st.write(f"Perplexity: {perplexity_status}")
        
        if not os.getenv('OPENAI_API_KEY') and not os.getenv('PERPLEXITY_API_KEY'):
            st.error("⚠️ No API keys configured. Please set up your .env file.")
            st.stop()
        
        # Analysis options
        st.subheader("📊 Analysis Options")
        include_vision = st.checkbox("🔍 OpenAI Vision Analysis", value=True)
        include_research = st.checkbox("📚 Perplexity Research Validation", value=True)
        include_pdf = st.checkbox("📄 Generate PDF Report", value=True)
        
        # Child information
        st.subheader("👶 Child Information")
        child_age = st.slider("Child's Age", 2, 17, 7)
        drawing_context = st.selectbox(
            "Drawing Context",
            ["Free Drawing", "Family Drawing", "House-Tree-Person", "Self Portrait", "School Assignment"]
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Drawing")
        uploaded_file = st.file_uploader(
            "Choose a drawing image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the child's drawing"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Drawing", use_column_width=True)
            
            # Save temporarily for analysis
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                temp_image_path = tmp_file.name
            
            # Analysis button
            if st.button("🚀 Start AI Analysis", type="primary"):
                with st.spinner("🤖 AI experts are analyzing the drawing..."):
                    try:
                        # Initialize analyzer
                        analyzer = ScientificallyValidatedAnalyzer()
                        
                        # Run comprehensive analysis
                        if include_pdf:
                            results = analyzer.analyze_drawing_with_pdf_report(
                                temp_image_path, child_age, drawing_context, generate_pdf=True
                            )
                        else:
                            results = analyzer.analyze_drawing_with_psydraw_validation(
                                temp_image_path, child_age, drawing_context
                            )
                        
                        # Store results in session state
                        st.session_state['analysis_results'] = results
                        st.session_state['image_path'] = temp_image_path
                        
                        st.success("✅ Analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            
            # Display confidence scores
            if 'confidence_scores' in results:
                st.subheader("🎯 Confidence Scores")
                confidence = results['confidence_scores']
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Overall", f"{confidence.get('overall', 0):.2f}")
                with col_b:
                    st.metric("AI Analysis", f"{confidence.get('ai_analysis', 0):.2f}")
                with col_c:
                    st.metric("Traditional", f"{confidence.get('traditional', 0):.2f}")
            
            # Display LLM analyses
            if 'llm_analyses' in results:
                st.subheader("🧠 AI Expert Analyses")
                
                for i, analysis in enumerate(results['llm_analyses']):
                    with st.expander(f"🤖 {analysis['provider'].title()} Analysis (Confidence: {analysis['confidence']:.2f})"):
                        st.write(analysis['analysis'])
            
            # Display enhanced summary
            if 'enhanced_summary' in results:
                st.subheader("📋 Executive Summary")
                summary = results['enhanced_summary']
                
                if 'executive_summary' in summary:
                    exec_summary = summary['executive_summary']
                    st.write("**Main Assessment:**", exec_summary.get('main_summary', 'Analysis completed'))
                    
                    if 'key_findings' in exec_summary:
                        st.write("**Key Findings:**")
                        for finding in exec_summary['key_findings']:
                            st.write(f"• {finding}")
            
            # PDF download
            if 'pdf_report_filename' in results and results['pdf_report_filename']:
                st.subheader("📄 Download Report")
                with open(results['pdf_report_filename'], 'rb') as pdf_file:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_file.read(),
                        file_name=f"drawing_analysis_{child_age}yo.pdf",
                        mime="application/pdf"
                    )
        
        else:
            st.info("👆 Upload a drawing and click 'Start AI Analysis' to begin")

if __name__ == "__main__":
    main()

