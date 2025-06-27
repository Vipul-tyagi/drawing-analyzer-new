# 🎨 Children's Drawing Analysis System

A comprehensive AI-powered system for analyzing children's drawings with psychological insights, developmental assessment, and interactive visualizations.

## ✨ Features

### 🤖 AI-Powered Analysis
- **Multi-Model AI**: OpenAI GPT-4 Vision, Perplexity, BLIP, CLIP
- **Computer Vision**: Advanced image processing and element detection
- **Psychological Assessment**: Evidence-based psychological evaluation
- **Scientific Validation**: Research-backed analysis with bias detection

### 📊 Comprehensive Reports
- **PDF Reports**: Professional, detailed analysis reports
- **Interactive Dashboard**: Real-time analysis visualization
- **Expert Collaboration**: Framework for professional review
- **Data Export**: JSON and PDF export capabilities

### 🎬 Memory Videos
- **Intelligent Animation**: AI-powered component animation
- **Element-Based**: Individual drawing elements come alive
- **Multiple Styles**: Particle effects, floating animations, and more
- **Context-Aware**: Animations adapt to drawing content

### 🔬 Advanced Features
- **Clinical Assessment**: Trauma and attachment indicators
- **Developmental Tracking**: Age-appropriate milestone assessment
- **Research Validation**: Scientific methodology compliance
- **Bias Detection**: Automated bias identification and mitigation

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for video generation)
- API keys for AI services (optional but recommended)

### Installation

1. **Clone or download the project**
2. **Run the setup script:**
   ```bash
   python setup_environment.py
   ```

3. **Configure API keys** (optional but recommended):
   Edit the `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PERPLEXITY_API_KEY=your_perplexity_api_key_here
   ```

4. **Start the application:**
   ```bash
   python run_app.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to the displayed URL (usually http://localhost:8501)

## 📱 How to Use

### Basic Analysis
1. **Upload a drawing** - Support for PNG, JPG, JPEG, BMP, TIFF
2. **Enter child information** - Age and drawing context
3. **Select analysis type** - Choose from basic to comprehensive analysis
4. **Click "Start Analysis"** - Wait for AI processing
5. **Review results** - Explore detailed insights and recommendations

### Analysis Types

- **Basic Analysis**: Core computer vision and AI description
- **Enhanced Analysis**: Multi-AI analysis with detailed insights
- **Scientific Validation**: Research-backed validation and bias detection
- **Clinical Assessment**: Trauma and attachment pattern analysis
- **AI Multi-Model**: Multiple AI models for comprehensive analysis
- **Complete Analysis**: All features combined with PsyDraw validation

### Features

#### 📊 Analysis Dashboard
- **Overview**: Key metrics and AI description
- **Detailed Analysis**: Computer vision, AI expert analyses
- **Metrics**: Confidence scores and quality indicators
- **Recommendations**: Actionable insights and development plans

#### 📄 Reports
- **PDF Generation**: Professional reports for sharing
- **Data Export**: JSON format for further analysis
- **Expert Review**: Standardized review packages

#### 🎬 Memory Videos
- **Animation Styles**: 
  - Intelligent: AI-powered component animation
  - Elements: Individual element animations
  - Particle: Particle effect animations
  - Floating: Floating and orbiting effects
  - Animated: Standard animation effects

## 🔧 Configuration

### API Keys (Optional)
The system works without API keys but provides enhanced analysis with them:

- **OpenAI**: For GPT-4 Vision analysis
- **Perplexity**: For research-backed insights
- **Anthropic**: For additional AI perspectives
- **HuggingFace**: For advanced AI models

### System Requirements

#### Minimum
- Python 3.8+
- 4GB RAM
- 2GB disk space

#### Recommended
- Python 3.9+
- 8GB RAM
- GPU for faster AI processing
- FFmpeg for video generation

## 🏗️ Architecture

### Core Components

1. **Enhanced Drawing Analyzer** (`enhanced_drawing_analyzer.py`)
   - Main analysis engine
   - Multi-AI integration
   - Scientific validation

2. **Video Generator** (`video_generator.py`)
   - Memory video creation
   - Multiple animation styles
   - AI-powered element animation

3. **AI Analysis Engine** (`ai_analysis_engine.py`)
   - Multi-model AI coordination
   - Consensus analysis
   - Expert collaboration

4. **Clinical Assessment** (`clinical_assessment_advanced.py`)
   - Trauma indicator detection
   - Attachment pattern analysis
   - Professional-grade assessment

5. **Research Validation** (`research_validation_module.py`)
   - Scientific methodology compliance
   - Bias detection and mitigation
   - Research alignment verification

### AI Technologies

- **Computer Vision**: BLIP, CLIP, OpenCV, SAM
- **Language Models**: GPT-4, Perplexity AI, Claude
- **Segmentation**: Segment Anything Model (SAM)
- **Classification**: Custom AI element classifier
- **Animation**: Context-aware smart animator

## 📚 Documentation

### Analysis Methodology
The system uses a multi-layered approach:

1. **Traditional Computer Vision**: Color, shape, and spatial analysis
2. **AI Description**: BLIP model for semantic understanding
3. **Expert AI Analysis**: Multiple LLMs provide psychological insights
4. **Scientific Validation**: Research-backed validation and bias detection
5. **Clinical Assessment**: Professional-grade psychological evaluation

### Psychological Framework
Based on established research:
- Goodenough-Harris Drawing Test principles
- House-Tree-Person (HTP) assessment methodology
- Kinetic Family Drawing (KFD) analysis
- Trauma-informed assessment practices
- Attachment theory applications

### Technical Details
- **Image Processing**: OpenCV, PIL, scikit-image
- **AI Models**: Transformers, PyTorch, CLIP
- **Web Interface**: Streamlit with custom CSS
- **Video Generation**: MoviePy with custom animations
- **Report Generation**: ReportLab for PDF creation

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- Additional AI model integrations
- New animation styles
- Enhanced clinical assessments
- Research validation improvements
- UI/UX enhancements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is designed for educational and supportive purposes only. It should not replace professional psychological assessment or consultation with qualified child development specialists, art therapists, or mental health professionals.

## 🆘 Support

### Common Issues

1. **Import Errors**: Run `python setup_environment.py` to install dependencies
2. **Video Generation Fails**: Install FFmpeg system-wide
3. **Slow Performance**: Consider using GPU acceleration
4. **API Errors**: Check your API keys in the `.env` file

### Getting Help

1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure API keys are correctly configured
4. Check system requirements are met

## 🎯 Roadmap

### Upcoming Features
- Real-time drawing analysis
- Multi-language support
- Advanced clinical modules
- Research data collection
- Mobile app version
- Collaborative analysis tools

### Research Integration
- Longitudinal study capabilities
- Cross-cultural validation
- Advanced statistical analysis
- Research data anonymization
- Academic collaboration tools

---

**Made with ❤️ for understanding children's creative expression**