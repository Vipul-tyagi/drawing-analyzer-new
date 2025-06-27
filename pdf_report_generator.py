from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white, blue, green, red, orange
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib import colors
from datetime import datetime
import os
from PIL import Image as PILImage
import io
import base64

class PDFReportGenerator:
    """
    Generates beautiful, comprehensive PDF reports for children's drawing analysis
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom styles for the PDF"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=HexColor('#2E86AB'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=HexColor('#A23B72'),
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            spaceBefore=20,
            textColor=HexColor('#F18F01'),
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=HexColor('#F18F01'),
            borderPadding=5
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            textColor=HexColor('#333333'),
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        # Highlight style
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            textColor=HexColor('#2E86AB'),
            fontName='Helvetica-Bold',
            backColor=HexColor('#F0F8FF'),
            borderWidth=1,
            borderColor=HexColor('#2E86AB'),
            borderPadding=8
        ))
        
        # Recommendation style
        self.styles.add(ParagraphStyle(
            name='Recommendation',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            textColor=HexColor('#2D5016'),
            fontName='Helvetica',
            leftIndent=20,
            bulletIndent=10,
            bulletFontName='Symbol'
        ))
    
    def generate_comprehensive_report(self, analysis_results, enhanced_summary, image_path=None, output_filename=None):
        """Generate a comprehensive PDF report"""
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            child_age = analysis_results['input_info']['child_age']
            output_filename = f"drawing_analysis_report_{child_age}yr_{timestamp}.pdf"
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            output_filename,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build the story (content)
        story = []
        
        # Add cover page
        story.extend(self._create_cover_page(analysis_results, enhanced_summary))
        story.append(PageBreak())
        
        # Add executive summary
        story.extend(self._create_executive_summary_section(enhanced_summary))
        story.append(PageBreak())
        
        # Add drawing image if provided
        if image_path and os.path.exists(image_path):
            story.extend(self._create_drawing_display_section(image_path, analysis_results))
            story.append(PageBreak())
        
        # Add detailed analysis
        story.extend(self._create_detailed_analysis_section(analysis_results, enhanced_summary))
        story.append(PageBreak())
        
        # Add recommendations
        story.extend(self._create_recommendations_section(enhanced_summary))
        story.append(PageBreak())
        
        # Add action plan
        story.extend(self._create_action_plan_section(enhanced_summary))
        story.append(PageBreak())
        
        # Add appendix
        story.extend(self._create_appendix_section(analysis_results))
        
        # Build the PDF
        doc.build(story)
        
        return output_filename
    
    def _create_cover_page(self, analysis_results, enhanced_summary):
        """Create an attractive cover page"""
        story = []
        
        # Main title
        story.append(Paragraph("Children's Drawing Analysis Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 30))
        
        # Child information box
        child_info = f"""
        <para align="center" fontSize="14" textColor="#2E86AB">
        <b>Child Age:</b> {analysis_results['input_info']['child_age']} years<br/>
        <b>Age Group:</b> {analysis_results['input_info']['age_group']}<br/>
        <b>Drawing Context:</b> {analysis_results['input_info']['drawing_context']}<br/>
        <b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y')}
        </para>
        """
        story.append(Paragraph(child_info, self.styles['CustomBody']))
        story.append(Spacer(1, 40))
        
        # Overall assessment highlight
        overall_assessment = enhanced_summary['executive_summary']['overall_assessment']
        assessment_color = self._get_assessment_color(overall_assessment)
        
        assessment_text = f"""
        <para align="center" fontSize="18" textColor="{assessment_color}">
        <b>Overall Assessment: {overall_assessment}</b>
        </para>
        """
        story.append(Paragraph(assessment_text, self.styles['Highlight']))
        story.append(Spacer(1, 30))
        
        # Key findings summary
        story.append(Paragraph("Key Findings", self.styles['CustomSubtitle']))
        
        key_findings = enhanced_summary['executive_summary']['key_findings']
        for finding in key_findings[:5]:
            story.append(Paragraph(f"• {finding}", self.styles['CustomBody']))
        
        story.append(Spacer(1, 40))
        
        # Confidence indicator
        confidence = enhanced_summary['confidence_indicators']
        confidence_text = f"""
        <para align="center" fontSize="12" textColor="#666666">
        Analysis Quality: {confidence['data_quality']} | 
        Depth: {confidence['analysis_depth']} | 
        Reliability: {confidence['reliability_score']:.1%}
        </para>
        """
        story.append(Paragraph(confidence_text, self.styles['CustomBody']))
        
        # Disclaimer
        story.append(Spacer(1, 60))
        disclaimer = """
        <para align="center" fontSize="10" textColor="#999999">
        <i>This analysis is generated by AI systems and is intended for educational and supportive purposes only. 
        It should not replace professional psychological assessment or consultation with qualified child development specialists.</i>
        </para>
        """
        story.append(Paragraph(disclaimer, self.styles['CustomBody']))
        
        return story
    
    def _create_executive_summary_section(self, enhanced_summary):
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['CustomTitle']))
        
        exec_summary = enhanced_summary['executive_summary']
        
        # Main summary
        story.append(Paragraph("Overview", self.styles['SectionHeader']))
        story.append(Paragraph(exec_summary['main_summary'], self.styles['CustomBody']))
        story.append(Spacer(1, 20))
        
        # Confidence level
        confidence_text = f"Analysis Confidence Level: {exec_summary['confidence_level']:.1%}"
        story.append(Paragraph(confidence_text, self.styles['Highlight']))
        story.append(Spacer(1, 20))
        
        # Key findings in a table
        story.append(Paragraph("Key Findings", self.styles['SectionHeader']))
        
        findings_data = [['Finding', 'Significance']]
        for i, finding in enumerate(exec_summary['key_findings'][:5], 1):
            findings_data.append([f"{i}. {finding}", "Important"])
        
        findings_table = Table(findings_data, colWidths=[4*inch, 1.5*inch])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#CCCCCC'))
        ]))
        
        story.append(findings_table)
        
        return story
    
    def _create_drawing_display_section(self, image_path, analysis_results):
        """Create section to display the drawing"""
        story = []
        
        story.append(Paragraph("The Drawing", self.styles['CustomTitle']))
        
        try:
            # Add the drawing image
            img = Image(image_path, width=4*inch, height=4*inch)
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 20))
        except Exception as e:
            story.append(Paragraph(f"Drawing image could not be displayed: {str(e)}", self.styles['CustomBody']))
        
        # Add AI description
        ai_description = analysis_results['traditional_analysis']['blip_description']
        story.append(Paragraph("AI Description", self.styles['SectionHeader']))
        story.append(Paragraph(f'"{ai_description}"', self.styles['Highlight']))
        
        # Add basic analysis
        story.append(Paragraph("Quick Analysis", self.styles['SectionHeader']))
        
        color_analysis = analysis_results['traditional_analysis']['color_analysis']
        shape_analysis = analysis_results['traditional_analysis']['shape_analysis']
        
        quick_analysis = f"""
        <b>Dominant Color:</b> {color_analysis['dominant_color']}<br/>
        <b>Color Diversity:</b> {color_analysis['color_diversity']} different colors<br/>
        <b>Brightness Level:</b> {color_analysis['brightness_level']:.0f}/255<br/>
        <b>Total Shapes:</b> {shape_analysis['total_shapes']}<br/>
        <b>Complexity:</b> {shape_analysis['complexity_level']}
        """
        
        story.append(Paragraph(quick_analysis, self.styles['CustomBody']))
        
        return story
    
    def _create_detailed_analysis_section(self, analysis_results, enhanced_summary):
        """Create detailed analysis section"""
        story = []
        
        story.append(Paragraph("Detailed Analysis", self.styles['CustomTitle']))
        
        # Developmental Assessment
        story.append(Paragraph("Developmental Assessment", self.styles['SectionHeader']))
        
        dev_assessment = enhanced_summary['developmental_assessment']
        
        # Milestone progress
        milestone_text = f"""
        <b>Age Group:</b> {dev_assessment['age_group']}<br/>
        <b>Milestone Progress:</b> {dev_assessment['milestone_progress']}<br/>
        <b>Expected Skills:</b> {', '.join(dev_assessment['expected_skills'])}<br/>
        <b>Demonstrated Skills:</b> {', '.join(dev_assessment['demonstrated_skills'])}
        """
        story.append(Paragraph(milestone_text, self.styles['CustomBody']))
        story.append(Spacer(1, 15))
        
        # Strengths and growth areas
        strengths_data = [['Areas of Strength', 'Areas for Growth']]
        max_items = max(len(dev_assessment['areas_of_strength']), len(dev_assessment['areas_for_growth']))
        
        for i in range(max_items):
            strength = dev_assessment['areas_of_strength'][i] if i < len(dev_assessment['areas_of_strength']) else ""
            growth = dev_assessment['areas_for_growth'][i] if i < len(dev_assessment['areas_for_growth']) else ""
            strengths_data.append([strength, growth])
        
        strengths_table = Table(strengths_data, colWidths=[2.75*inch, 2.75*inch])
        strengths_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#A23B72')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#FFF0F5')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#CCCCCC')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        
        story.append(strengths_table)
        story.append(Spacer(1, 20))
        
        # Psychological Indicators
        story.append(Paragraph("Psychological Indicators", self.styles['SectionHeader']))
        
        emotional_indicators = analysis_results['traditional_analysis']['emotional_indicators']
        
        psych_text = f"""
        <b>Overall Mood:</b> {emotional_indicators['overall_mood'].title()}<br/>
        <b>Emotional Tone:</b> {emotional_indicators['tone'].title()}<br/>
        <b>Positive Indicators:</b> {emotional_indicators['positive_words_found']}<br/>
        <b>Areas of Attention:</b> {emotional_indicators['negative_words_found']}
        """
        story.append(Paragraph(psych_text, self.styles['CustomBody']))
        
        # AI Expert Insights
        if 'ai_expert_insights' in enhanced_summary['detailed_insights']:
            story.append(Spacer(1, 20))
            story.append(Paragraph("AI Expert Insights", self.styles['SectionHeader']))
            
            ai_insights = enhanced_summary['detailed_insights']['ai_expert_insights']
            for insight in ai_insights:
                provider_text = f"<b>{insight['provider']}:</b> {insight['key_insights']}"
                story.append(Paragraph(provider_text, self.styles['CustomBody']))
                story.append(Spacer(1, 10))
        
        return story
    
    def _create_recommendations_section(self, enhanced_summary):
        """Create recommendations section"""
        story = []
        
        story.append(Paragraph("Recommendations", self.styles['CustomTitle']))
        
        recommendations = enhanced_summary['enhanced_recommendations']
        
        # Immediate Actions
        if recommendations['immediate_actions']:
            story.append(Paragraph("Immediate Actions", self.styles['SectionHeader']))
            for action in recommendations['immediate_actions']:
                story.append(Paragraph(f"• {action}", self.styles['Recommendation']))
            story.append(Spacer(1, 15))
        
        # Short-term Goals
        if recommendations['short_term_goals']:
            story.append(Paragraph("Short-term Goals (1-3 months)", self.styles['SectionHeader']))
            for goal in recommendations['short_term_goals']:
                story.append(Paragraph(f"• {goal}", self.styles['Recommendation']))
            story.append(Spacer(1, 15))
        
        # Long-term Development
        if recommendations['long_term_development']:
            story.append(Paragraph("Long-term Development (3-12 months)", self.styles['SectionHeader']))
            for goal in recommendations['long_term_development']:
                story.append(Paragraph(f"• {goal}", self.styles['Recommendation']))
            story.append(Spacer(1, 15))
        
        # Materials and Activities
        if recommendations['materials_and_activities']:
            story.append(Paragraph("Recommended Materials & Activities", self.styles['SectionHeader']))
            
            materials = recommendations['materials_and_activities']
            if 'recommended_materials' in materials:
                story.append(Paragraph("<b>Materials:</b>", self.styles['CustomBody']))
                material_text = ", ".join(materials['recommended_materials'])
                story.append(Paragraph(material_text, self.styles['CustomBody']))
                story.append(Spacer(1, 10))
            
            if 'suggested_activities' in materials:
                story.append(Paragraph("<b>Activities:</b>", self.styles['CustomBody']))
                for activity in materials['suggested_activities']:
                    story.append(Paragraph(f"• {activity}", self.styles['Recommendation']))
        
        # When to Seek Help
        if recommendations['when_to_seek_help']:
            story.append(Spacer(1, 20))
            story.append(Paragraph("When to Seek Professional Help", self.styles['SectionHeader']))
            
            help_text = "Consider consulting with a child development specialist or counselor if you observe:"
            story.append(Paragraph(help_text, self.styles['CustomBody']))
            
            for indicator in recommendations['when_to_seek_help']:
                story.append(Paragraph(f"• {indicator}", self.styles['Recommendation']))
        
        return story
    
    def _create_action_plan_section(self, enhanced_summary):
        """Create action plan section"""
        story = []
        
        story.append(Paragraph("Action Plan", self.styles['CustomTitle']))
        
        action_plan = enhanced_summary['action_plan']
        
        # Create action plan table
        plan_data = [['Timeframe', 'Actions']]
        
        timeframes = ['this_week', 'this_month', 'next_3_months', 'ongoing_support']
        timeframe_labels = ['This Week', 'This Month', 'Next 3 Months', 'Ongoing Support']
        
        for timeframe, label in zip(timeframes, timeframe_labels):
            if timeframe in action_plan and action_plan[timeframe]:
                actions_text = '\n'.join([f"• {action}" for action in action_plan[timeframe]])
                plan_data.append([label, actions_text])
        
        plan_table = Table(plan_data, colWidths=[1.5*inch, 4*inch])
        plan_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#F18F01')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#FFF8F0')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#CCCCCC')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTSIZE', (0, 1), (-1, -1), 10)
        ]))
        
        story.append(plan_table)
        
        return story
    
    def _create_appendix_section(self, analysis_results):
        """Create appendix with technical details"""
        story = []
        
        story.append(Paragraph("Technical Appendix", self.styles['CustomTitle']))
        
        # Analysis metadata
        story.append(Paragraph("Analysis Details", self.styles['SectionHeader']))
        
        metadata_text = f"""
        <b>Analysis Timestamp:</b> {analysis_results['timestamp']}<br/>
        <b>Traditional ML Confidence:</b> {analysis_results['confidence_scores']['traditional_ml']:.1%}<br/>
        <b>LLM Average Confidence:</b> {analysis_results['confidence_scores']['llm_average']:.1%}<br/>
        <b>Overall Confidence:</b> {analysis_results['confidence_scores']['overall']:.1%}<br/>
        <b>Number of AI Analyses:</b> {len(analysis_results.get('llm_analyses', []))}
        """
        story.append(Paragraph(metadata_text, self.styles['CustomBody']))
        
        # Available providers
        if 'summary' in analysis_results:
            story.append(Spacer(1, 15))
            story.append(Paragraph("AI Providers Used", self.styles['SectionHeader']))
            
            providers = analysis_results['summary'].get('available_providers', [])
            if providers:
                provider_text = ", ".join([p.title() for p in providers])
                story.append(Paragraph(f"Active providers: {provider_text}", self.styles['CustomBody']))
            else:
                story.append(Paragraph("No external AI providers were available for this analysis.", self.styles['CustomBody']))
        
        # Disclaimer
        story.append(Spacer(1, 30))
        story.append(Paragraph("Important Disclaimer", self.styles['SectionHeader']))
        
        disclaimer_text = """
        This analysis is generated by artificial intelligence systems and computer vision algorithms. 
        It is intended for educational and supportive purposes only and should not replace professional 
        psychological assessment or consultation with qualified child development specialists, art therapists, 
        or mental health professionals. If you have concerns about a child's well-being, emotional state, 
        or development, please consult with appropriate licensed professionals.
        
        The AI systems used in this analysis are continuously improving but may occasionally produce 
        inaccuracies or miss important details. Always use your own judgment and seek professional 
        guidance when making important decisions about a child's care and development.
        """
        story.append(Paragraph(disclaimer_text, self.styles['CustomBody']))
        
        return story
    
    def _get_assessment_color(self, assessment):
        """Get color based on assessment level"""
        color_map = {
            'Excellent': '#2D5016',
            'Good': '#2E86AB',
            'Satisfactory': '#F18F01',
            'Needs Attention': '#A23B72'
        }
        return color_map.get(assessment, '#333333')

# Test function
def test_pdf_generator():
    """Test the PDF generator with sample data"""
    
    # Sample analysis results (simplified)
    sample_results = {
        'timestamp': datetime.now().isoformat(),
        'input_info': {
            'child_age': 6,
            'age_group': 'School Age (7-11 years)',
            'drawing_context': 'House Drawing'
        },
        'traditional_analysis': {
            'blip_description': 'a drawing of a house with a red roof and yellow sun',
            'color_analysis': {
                'dominant_color': 'Red',
                'color_diversity': 8,
                'brightness_level': 150
            },
            'shape_analysis': {
                'total_shapes': 5,
                'complexity_level': 'Medium'
            },
            'spatial_analysis': {
                'spatial_balance': 'Balanced'
            },
            'emotional_indicators': {
                'overall_mood': 'positive',
                'tone': 'bright_positive',
                'positive_words_found': 2,
                'negative_words_found': 0
            },
            'developmental_assessment': {
                'level': 'age_appropriate'
            }
        },
        'llm_analyses': [],
        'confidence_scores': {
            'traditional_ml': 0.85,
            'llm_average': 0.0,
            'overall': 0.85
        },
        'summary': {
            'available_providers': []
        }
    }
    
    # Sample enhanced summary
    sample_enhanced_summary = {
        'executive_summary': {
            'main_summary': 'This is a test summary of the drawing analysis.',
            'key_findings': ['Positive emotional expression', 'Age-appropriate skills'],
            'overall_assessment': 'Good',
            'confidence_level': 0.85
        },
        'detailed_insights': {
            'ai_expert_insights': []
        },
        'enhanced_recommendations': {
            'immediate_actions': ['Encourage continued drawing'],
            'short_term_goals': ['Introduce new materials'],
            'long_term_development': ['Develop artistic skills'],
            'materials_and_activities': {
                'recommended_materials': ['Crayons', 'Paper'],
                'suggested_activities': ['Free drawing']
            },
            'when_to_seek_help': ['Persistent concerning themes']
        },
        'developmental_assessment': {
            'age_group': 'School Age (7-11 years)',
            'milestone_progress': '85%',
            'expected_skills': ['Basic shapes', 'Color usage'],
            'demonstrated_skills': ['Shape creation', 'Color variety'],
            'areas_of_strength': ['Color usage'],
            'areas_for_growth': ['Spatial organization']
        },
        'action_plan': {
            'this_week': ['Display artwork', 'Art time together'],
            'this_month': ['New materials', 'Visit museum'],
            'next_3_months': ['Regular practice'],
            'ongoing_support': ['Maintain routine']
        },
        'confidence_indicators': {
            'data_quality': 'Medium',
            'analysis_depth': 'Standard',
            'reliability_score': 0.85
        }
    }
    
    # Generate PDF
    generator = PDFReportGenerator()
    filename = generator.generate_comprehensive_report(
        sample_results, 
        sample_enhanced_summary,
        output_filename="test_report.pdf"
    )
    
    print(f"Test PDF generated: {filename}")

if __name__ == "__main__":
    test_pdf_generator()

