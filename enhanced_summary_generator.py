import json
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

class EnhancedSummaryGenerator:
    """
    Creates high-quality summaries and recommendations from analysis results
    """
    
    def __init__(self):
        self.age_specific_insights = {
            'Toddler (2-3 years)': {
                'expected_skills': ['Scribbling', 'Basic shapes', 'Large movements'],
                'positive_indicators': ['Controlled scribbling', 'Color recognition', 'Shape attempts'],
                'concerns': ['No mark-making', 'Extreme aggression in strokes', 'Avoidance of drawing']
            },
            'Preschool (4-6 years)': {
                'expected_skills': ['Basic figures', 'Houses', 'Trees', 'People with heads and limbs'],
                'positive_indicators': ['Detailed figures', 'Color variety', 'Storytelling through art'],
                'concerns': ['Regression to scribbles', 'Limited color use', 'Repetitive themes']
            },
            'School Age (7-11 years)': {
                'expected_skills': ['Realistic proportions', 'Detailed scenes', 'Perspective attempts'],
                'positive_indicators': ['Complex compositions', 'Emotional expression', 'Creative themes'],
                'concerns': ['Perfectionism anxiety', 'Self-criticism', 'Copying only']
            },
            'Adolescent (12+ years)': {
                'expected_skills': ['Advanced techniques', 'Abstract concepts', 'Personal style'],
                'positive_indicators': ['Unique expression', 'Technical skill', 'Emotional depth'],
                'concerns': ['Artistic blocks', 'Comparison with others', 'Loss of creativity']
            }
        }
    
    def generate_enhanced_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive, high-quality summary"""
        
        # Extract key information
        child_age = analysis_results['input_info']['child_age']
        age_group = analysis_results['input_info']['age_group']
        drawing_context = analysis_results['input_info']['drawing_context']
        traditional_analysis = analysis_results['traditional_analysis']
        llm_analyses = analysis_results.get('llm_analyses', [])
        
        # Generate executive summary
        executive_summary = self._create_executive_summary(
            child_age, age_group, drawing_context, traditional_analysis, llm_analyses
        )
        
        # Generate detailed insights
        detailed_insights = self._create_detailed_insights(traditional_analysis, llm_analyses)
        
        # Generate enhanced recommendations
        enhanced_recommendations = self._create_enhanced_recommendations(
            child_age, age_group, traditional_analysis, llm_analyses
        )
        
        # Generate developmental milestones assessment
        developmental_assessment = self._assess_developmental_milestones(
            child_age, age_group, traditional_analysis
        )
        
        # Generate action plan
        action_plan = self._create_action_plan(
            child_age, traditional_analysis, enhanced_recommendations
        )
        
        return {
            'executive_summary': executive_summary,
            'detailed_insights': detailed_insights,
            'enhanced_recommendations': enhanced_recommendations,
            'developmental_assessment': developmental_assessment,
            'action_plan': action_plan,
            'confidence_indicators': self._calculate_confidence_indicators(analysis_results),
            'next_steps': self._suggest_next_steps(traditional_analysis, llm_analyses)
        }
    
    def _create_executive_summary(self, child_age, age_group, drawing_context, traditional_analysis, llm_analyses):
        """Create a concise executive summary"""
        
        # Analyze key themes from LLM responses
        key_themes = self._extract_key_themes(llm_analyses)
        
        # Determine overall developmental level
        dev_level = traditional_analysis['developmental_assessment']['level']
        emotional_mood = traditional_analysis['emotional_indicators']['overall_mood']
        
        # Create summary based on findings
        summary_parts = []
        
        # Opening statement
        summary_parts.append(
            f"Analysis of a {drawing_context.lower()} created by a {child_age}-year-old child "
            f"reveals {dev_level.replace('_', ' ')} developmental indicators."
        )
        
        # Emotional assessment
        if emotional_mood == 'positive':
            summary_parts.append("The drawing demonstrates positive emotional expression and healthy creative development.")
        elif emotional_mood == 'concerning':
            summary_parts.append("The analysis indicates some emotional themes that may benefit from gentle attention and support.")
        else:
            summary_parts.append("The drawing shows balanced emotional expression typical for this age group.")
        
        # Key strengths
        strengths = self._identify_key_strengths(traditional_analysis)
        if strengths:
            summary_parts.append(f"Notable strengths include: {', '.join(strengths[:3])}.")
        
        # AI expert consensus (if available)
        if llm_analyses:
            consensus = self._find_ai_consensus(llm_analyses)
            if consensus:
                summary_parts.append(f"AI expert analysis consensus: {consensus}")
        
        return {
            'main_summary': ' '.join(summary_parts),
            'key_findings': key_themes[:5],  # Top 5 findings
            'overall_assessment': self._determine_overall_assessment(dev_level, emotional_mood),
            'confidence_level': self._calculate_summary_confidence(traditional_analysis, llm_analyses)
        }
    
    def _create_detailed_insights(self, traditional_analysis, llm_analyses):
        """Create detailed insights from all analyses"""
        
        insights = {
            'visual_elements': self._analyze_visual_elements(traditional_analysis),
            'psychological_indicators': self._analyze_psychological_indicators(traditional_analysis),
            'developmental_markers': self._analyze_developmental_markers(traditional_analysis),
            'ai_expert_insights': self._synthesize_ai_insights(llm_analyses)
        }
        
        return insights
    
    def _create_enhanced_recommendations(self, child_age, age_group, traditional_analysis, llm_analyses):
        """Create comprehensive, actionable recommendations"""
        
        recommendations = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_development': [],
            'materials_and_activities': [],
            'when_to_seek_help': []
        }
        
        # Age-specific recommendations
        age_insights = self.age_specific_insights.get(age_group, {})
        
        # Immediate actions based on analysis
        dev_level = traditional_analysis['developmental_assessment']['level']
        emotional_mood = traditional_analysis['emotional_indicators']['overall_mood']
        
        if emotional_mood == 'concerning':
            recommendations['immediate_actions'].extend([
                "Engage in gentle conversation about the child's feelings and experiences",
                "Provide additional emotional support and validation",
                "Consider consulting with a school counselor or child psychologist if concerns persist"
            ])
        
        if dev_level == 'below_expected':
            recommendations['immediate_actions'].extend([
                "Increase drawing and creative activities in daily routine",
                "Provide age-appropriate art materials and tools",
                "Celebrate all artistic attempts to build confidence"
            ])
        elif dev_level == 'above_expected':
            recommendations['immediate_actions'].extend([
                "Provide more challenging artistic activities",
                "Consider enrolling in age-appropriate art classes",
                "Encourage exploration of different artistic mediums"
            ])
        
        # Short-term goals (1-3 months)
        recommendations['short_term_goals'] = self._generate_short_term_goals(child_age, traditional_analysis)
        
        # Long-term development (3-12 months)
        recommendations['long_term_development'] = self._generate_long_term_goals(child_age, age_group)
        
        # Materials and activities
        recommendations['materials_and_activities'] = self._suggest_materials_activities(child_age, traditional_analysis)
        
        # When to seek help
        recommendations['when_to_seek_help'] = self._generate_help_indicators(traditional_analysis, llm_analyses)
        
        # Extract recommendations from AI analyses
        ai_recommendations = self._extract_ai_recommendations(llm_analyses)
        if ai_recommendations:
            recommendations['ai_expert_suggestions'] = ai_recommendations
        
        return recommendations
    
    def _assess_developmental_milestones(self, child_age, age_group, traditional_analysis):
        """Assess against developmental milestones"""
        
        expected_skills = self.age_specific_insights.get(age_group, {}).get('expected_skills', [])
        
        # Analyze current skills demonstrated
        shape_count = traditional_analysis['shape_analysis']['total_shapes']
        complexity = traditional_analysis['shape_analysis']['complexity_level']
        spatial_balance = traditional_analysis['spatial_analysis']['spatial_balance']
        
        milestone_assessment = {
            'age_group': age_group,
            'expected_skills': expected_skills,
            'demonstrated_skills': self._identify_demonstrated_skills(traditional_analysis),
            'milestone_progress': self._calculate_milestone_progress(child_age, traditional_analysis),
            'areas_of_strength': self._identify_strength_areas(traditional_analysis),
            'areas_for_growth': self._identify_growth_areas(traditional_analysis)
        }
        
        return milestone_assessment
    
    def _create_action_plan(self, child_age, traditional_analysis, recommendations):
        """Create a specific action plan for parents/caregivers"""
        
        action_plan = {
            'this_week': [],
            'this_month': [],
            'next_3_months': [],
            'ongoing_support': []
        }
        
        # This week actions
        action_plan['this_week'] = [
            "Display the child's artwork prominently to show appreciation",
            "Spend 15-20 minutes doing art activities together",
            "Ask the child to tell stories about their drawings"
        ]
        
        # This month actions
        action_plan['this_month'] = [
            "Introduce new art materials (crayons, markers, colored pencils)",
            "Visit a local art museum or gallery together",
            "Create a dedicated art space in your home"
        ]
        
        # Next 3 months
        action_plan['next_3_months'] = recommendations['short_term_goals']
        
        # Ongoing support
        action_plan['ongoing_support'] = [
            "Maintain regular art time in daily routine",
            "Continue positive reinforcement and encouragement",
            "Monitor emotional expression through art",
            "Document artistic progress with photos"
        ]
        
        return action_plan
    
    # Helper methods for analysis
    def _extract_key_themes(self, llm_analyses):
        """Extract key themes from LLM analyses"""
        themes = []
        for analysis in llm_analyses:
            # Simple keyword extraction - in production, use more sophisticated NLP
            text = analysis['analysis'].lower()
            if 'positive' in text or 'healthy' in text:
                themes.append("Positive emotional expression")
            if 'creative' in text or 'imagination' in text:
                themes.append("Strong creative development")
            if 'age-appropriate' in text or 'typical' in text:
                themes.append("Age-appropriate skills")
        return list(set(themes))
    
    def _identify_key_strengths(self, traditional_analysis):
        """Identify key strengths from traditional analysis"""
        strengths = []
        
        if traditional_analysis['color_analysis']['color_diversity'] > 10:
            strengths.append("rich color usage")
        
        if traditional_analysis['shape_analysis']['complexity_level'] != 'Simple':
            strengths.append("complex composition")
        
        if traditional_analysis['spatial_analysis']['spatial_balance'] in ['Balanced', 'Very balanced']:
            strengths.append("good spatial organization")
        
        return strengths
    
    def _find_ai_consensus(self, llm_analyses):
        """Find consensus among AI analyses"""
        if len(llm_analyses) < 2:
            return None
        
        # Simple consensus finding - look for common themes
        common_words = ['positive', 'appropriate', 'healthy', 'creative', 'developing']
        consensus_themes = []
        
        for word in common_words:
            count = sum(1 for analysis in llm_analyses if word in analysis['analysis'].lower())
            if count >= len(llm_analyses) // 2:  # Majority agreement
                consensus_themes.append(word)
        
        if consensus_themes:
            return f"Shows {', '.join(consensus_themes)} development patterns"
        return None
    
    def _determine_overall_assessment(self, dev_level, emotional_mood):
        """Determine overall assessment"""
        if dev_level == 'above_expected' and emotional_mood == 'positive':
            return "Excellent"
        elif dev_level == 'age_appropriate' and emotional_mood in ['positive', 'neutral']:
            return "Good"
        elif dev_level == 'below_expected' or emotional_mood == 'concerning':
            return "Needs Attention"
        else:
            return "Satisfactory"
    
    def _calculate_summary_confidence(self, traditional_analysis, llm_analyses):
        """Calculate confidence in summary"""
        base_confidence = 0.8
        if llm_analyses:
            ai_confidence = np.mean([0.9 for _ in llm_analyses])  # Simplified
            return (base_confidence + ai_confidence) / 2
        return base_confidence
    
    # Additional helper methods would go here...
    def _analyze_visual_elements(self, traditional_analysis):
        return {
            'color_usage': traditional_analysis['color_analysis'],
            'shape_complexity': traditional_analysis['shape_analysis'],
            'spatial_organization': traditional_analysis['spatial_analysis']
        }
    
    def _analyze_psychological_indicators(self, traditional_analysis):
        return traditional_analysis['emotional_indicators']
    
    def _analyze_developmental_markers(self, traditional_analysis):
        return traditional_analysis['developmental_assessment']
    
    def _synthesize_ai_insights(self, llm_analyses):
        return [{'provider': a['provider'], 'key_insights': a['analysis'][:200] + '...'} for a in llm_analyses]
    
    def _generate_short_term_goals(self, child_age, traditional_analysis):
        goals = [
            "Encourage daily creative expression through various art forms",
            "Introduce new artistic techniques appropriate for age",
            "Build confidence through positive reinforcement"
        ]
        return goals
    
    def _generate_long_term_goals(self, child_age, age_group):
        goals = [
            "Develop consistent artistic practice and routine",
            "Explore various artistic mediums and techniques",
            "Foster emotional expression through creative outlets"
        ]
        return goals
    
    def _suggest_materials_activities(self, child_age, traditional_analysis):
        return {
            'recommended_materials': ['Crayons', 'Colored pencils', 'Markers', 'Paint', 'Paper varieties'],
            'suggested_activities': ['Free drawing', 'Guided drawing exercises', 'Art games', 'Story illustration']
        }
    
    def _generate_help_indicators(self, traditional_analysis, llm_analyses):
        return [
            "Persistent concerning themes in artwork",
            "Regression in developmental skills",
            "Extreme emotional reactions to art activities",
            "Complete avoidance of creative expression"
        ]
    
    def _extract_ai_recommendations(self, llm_analyses):
        # Extract recommendation sections from AI analyses
        recommendations = []
        for analysis in llm_analyses:
            text = analysis['analysis']
            # Simple extraction - look for recommendation sections
            if 'recommend' in text.lower():
                # Extract sentences containing recommendations
                sentences = text.split('.')
                rec_sentences = [s.strip() for s in sentences if 'recommend' in s.lower()]
                recommendations.extend(rec_sentences[:2])  # Top 2 per analysis
        return recommendations[:5]  # Limit to 5 total
    
    def _identify_demonstrated_skills(self, traditional_analysis):
        skills = []
        if traditional_analysis['shape_analysis']['total_shapes'] > 3:
            skills.append("Multiple shape creation")
        if traditional_analysis['color_analysis']['color_diversity'] > 5:
            skills.append("Color variety usage")
        if traditional_analysis['spatial_analysis']['spatial_balance'] != 'Unbalanced':
            skills.append("Spatial awareness")
        return skills
    
    def _calculate_milestone_progress(self, child_age, traditional_analysis):
        # Simplified milestone calculation
        expected_shapes = {2: 2, 3: 3, 4: 5, 5: 7, 6: 9, 7: 11, 8: 13}
        expected = expected_shapes.get(child_age, 10)
        actual = traditional_analysis['shape_analysis']['total_shapes']
        progress = min(100, (actual / expected) * 100)
        return f"{progress:.0f}%"
    
    def _identify_strength_areas(self, traditional_analysis):
        strengths = []
        if traditional_analysis['color_analysis']['color_diversity'] > 8:
            strengths.append("Color exploration")
        if traditional_analysis['shape_analysis']['complexity_level'] != 'Simple':
            strengths.append("Shape complexity")
        return strengths
    
    def _identify_growth_areas(self, traditional_analysis):
        growth_areas = []
        if traditional_analysis['shape_analysis']['total_shapes'] < 3:
            growth_areas.append("Shape development")
        if traditional_analysis['spatial_analysis']['spatial_balance'] == 'Unbalanced':
            growth_areas.append("Spatial organization")
        return growth_areas
    
    def _calculate_confidence_indicators(self, analysis_results):
        return {
            'data_quality': 'High' if analysis_results.get('llm_analyses') else 'Medium',
            'analysis_depth': 'Comprehensive' if len(analysis_results.get('llm_analyses', [])) > 1 else 'Standard',
            'reliability_score': analysis_results['confidence_scores']['overall']
        }
    
    def _suggest_next_steps(self, traditional_analysis, llm_analyses):
        return [
            "Continue regular artistic activities",
            "Monitor progress over next 2-3 months",
            "Consider follow-up analysis if concerns arise",
            "Consult professional if significant changes occur"
        ]

