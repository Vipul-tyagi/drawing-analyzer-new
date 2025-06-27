from datetime import datetime
from typing import Dict, List
import json

class ExpertCollaborationFramework:
    """
    Framework for expert collaboration and validation
    """
    
    def __init__(self):
        self.expert_queue = []
        self.validation_requests = []
        
    def prepare_expert_review_package(self, analysis_results: Dict, image_path: str) -> Dict:
        """Prepare standardized package for expert review"""
        # Safe extraction of data with fallbacks
        input_info = analysis_results.get('input_info', {})
        traditional_analysis = analysis_results.get('traditional_analysis', {})
        
        review_package = {
            'case_id': f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'child_age': input_info.get('child_age', 'Unknown'),
            'drawing_context': input_info.get('drawing_context', 'Unknown'),
            'ai_assessment_summary': self._create_expert_summary(analysis_results),
            'specific_questions': self._generate_expert_questions(analysis_results),
            'image_path': image_path,
            'review_form': self._create_review_form(),
            'estimated_review_time': '15-20 minutes'
        }
        
        return review_package
    
    def _create_expert_summary(self, results: Dict) -> str:
        """Create concise summary for expert review"""
        traditional = results.get('traditional_analysis', {})
        dev_assessment = traditional.get('developmental_assessment', {})
        emotional = traditional.get('emotional_indicators', {})
        confidence_scores = results.get('confidence_scores', {})
        
        summary = f"""
AI Analysis Summary:
- Developmental Level: {dev_assessment.get('level', 'unknown')}
- Emotional Indicators: {emotional.get('overall_mood', 'unknown')}
- Complexity Level: {traditional.get('shape_analysis', {}).get('complexity_level', 'unknown')}
- Overall Confidence: {confidence_scores.get('overall', 0):.1%}
- Number of AI Analyses: {len(results.get('llm_analyses', []))}
        """
        return summary.strip()
    
    def _generate_expert_questions(self, results: Dict) -> List[str]:
        """Generate specific questions for expert review"""
        questions = [
            "Do you agree with the developmental level assessment?",
            "Are there emotional indicators we may have missed?",
            "What cultural factors should be considered?",
            "Are there any concerning elements requiring attention?",
            "What recommendations would you add or modify?"
        ]
        
        # Add specific questions based on AI findings
        traditional = results.get('traditional_analysis', {})
        emotional = traditional.get('emotional_indicators', {})
        
        if emotional.get('overall_mood') == 'concerning':
            questions.append("What specific emotional support would you recommend?")
        
        dev_level = traditional.get('developmental_assessment', {}).get('level')
        if dev_level == 'below_expected':
            questions.append("What developmental support strategies would you suggest?")
        elif dev_level == 'above_expected':
            questions.append("How can we nurture this child's advanced abilities?")
        
        return questions
    
    def _create_review_form(self) -> Dict:
        """Create standardized review form for experts"""
        return {
            'developmental_assessment': {
                'scale': 'below_expected|age_appropriate|above_expected',
                'confidence': 'scale_1_to_10',
                'notes': 'text_field'
            },
            'emotional_assessment': {
                'overall_mood': 'positive|neutral|concerning',
                'specific_emotions': 'multiple_choice',
                'confidence': 'scale_1_to_10',
                'notes': 'text_field'
            },
            'recommendations': {
                'immediate_actions': 'text_field',
                'follow_up_needed': 'yes|no',
                'referral_suggested': 'yes|no',
                'notes': 'text_field'
            },
            'ai_assessment_feedback': {
                'accuracy_rating': 'scale_1_to_10',
                'missed_elements': 'text_field',
                'incorrect_interpretations': 'text_field',
                'suggestions_for_improvement': 'text_field'
            }
        }
    
    def create_crowdsourced_validation_system(self) -> Dict:
        """Create system for crowdsourced expert validation"""
        return {
            'platform_requirements': [
                'Secure, HIPAA-compliant platform',
                'Expert credential verification',
                'Anonymized case presentation',
                'Standardized review interface',
                'Inter-rater reliability tracking'
            ],
            'expert_recruitment': [
                'Licensed child psychologists',
                'Art therapists',
                'Developmental specialists',
                'School counselors',
                'Pediatric professionals'
            ],
            'incentive_structure': [
                'Professional development credits',
                'Research collaboration opportunities',
                'Access to aggregated insights',
                'Contribution recognition'
            ],
            'quality_assurance': [
                'Multiple expert reviews per case',
                'Consensus requirement for validation',
                'Regular calibration exercises',
                'Feedback loop for AI improvement'
            ]
        }
    
    def validate_expert_credentials(self, expert_info: Dict) -> bool:
        """Validate expert credentials (placeholder implementation)"""
        required_fields = ['license_number', 'specialization', 'years_experience']
        return all(field in expert_info for field in required_fields)
    
    def calculate_inter_rater_reliability(self, expert_assessments: List[Dict]) -> Dict:
        """Calculate inter-rater reliability metrics"""
        if len(expert_assessments) < 2:
            return {'error': 'Need at least 2 expert assessments'}
        
        # Simple agreement calculation (in production, use more sophisticated metrics)
        agreements = []
        for i in range(len(expert_assessments)):
            for j in range(i + 1, len(expert_assessments)):
                assessment1 = expert_assessments[i]
                assessment2 = expert_assessments[j]
                
                # Compare developmental assessments
                dev_agreement = (
                    assessment1.get('developmental_assessment', {}).get('level') ==
                    assessment2.get('developmental_assessment', {}).get('level')
                )
                agreements.append(dev_agreement)
        
        agreement_rate = sum(agreements) / len(agreements) if agreements else 0
        
        return {
            'agreement_rate': agreement_rate,
            'total_comparisons': len(agreements),
            'reliability_level': 'high' if agreement_rate > 0.8 else 'moderate' if agreement_rate > 0.6 else 'low'
        }
    
    def generate_consensus_report(self, expert_assessments: List[Dict]) -> Dict:
        """Generate consensus report from multiple expert assessments"""
        if not expert_assessments:
            return {'error': 'No expert assessments provided'}
        
        # Count developmental level assessments
        dev_levels = [
            assessment.get('developmental_assessment', {}).get('level', 'unknown')
            for assessment in expert_assessments
        ]
        dev_level_counts = {}
        for level in dev_levels:
            dev_level_counts[level] = dev_level_counts.get(level, 0) + 1
        
        # Find consensus developmental level
        consensus_dev_level = max(dev_level_counts, key=dev_level_counts.get)
        
        # Count emotional assessments
        emotional_moods = [
            assessment.get('emotional_assessment', {}).get('overall_mood', 'unknown')
            for assessment in expert_assessments
        ]
        mood_counts = {}
        for mood in emotional_moods:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        
        consensus_mood = max(mood_counts, key=mood_counts.get)
        
        # Calculate confidence in consensus
        total_experts = len(expert_assessments)
        dev_consensus_strength = dev_level_counts[consensus_dev_level] / total_experts
        mood_consensus_strength = mood_counts[consensus_mood] / total_experts
        
        return {
            'consensus_developmental_level': consensus_dev_level,
            'developmental_consensus_strength': dev_consensus_strength,
            'consensus_emotional_mood': consensus_mood,
            'emotional_consensus_strength': mood_consensus_strength,
            'total_expert_assessments': total_experts,
            'consensus_quality': 'strong' if min(dev_consensus_strength, mood_consensus_strength) > 0.7 else 'moderate'
        }
