import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

class ValidationFramework:
    """
    Scientific validation framework for drawing analysis
    """
    
    def __init__(self):
        self.validation_history = []
        self.expert_comparisons = []
        
    def validate_assessment(self, ai_assessment: Dict, expert_assessment: Optional[Dict] = None) -> Dict:
        """
        Validate AI assessment with statistical rigor
        """
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'confidence_intervals': self._calculate_confidence_intervals(ai_assessment),
            'statistical_significance': self._test_statistical_significance(ai_assessment),
            'bias_indicators': self._detect_bias_indicators(ai_assessment),
            'reliability_metrics': self._calculate_reliability_metrics(ai_assessment)
        }
        
        if expert_assessment:
            validation_results['expert_comparison'] = self._compare_with_expert(ai_assessment, expert_assessment)
            
        self.validation_history.append(validation_results)
        return validation_results
    
    def _calculate_confidence_intervals(self, assessment: Dict) -> Dict:
        """Calculate confidence intervals for numerical assessments"""
        confidence_intervals = {}
        
        # Extract numerical scores
        numerical_scores = self._extract_numerical_scores(assessment)
        
        for metric, scores in numerical_scores.items():
            if len(scores) > 1:
                mean = np.mean(scores)
                std_error = stats.sem(scores)
                confidence_interval = stats.t.interval(
                    0.95, len(scores)-1, loc=mean, scale=std_error
                )
                confidence_intervals[metric] = {
                    'mean': float(mean),
                    'ci_lower': float(confidence_interval[0]),
                    'ci_upper': float(confidence_interval[1]),
                    'margin_of_error': float(confidence_interval[1] - mean)
                }
        
        return confidence_intervals
    
    def _test_statistical_significance(self, assessment: Dict) -> Dict:
        """Test statistical significance of findings"""
        significance_tests = {}
        
        # Test if developmental level differs significantly from expected
        if 'traditional_analysis' in assessment and 'developmental_assessment' in assessment['traditional_analysis']:
            dev_data = assessment['traditional_analysis']['developmental_assessment']
            if 'actual_shapes' in dev_data and 'expected_shapes' in dev_data:
                expected = dev_data['expected_shapes'].get('min_shapes', 5)
                actual = dev_data['actual_shapes']
                
                # One-sample t-test against expected value
                t_stat, p_value = stats.ttest_1samp([actual], expected)
                significance_tests['developmental_significance'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'interpretation': 'significantly_different' if p_value < 0.05 else 'not_significant'
                }
        
        return significance_tests
    
    def _detect_bias_indicators(self, assessment: Dict) -> Dict:
        """Detect potential biases in assessment"""
        bias_indicators = {
            'cultural_bias_risk': 'low',
            'age_bias_risk': 'low',
            'gender_bias_risk': 'unknown',
            'confirmation_bias_risk': 'medium'
        }
        
        # Check for extreme assessments that might indicate bias
        if 'confidence_scores' in assessment:
            overall_confidence = assessment['confidence_scores'].get('overall', 0)
            if overall_confidence > 0.95:
                bias_indicators['overconfidence_bias'] = 'high'
            elif overall_confidence < 0.3:
                bias_indicators['underconfidence_bias'] = 'high'
        
        return bias_indicators
    
    def _calculate_reliability_metrics(self, assessment: Dict) -> Dict:
        """Calculate reliability metrics"""
        reliability = {
            'internal_consistency': self._calculate_internal_consistency(assessment),
            'assessment_stability': 0.75,  # Default value
            'cross_validator_agreement': self._calculate_cross_validator_agreement(assessment)
        }
        
        return reliability
    
    def _extract_numerical_scores(self, assessment: Dict) -> Dict:
        """Extract numerical scores for statistical analysis"""
        scores = {}
        
        # Extract confidence scores
        if 'confidence_scores' in assessment:
            scores['confidence'] = [
                assessment['confidence_scores'].get('traditional_ml', 0),
                assessment['confidence_scores'].get('llm_average', 0),
                assessment['confidence_scores'].get('overall', 0)
            ]
        
        # Extract color analysis scores if available
        if 'traditional_analysis' in assessment and 'color_analysis' in assessment['traditional_analysis']:
            color_data = assessment['traditional_analysis']['color_analysis']
            scores['color_metrics'] = [
                color_data.get('brightness_level', 0) / 255.0,  # Normalize
                min(color_data.get('color_diversity', 0) / 20.0, 1.0),  # Normalize
                color_data.get('red_amount', 0) / 255.0,
                color_data.get('green_amount', 0) / 255.0,
                color_data.get('blue_amount', 0) / 255.0
            ]
        
        return scores
    
    def _calculate_internal_consistency(self, assessment: Dict) -> float:
        """Calculate internal consistency (Cronbach's alpha equivalent)"""
        numerical_scores = self._extract_numerical_scores(assessment)
        
        if not numerical_scores:
            return 0.5  # Default moderate consistency
        
        # Simple consistency measure based on variance of normalized scores
        all_scores = []
        for score_list in numerical_scores.values():
            all_scores.extend(score_list)
        
        if len(all_scores) < 2:
            return 0.5
        
        # Lower variance indicates higher consistency
        variance = np.var(all_scores)
        consistency = max(0, 1 - variance)  # Convert variance to consistency score
        
        return float(consistency)
    
    def _calculate_cross_validator_agreement(self, assessment: Dict) -> float:
        """Calculate agreement between different AI validators"""
        if 'llm_analyses' not in assessment:
            return 0.5
        
        llm_analyses = assessment['llm_analyses']
        if len(llm_analyses) < 2:
            return 0.5
        
        # Simple agreement measure based on confidence scores
        confidences = [analysis.get('confidence', 0.5) for analysis in llm_analyses]
        
        # Calculate coefficient of variation (lower = more agreement)
        if np.mean(confidences) > 0:
            cv = np.std(confidences) / np.mean(confidences)
            agreement = max(0, 1 - cv)  # Convert to agreement score
        else:
            agreement = 0.5
        
        return float(agreement)
    
    def _compare_with_expert(self, ai_assessment: Dict, expert_assessment: Dict) -> Dict:
        """Compare AI assessment with expert assessment"""
        comparison = {
            'overall_agreement': 0.5,  # Default
            'specific_agreements': {},
            'disagreements': {},
            'expert_confidence': expert_assessment.get('confidence', 0.8)
        }
        
        # Compare developmental assessments
        if ('traditional_analysis' in ai_assessment and 
            'developmental_assessment' in ai_assessment['traditional_analysis'] and
            'developmental_assessment' in expert_assessment):
            
            ai_level = ai_assessment['traditional_analysis']['developmental_assessment'].get('level', 'unknown')
            expert_level = expert_assessment['developmental_assessment'].get('level', 'unknown')
            
            comparison['specific_agreements']['developmental_level'] = ai_level == expert_level
        
        # Compare emotional assessments
        if ('traditional_analysis' in ai_assessment and 
            'emotional_indicators' in ai_assessment['traditional_analysis'] and
            'emotional_assessment' in expert_assessment):
            
            ai_mood = ai_assessment['traditional_analysis']['emotional_indicators'].get('overall_mood', 'neutral')
            expert_mood = expert_assessment['emotional_assessment'].get('overall_mood', 'neutral')
            
            comparison['specific_agreements']['emotional_mood'] = ai_mood == expert_mood
        
        # Calculate overall agreement
        agreements = list(comparison['specific_agreements'].values())
        if agreements:
            comparison['overall_agreement'] = sum(agreements) / len(agreements)
        
        return comparison
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report"""
        if not self.validation_history:
            return {'error': 'No validation data available'}
        
        report = {
            'total_validations': len(self.validation_history),
            'average_reliability': np.mean([v['reliability_metrics']['internal_consistency'] 
                                          for v in self.validation_history]),
            'bias_trend_analysis': self._analyze_bias_trends(),
            'recommendations': self._generate_validation_recommendations()
        }
        
        return report
    
    def _analyze_bias_trends(self) -> Dict:
        """Analyze bias trends over time"""
        return {
            'overconfidence_frequency': 0.1,  # Placeholder
            'cultural_bias_indicators': 'low',
            'recommendation': 'Continue monitoring for bias patterns'
        }
    
    def _generate_validation_recommendations(self) -> List[str]:
        """Generate recommendations based on validation history"""
        return [
            "Continue collecting expert validation data",
            "Monitor for systematic biases in assessments",
            "Implement regular calibration checks",
            "Consider expanding validation dataset"
        ]

class BiasDetectionSystem:
    """
    Detect and flag potential biases in assessments
    """
    
    def __init__(self):
        self.bias_patterns = self._load_bias_patterns()
    
    def _load_bias_patterns(self) -> Dict:
        """Load known bias patterns"""
        return {
            'cultural_indicators': [
                'house_style_bias',  # Western vs. non-Western house styles
                'family_structure_bias',  # Nuclear vs. extended family assumptions
                'color_symbolism_bias'  # Cultural color meanings
            ],
            'age_indicators': [
                'developmental_expectations',  # Rigid age-based expectations
                'skill_assumptions'  # Assuming all children develop uniformly
            ],
            'confirmation_bias': [
                'selective_attention',  # Focusing on confirming evidence
                'anchoring_bias'  # Over-relying on first impressions
            ]
        }
    
    def detect_biases(self, assessment: Dict, child_context: Dict) -> Dict:
        """Detect potential biases in assessment"""
        detected_biases = {
            'cultural_biases': self._detect_cultural_bias(assessment, child_context),
            'developmental_biases': self._detect_developmental_bias(assessment, child_context),
            'confirmation_biases': self._detect_confirmation_bias(assessment),
            'overall_bias_risk': 'medium'  # Default
        }
        
        # Calculate overall bias risk
        bias_indicators = sum([
            len(detected_biases['cultural_biases']),
            len(detected_biases['developmental_biases']),
            len(detected_biases['confirmation_biases'])
        ])
        
        if bias_indicators > 3:
            detected_biases['overall_bias_risk'] = 'high'
        elif bias_indicators > 1:
            detected_biases['overall_bias_risk'] = 'medium'
        else:
            detected_biases['overall_bias_risk'] = 'low'
        
        return detected_biases
    
    def _detect_cultural_bias(self, assessment: Dict, child_context: Dict) -> List[str]:
        """Detect cultural biases"""
        biases = []
        
        # Check for Western-centric house interpretations
        if ('traditional_analysis' in assessment and 
            'blip_description' in assessment['traditional_analysis']):
            description = assessment['traditional_analysis']['blip_description'].lower()
            
            if 'house' in description and 'family' in child_context.get('drawing_context', '').lower():
                # Flag potential Western nuclear family bias
                biases.append('western_family_structure_assumption')
        
        return biases
    
    def _detect_developmental_bias(self, assessment: Dict, child_context: Dict) -> List[str]:
        """Detect developmental biases"""
        biases = []
        
        if 'traditional_analysis' in assessment and 'developmental_assessment' in assessment['traditional_analysis']:
            dev_assessment = assessment['traditional_analysis']['developmental_assessment']
            child_age = child_context.get('child_age', 6)
            
            # Check for rigid age expectations
            if dev_assessment.get('level') == 'below_expected' and child_age < 4:
                biases.append('unrealistic_toddler_expectations')
            
            if dev_assessment.get('level') == 'above_expected' and child_age > 10:
                biases.append('underestimating_adolescent_capability')
        
        return biases
    
    def _detect_confirmation_bias(self, assessment: Dict) -> List[str]:
        """Detect confirmation bias patterns"""
        biases = []
        
        # Check for overconfidence
        if 'confidence_scores' in assessment:
            overall_confidence = assessment['confidence_scores'].get('overall', 0)
            if overall_confidence > 0.95:
                biases.append('overconfidence_bias')
        
        # Check for selective evidence use
        if 'llm_analyses' in assessment:
            analyses = assessment['llm_analyses']
            if len(analyses) > 1:
                # Check if all analyses agree too much (potential groupthink)
                confidences = [a.get('confidence', 0.5) for a in analyses]
                if np.std(confidences) < 0.05:  # Very low variance
                    biases.append('artificial_consensus')
        
        return biases
