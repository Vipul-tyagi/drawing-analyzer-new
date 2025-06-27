import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean

class TraumaIndicator(Enum):
    FRAGMENTATION = "fragmentation"
    DISSOCIATION = "dissociation"
    HYPERVIGILANCE = "hypervigilance"
    REGRESSION = "regression"
    SOMATIC_MARKERS = "somatic_markers"

class AttachmentStyle(Enum):
    SECURE = "secure"
    ANXIOUS_AMBIVALENT = "anxious_ambivalent"
    AVOIDANT = "avoidant"
    DISORGANIZED = "disorganized"

@dataclass
class ClinicalFlag:
    indicator_type: str
    severity: str  # low, moderate, high, critical
    confidence: float
    description: str
    research_basis: str
    recommended_action: str

class AdvancedClinicalAssessment:
    """
    Advanced clinical assessment features for trauma, attachment, and family dynamics
    Based on latest research in art therapy and child psychology
    """
    
    def __init__(self):
        self.trauma_indicators = self._load_trauma_indicators()
        self.attachment_patterns = self._load_attachment_patterns()
        self.family_dynamics_markers = self._load_family_dynamics_markers()
    
    def _load_trauma_indicators(self) -> Dict:
        """Load research-based trauma indicators in children's drawings"""
        return {
            'fragmentation_markers': {
                'broken_lines': {'weight': 0.7, 'threshold': 0.3},
                'disconnected_elements': {'weight': 0.8, 'threshold': 0.4},
                'incomplete_figures': {'weight': 0.6, 'threshold': 0.5}
            },
            'dissociation_markers': {
                'floating_elements': {'weight': 0.8, 'threshold': 0.3},
                'lack_of_ground_line': {'weight': 0.5, 'threshold': 0.6},
                'surreal_combinations': {'weight': 0.9, 'threshold': 0.2}
            },
            'hypervigilance_markers': {
                'excessive_detail': {'weight': 0.6, 'threshold': 0.7},
                'multiple_eyes': {'weight': 0.9, 'threshold': 0.1},
                'defensive_postures': {'weight': 0.8, 'threshold': 0.2}
            },
            'regression_markers': {
                'age_inappropriate_simplicity': {'weight': 0.7, 'threshold': 0.4},
                'primitive_representations': {'weight': 0.6, 'threshold': 0.5}
            },
            'somatic_markers': {
                'body_distortions': {'weight': 0.8, 'threshold': 0.3},
                'missing_body_parts': {'weight': 0.7, 'threshold': 0.4},
                'emphasis_on_wounds': {'weight': 0.9, 'threshold': 0.1}
            }
        }
    
    def _load_attachment_patterns(self) -> Dict:
        """Load attachment patterns for psychological analysis"""
        try:
            # Define common attachment patterns used in clinical assessment
            attachment_patterns = {
                'secure': {
                    'indicators': ['consistent_caregiving', 'emotional_regulation', 'trust_building'],
                    'behaviors': ['seeks_comfort', 'explores_confidently', 'manages_separation'],
                    'drawing_markers': {
                        'family_proximity': {'weight': 0.8, 'threshold': 0.6},
                        'positive_expressions': {'weight': 0.7, 'threshold': 0.5},
                        'balanced_composition': {'weight': 0.6, 'threshold': 0.4}
                    }
                },
                'anxious_ambivalent': {
                    'indicators': ['inconsistent_caregiving', 'heightened_anxiety', 'fear_of_abandonment'],
                    'behaviors': ['clingy_behavior', 'difficulty_self_soothing', 'hypervigilant'],
                    'drawing_markers': {
                        'exaggerated_size_differences': {'weight': 0.8, 'threshold': 0.4},
                        'overlapping_figures': {'weight': 0.7, 'threshold': 0.3},
                        'anxious_expressions': {'weight': 0.9, 'threshold': 0.2}
                    }
                },
                'avoidant': {
                    'indicators': ['emotional_unavailability', 'rejection_sensitivity', 'self_reliance'],
                    'behaviors': ['avoids_intimacy', 'suppresses_emotions', 'independent_facade'],
                    'drawing_markers': {
                        'distant_figures': {'weight': 0.9, 'threshold': 0.3},
                        'minimal_interaction': {'weight': 0.8, 'threshold': 0.4},
                        'isolated_self_representation': {'weight': 0.7, 'threshold': 0.5}
                    }
                },
                'disorganized': {
                    'indicators': ['trauma_history', 'inconsistent_responses', 'conflicted_behaviors'],
                    'behaviors': ['approach_avoidance', 'emotional_dysregulation', 'dissociation'],
                    'drawing_markers': {
                        'chaotic_composition': {'weight': 0.9, 'threshold': 0.2},
                        'contradictory_elements': {'weight': 0.8, 'threshold': 0.3},
                        'fragmented_figures': {'weight': 0.8, 'threshold': 0.4}
                    }
                }
            }
            return attachment_patterns
        except Exception as e:
            print(f"Error loading attachment patterns: {e}")
            return {}
    
    def _load_family_dynamics_markers(self) -> Dict:
        """Load family dynamics assessment markers"""
        return {
            'healthy_dynamics': {
                'equal_sizing': {'weight': 0.7, 'threshold': 0.5},
                'appropriate_spacing': {'weight': 0.6, 'threshold': 0.4},
                'positive_interactions': {'weight': 0.8, 'threshold': 0.6}
            },
            'concerning_dynamics': {
                'power_imbalances': {'weight': 0.8, 'threshold': 0.3},
                'isolation_patterns': {'weight': 0.9, 'threshold': 0.2},
                'conflict_indicators': {'weight': 0.7, 'threshold': 0.4}
            }
        }
    
    def conduct_trauma_assessment(self, image: np.ndarray, psydraw_features: Dict, child_age: int) -> Dict:
        """Comprehensive trauma indicator assessment"""
        print("🚨 Conducting trauma indicator assessment...")
        trauma_flags = []
        
        # Analyze fragmentation patterns
        fragmentation_score = self._assess_fragmentation(image, psydraw_features)
        if fragmentation_score > self.trauma_indicators['fragmentation_markers']['broken_lines']['threshold']:
            trauma_flags.append(ClinicalFlag(
                indicator_type="Fragmentation",
                severity="moderate" if fragmentation_score < 0.7 else "high",
                confidence=0.75,
                description="Drawing shows signs of psychological fragmentation",
                research_basis="Malchiodi (2012) - Trauma indicators in children's art",
                recommended_action="Consider trauma-informed assessment"
            ))
        
        # Analyze dissociation markers
        dissociation_score = self._assess_dissociation_markers(image, psydraw_features)
        if dissociation_score > 0.4:
            trauma_flags.append(ClinicalFlag(
                indicator_type="Dissociation",
                severity="high",
                confidence=0.8,
                description="Possible dissociative patterns in drawing",
                research_basis="Cohen & Cox (1995) - Dissociation in children's art",
                recommended_action="Immediate clinical consultation recommended"
            ))
        
        # Analyze hypervigilance indicators
        hypervigilance_score = self._assess_hypervigilance_markers(image, psydraw_features)
        
        # Assess regression patterns
        regression_score = self._assess_regression_patterns(psydraw_features, child_age)
        
        # Analyze somatic markers
        somatic_score = self._assess_somatic_markers(image, psydraw_features)
        
        # Calculate overall trauma risk
        overall_trauma_risk = np.mean([
            fragmentation_score, dissociation_score,
            hypervigilance_score, regression_score, somatic_score
        ])
        
        return {
            'trauma_flags': trauma_flags,
            'trauma_risk_scores': {
                'fragmentation': fragmentation_score,
                'dissociation': dissociation_score,
                'hypervigilance': hypervigilance_score,
                'regression': regression_score,
                'somatic_markers': somatic_score
            },
            'overall_trauma_risk': overall_trauma_risk,
            'risk_level': self._categorize_trauma_risk(overall_trauma_risk),
            'immediate_recommendations': self._generate_trauma_recommendations(trauma_flags, overall_trauma_risk)
        }
    
    def assess_attachment_patterns(self, image: np.ndarray, psydraw_features: Dict, drawing_context: str) -> Dict:
        """Assess attachment patterns through drawing analysis"""
        print("👨‍👩‍👧‍👦 Assessing attachment patterns...")
        
        # Analyze family dynamics if family drawing
        if 'family' in drawing_context.lower():
            family_analysis = self._analyze_family_drawing_attachment(image, psydraw_features)
        else:
            family_analysis = self._infer_attachment_from_general_drawing(image, psydraw_features)
        
        # Assess attachment security indicators
        security_indicators = self._assess_attachment_security(image, psydraw_features)
        
        # Identify attachment style
        attachment_style = self._identify_attachment_style(family_analysis, security_indicators)
        
        # Generate attachment-based recommendations
        attachment_recommendations = self._generate_attachment_recommendations(attachment_style)
        
        return {
            'attachment_style': attachment_style.value,
            'attachment_security_score': security_indicators['overall_security'],
            'family_dynamics_analysis': family_analysis,
            'attachment_indicators': security_indicators,
            'attachment_recommendations': attachment_recommendations,
            'relationship_quality_assessment': self._assess_relationship_quality(family_analysis)
        }
    
    def _assess_fragmentation(self, image: np.ndarray, psydraw_features: Dict) -> float:
        """Assess psychological fragmentation indicators"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect broken lines and disconnected elements
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate fragmentation metrics
        total_contours = len(contours)
        small_fragments = len([c for c in contours if cv2.contourArea(c) < 100])
        fragmentation_ratio = small_fragments / total_contours if total_contours > 0 else 0
        
        # Analyze line continuity
        line_breaks = self._count_line_breaks(edges)
        continuity_score = 1.0 - (line_breaks / (total_contours + 1))
        
        # Combine metrics
        fragmentation_score = (fragmentation_ratio * 0.6 + (1 - continuity_score) * 0.4)
        return float(np.clip(fragmentation_score, 0, 1))
    
    def _assess_dissociation_markers(self, image: np.ndarray, psydraw_features: Dict) -> float:
        """Assess dissociation markers in the drawing"""
        # Look for floating elements (objects without ground connection)
        floating_score = self._detect_floating_elements(image)
        
        # Assess surreal or impossible combinations
        surreal_score = self._detect_surreal_elements(image, psydraw_features)
        
        # Check for lack of coherent narrative
        narrative_coherence = psydraw_features.get('temporal_features', {}).get('narrative_complexity', 0.5)
        coherence_score = 1.0 - narrative_coherence
        
        # Combine dissociation indicators
        dissociation_score = (floating_score * 0.4 + surreal_score * 0.4 + coherence_score * 0.2)
        return float(np.clip(dissociation_score, 0, 1))
    
    def _assess_hypervigilance_markers(self, image: np.ndarray, psydraw_features: Dict) -> float:
        """Assess hypervigilance markers"""
        # Placeholder implementation
        return 0.3
    
    def _assess_regression_patterns(self, psydraw_features: Dict, child_age: int) -> float:
        """Assess regression patterns"""
        # Placeholder implementation
        return 0.2
    
    def _assess_somatic_markers(self, image: np.ndarray, psydraw_features: Dict) -> float:
        """Assess somatic markers"""
        # Placeholder implementation
        return 0.25
    
    def _count_line_breaks(self, edges: np.ndarray) -> int:
        """Count line breaks in edge image"""
        # Simplified implementation
        return np.sum(edges > 0) // 100
    
    def _detect_floating_elements(self, image: np.ndarray) -> float:
        """Detect floating elements"""
        # Placeholder implementation
        return 0.3
    
    def _detect_surreal_elements(self, image: np.ndarray, psydraw_features: Dict) -> float:
        """Detect surreal elements"""
        # Placeholder implementation
        return 0.2
    
    def _categorize_trauma_risk(self, overall_risk: float) -> str:
        """Categorize trauma risk level"""
        if overall_risk > 0.7:
            return "high"
        elif overall_risk > 0.4:
            return "moderate"
        else:
            return "low"
    
    def _generate_trauma_recommendations(self, trauma_flags: List[ClinicalFlag], overall_risk: float) -> List[str]:
        """Generate trauma-informed recommendations"""
        recommendations = []
        
        if overall_risk > 0.7:
            recommendations.extend([
                "IMMEDIATE: Consult with trauma-informed mental health professional",
                "Implement trauma-sensitive approaches in all interactions",
                "Ensure child feels safe and supported",
                "Consider specialized trauma assessment (e.g., TSCC, TSCYC)"
            ])
        elif overall_risk > 0.4:
            recommendations.extend([
                "Monitor for additional trauma indicators",
                "Provide extra emotional support and validation",
                "Consider consultation with school counselor or psychologist",
                "Implement calming and grounding activities"
            ])
        else:
            recommendations.extend([
                "Continue supportive environment",
                "Monitor emotional well-being",
                "Encourage healthy expression through art"
            ])
        
        # Add specific recommendations based on trauma flags
        for flag in trauma_flags:
            if flag.indicator_type == "Fragmentation":
                recommendations.append("Focus on activities that promote integration and wholeness")
            elif flag.indicator_type == "Dissociation":
                recommendations.append("Implement grounding techniques and present-moment awareness")
            elif flag.indicator_type == "Hypervigilance":
                recommendations.append("Create predictable, safe environments")
        
        return recommendations
    
    def _analyze_family_drawing_attachment(self, image: np.ndarray, psydraw_features: Dict) -> Dict:
        """Analyze family drawing for attachment patterns"""
        # Placeholder implementation
        return {'family_cohesion': 0.6, 'power_dynamics': 'balanced'}
    
    def _infer_attachment_from_general_drawing(self, image: np.ndarray, psydraw_features: Dict) -> Dict:
        """Infer attachment from general drawing"""
        # Placeholder implementation
        return {'attachment_security': 0.7}
    
    def _assess_attachment_security(self, image: np.ndarray, psydraw_features: Dict) -> Dict:
        """Assess attachment security indicators"""
        # Placeholder implementation
        return {'overall_security': 0.65, 'trust_indicators': 0.7}
    
    def _identify_attachment_style(self, family_analysis: Dict, security_indicators: Dict) -> AttachmentStyle:
        """Identify attachment style"""
        # Simplified logic
        security_score = security_indicators.get('overall_security', 0.5)
        
        if security_score > 0.7:
            return AttachmentStyle.SECURE
        elif security_score > 0.5:
            return AttachmentStyle.ANXIOUS_AMBIVALENT
        elif security_score > 0.3:
            return AttachmentStyle.AVOIDANT
        else:
            return AttachmentStyle.DISORGANIZED
    
    def _generate_attachment_recommendations(self, attachment_style: AttachmentStyle) -> List[str]:
        """Generate attachment-based recommendations"""
        recommendations = {
            AttachmentStyle.SECURE: [
                "Continue fostering secure relationships",
                "Maintain consistent, responsive caregiving"
            ],
            AttachmentStyle.ANXIOUS_AMBIVALENT: [
                "Provide consistent, predictable responses",
                "Help develop emotional regulation skills"
            ],
            AttachmentStyle.AVOIDANT: [
                "Gradually build trust through reliable interactions",
                "Respect need for independence while offering support"
            ],
            AttachmentStyle.DISORGANIZED: [
                "Seek professional attachment-focused therapy",
                "Create highly structured, safe environment"
            ]
        }
        
        return recommendations.get(attachment_style, ["Monitor attachment development"])
    
    def _assess_relationship_quality(self, family_analysis: Dict) -> Dict:
        """Assess relationship quality"""
        # Placeholder implementation
        return {'overall_quality': 'good', 'areas_of_concern': []}
