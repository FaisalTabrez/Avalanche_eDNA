"""
Environmental Context Analysis Module

This module provides comprehensive environmental context analysis for eDNA samples,
including habitat classification, environmental clustering, and ecological insights.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class EnvironmentalContextAnalyzer:
    """Analyze environmental context and ecological patterns in eDNA data."""
    
    def __init__(self):
        """Initialize environmental context analyzer."""
        self.habitat_classifiers = {
            'marine': {'salinity': (30, 40), 'depth': (0, 11000), 'temperature': (-2, 35)},
            'freshwater': {'salinity': (0, 0.5), 'depth': (0, 1000), 'temperature': (0, 40)},
            'brackish': {'salinity': (0.5, 30), 'depth': (0, 100), 'temperature': (5, 35)},
            'terrestrial': {'salinity': (0, 0.1), 'depth': (0, 10), 'temperature': (-20, 50)},
            'extreme': {'temperature': (60, 100), 'ph': (0, 3)}  # Hot springs, acidic environments
        }
    
    def analyze_environmental_context(self, environmental_data: Dict[str, Any], 
                                    taxonomic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive environmental context analysis.
        
        Args:
            environmental_data: Environmental metadata
            taxonomic_data: Taxonomic composition data
            
        Returns:
            Comprehensive environmental analysis results
        """
        results = {
            'habitat_classification': {},
            'environmental_gradients': {},
            'ecological_indicators': {},
            'habitat_specificity': {},
            'environmental_stress_indicators': {},
            'seasonal_patterns': {},
            'depth_stratification': {},
            'biogeochemical_indicators': {},
            'conservation_insights': {},
            'sampling_recommendations': {}
        }
        
        # Classify habitat type
        results['habitat_classification'] = self._classify_habitat(environmental_data)
        
        # Analyze environmental gradients
        results['environmental_gradients'] = self._analyze_gradients(environmental_data)
        
        # Identify ecological indicators
        results['ecological_indicators'] = self._identify_ecological_indicators(
            environmental_data, taxonomic_data
        )
        
        # Analyze habitat specificity
        results['habitat_specificity'] = self._analyze_habitat_specificity(
            environmental_data, taxonomic_data
        )
        
        # Detect environmental stress
        results['environmental_stress_indicators'] = self._detect_environmental_stress(
            environmental_data, taxonomic_data
        )
        
        # Analyze seasonal patterns
        results['seasonal_patterns'] = self._analyze_seasonal_patterns(environmental_data)
        
        # Depth stratification analysis
        results['depth_stratification'] = self._analyze_depth_stratification(
            environmental_data, taxonomic_data
        )
        
        # Biogeochemical indicators
        results['biogeochemical_indicators'] = self._analyze_biogeochemical_context(
            environmental_data, taxonomic_data
        )
        
        # Conservation insights
        results['conservation_insights'] = self._generate_conservation_insights(
            environmental_data, taxonomic_data
        )
        
        # Sampling recommendations
        results['sampling_recommendations'] = self._generate_sampling_recommendations(
            environmental_data, results
        )
        
        return results
    
    def _classify_habitat(self, env_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify habitat type based on environmental parameters."""
        habitat_scores = {}
        
        for habitat, criteria in self.habitat_classifiers.items():
            score = 0
            max_score = 0
            
            for param, (min_val, max_val) in criteria.items():
                if param in env_data and env_data[param] is not None:
                    try:
                        value = float(env_data[param])  # Ensure numeric value
                        if min_val <= value <= max_val:
                            score += 1
                        max_score += 1
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        continue
            
            if max_score > 0:
                habitat_scores[habitat] = score / max_score
        
        # Determine primary habitat
        primary_habitat = max(habitat_scores.keys(), key=lambda k: habitat_scores[k]) if habitat_scores else 'unknown'
        
        return {
            'primary_habitat': primary_habitat,
            'habitat_scores': habitat_scores,
            'confidence': max(habitat_scores.values()) if habitat_scores else 0,
            'habitat_complexity': self._assess_habitat_complexity(env_data, primary_habitat),
            'habitat_description': self._get_habitat_description(primary_habitat, env_data)
        }
    
    def _get_habitat_description(self, habitat: str, env_data: Dict[str, Any]) -> str:
        """Generate detailed habitat description."""
        descriptions = {
            'marine': f"Marine environment at {env_data.get('depth', 'unknown')}m depth",
            'freshwater': f"Freshwater system with {env_data.get('temperature', 'unknown')}°C",
            'brackish': f"Brackish water with salinity {env_data.get('salinity', 'unknown')}ppt",
            'terrestrial': f"Terrestrial environment at {env_data.get('location', 'unknown location')}",
            'extreme': f"Extreme environment with extreme conditions"
        }
        
        base_desc = descriptions.get(habitat, "Unknown habitat type")
        
        # Add environmental qualifiers
        qualifiers = []
        temp = env_data.get('temperature')
        if temp is not None:
            if temp < 5:
                qualifiers.append("cold")
            elif temp > 25:
                qualifiers.append("warm")
        
        depth = env_data.get('depth')
        if depth is not None:
            if depth > 1000:
                qualifiers.append("deep")
            elif depth < 10:
                qualifiers.append("shallow")
        
        if qualifiers:
            base_desc += f" ({', '.join(qualifiers)})"
        
        return base_desc
    
    def _assess_habitat_complexity(self, env_data: Dict[str, Any], habitat: str) -> str:
        """Assess habitat complexity level."""
        complexity_factors = 0
        
        # Check for multiple environmental gradients
        if env_data.get('temperature') is not None:
            complexity_factors += 1
        if env_data.get('depth') is not None:
            complexity_factors += 1
        if env_data.get('ph') is not None:
            complexity_factors += 1
        if env_data.get('salinity') is not None:
            complexity_factors += 1
        
        if complexity_factors >= 3:
            return 'high'
        elif complexity_factors >= 2:
            return 'moderate'
        else:
            return 'low'
    
    def _analyze_gradients(self, env_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze environmental gradients and their implications."""
        gradients = {}
        
        # Temperature gradient analysis
        temp = env_data.get('temperature')
        if temp is not None:
            gradients['temperature'] = {
                'value': temp,
                'classification': self._classify_temperature(temp),
                'biological_implications': self._get_temperature_implications(temp),
                'metabolic_predictions': self._predict_metabolic_activity(temp)
            }
        
        # Depth gradient analysis
        depth = env_data.get('depth')
        if depth is not None:
            gradients['depth'] = {
                'value': depth,
                'zone': self._classify_depth_zone(depth),
                'pressure_mpa': depth * 0.1,  # Approximate pressure
                'light_availability': self._estimate_light_availability(depth),
                'expected_adaptations': self._predict_depth_adaptations(depth)
            }
        
        # pH gradient analysis
        ph = env_data.get('ph')
        if ph is not None:
            gradients['ph'] = {
                'value': ph,
                'classification': self._classify_ph(ph),
                'buffer_capacity': self._estimate_buffer_capacity(ph),
                'microbial_implications': self._get_ph_implications(ph)
            }
        
        # Salinity gradient analysis
        salinity = env_data.get('salinity')
        if salinity is not None:
            gradients['salinity'] = {
                'value': salinity,
                'classification': self._classify_salinity(salinity),
                'osmotic_stress': self._calculate_osmotic_stress(salinity),
                'adaptation_requirements': self._get_salinity_adaptations(salinity)
            }
        
        return gradients
    
    def _identify_ecological_indicators(self, env_data: Dict[str, Any], 
                                      tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify ecological indicator species and patterns."""
        indicators = {
            'pollution_indicators': [],
            'pristine_environment_indicators': [],
            'climate_change_indicators': [],
            'ecosystem_health_score': 0.0,
            'keystone_species_candidates': [],
            'invasive_species_risk': [],
            'endemic_species_potential': []
        }
        
        # Simulate pollution indicators based on environmental conditions
        temp = env_data.get('temperature')
        if temp is not None and temp > 30:
            indicators['pollution_indicators'].append({
                'indicator': 'thermal_pollution',
                'severity': 'moderate',
                'description': 'Elevated temperature may indicate thermal pollution'
            })
        
        # Pristine environment indicators
        ph = env_data.get('ph')
        depth = env_data.get('depth')
        if (ph is not None and ph > 7.5 and 
            depth is not None and depth > 100):
            indicators['pristine_environment_indicators'].append({
                'indicator': 'deep_alkaline_conditions',
                'confidence': 0.8,
                'description': 'Deep alkaline conditions suggest minimal human impact'
            })
        
        # Climate change indicators
        temp = env_data.get('temperature')
        if temp is not None and temp > 25:
            indicators['climate_change_indicators'].append({
                'indicator': 'ocean_warming',
                'trend': 'increasing',
                'description': 'Temperature suggests warming trend'
            })
        
        # Calculate ecosystem health score
        health_factors = []
        
        # Factor in diversity
        if isinstance(tax_data, dict) and 'alpha_diversity' in tax_data:
            alpha_div = tax_data['alpha_diversity']
            if isinstance(alpha_div, dict):
                shannon = alpha_div.get('genus_shannon', 0)
                health_factors.append(min(shannon / 4.0, 1.0))  # Normalize to 0-1
        
        # Factor in environmental conditions
        ph = env_data.get('ph')
        if ph is not None:
            ph_health = 1.0 - abs(ph - 8.0) / 6.0  # Optimal around pH 8
            health_factors.append(max(0, ph_health))
        
        indicators['ecosystem_health_score'] = np.mean(health_factors) if health_factors else 0.5
        
        return indicators
    
    def _analyze_habitat_specificity(self, env_data: Dict[str, Any], 
                                   tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze habitat specificity of detected taxa."""
        specificity = {
            'habitat_specialists': [],
            'habitat_generalists': [],
            'endemic_potential': [],
            'cosmopolitan_taxa': [],
            'habitat_associations': {}
        }
        
        # Simulate habitat specificity analysis
        habitat_classification = env_data.get('habitat_classification', {})
        if isinstance(habitat_classification, dict):
            habitat = habitat_classification.get('primary_habitat', 'unknown')
        else:
            habitat = 'unknown'
        
        # Example specialist identification based on habitat
        if habitat == 'marine':
            specificity['habitat_specialists'] = [
                {'taxon': 'Marine_Proteobacteria_sp1', 'specificity_score': 0.95},
                {'taxon': 'Deep_sea_Archaea_sp2', 'specificity_score': 0.88}
            ]
        elif habitat == 'freshwater':
            specificity['habitat_specialists'] = [
                {'taxon': 'Freshwater_Actinobacteria_sp1', 'specificity_score': 0.92},
                {'taxon': 'Lake_Cyanobacteria_sp3', 'specificity_score': 0.85}
            ]
        
        # Generalists (found across habitats)
        specificity['habitat_generalists'] = [
            {'taxon': 'Proteobacteria_general', 'habitat_breadth': 0.8},
            {'taxon': 'Common_Firmicutes', 'habitat_breadth': 0.7}
        ]
        
        return specificity
    
    def _detect_environmental_stress(self, env_data: Dict[str, Any], 
                                   tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect environmental stress indicators."""
        stress_indicators = {
            'thermal_stress': False,
            'chemical_stress': False,
            'osmotic_stress': False,
            'oxygen_stress': False,
            'overall_stress_level': 'low',
            'stress_response_taxa': [],
            'stress_tolerant_indicators': []
        }
        
        # Thermal stress
        temp = env_data.get('temperature')
        if temp is not None and (temp > 35 or temp < 0):
            stress_indicators['thermal_stress'] = True
            stress_indicators['stress_response_taxa'].append({
                'stressor': 'extreme_temperature',
                'expected_taxa': ['Thermophiles', 'Psychrophiles'],
                'stress_level': 'high' if abs(temp - 20) > 30 else 'moderate'
            })
        
        # Chemical stress (pH)
        ph = env_data.get('ph')
        if ph is not None and (ph < 5 or ph > 9):
            stress_indicators['chemical_stress'] = True
            stress_indicators['stress_response_taxa'].append({
                'stressor': 'extreme_pH',
                'expected_taxa': ['Acidophiles', 'Alkaliphiles'],
                'stress_level': 'high' if abs(ph - 7) > 3 else 'moderate'
            })
        
        # Osmotic stress
        salinity = env_data.get('salinity')
        if salinity is not None and (salinity > 50 or salinity < 0.1):
            stress_indicators['osmotic_stress'] = True
            stress_indicators['stress_response_taxa'].append({
                'stressor': 'extreme_salinity',
                'expected_taxa': ['Halophiles', 'Halotolerant_bacteria'],
                'stress_level': 'high' if salinity > 80 or salinity < 0.01 else 'moderate'
            })
        
        # Calculate overall stress level
        stress_count = sum([
            stress_indicators['thermal_stress'],
            stress_indicators['chemical_stress'],
            stress_indicators['osmotic_stress']
        ])
        
        if stress_count >= 2:
            stress_indicators['overall_stress_level'] = 'high'
        elif stress_count == 1:
            stress_indicators['overall_stress_level'] = 'moderate'
        
        return stress_indicators
    
    def _analyze_seasonal_patterns(self, env_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze seasonal patterns and temporal dynamics."""
        seasonal = {
            'sampling_season': 'unknown',
            'seasonal_indicators': [],
            'temporal_recommendations': [],
            'expected_seasonal_variation': {}
        }
        
        # Estimate season from temperature and date
        temp = env_data.get('temperature')
        collection_date = env_data.get('collection_date')
        
        if collection_date:
            # Extract month from date
            if isinstance(collection_date, str):
                try:
                    date_obj = datetime.fromisoformat(collection_date)
                    month = date_obj.month
                except:
                    month = 6  # Default to summer
            else:
                month = 6
            
            # Estimate season (Northern Hemisphere)
            if month in [12, 1, 2]:
                season = 'winter'
            elif month in [3, 4, 5]:
                season = 'spring'
            elif month in [6, 7, 8]:
                season = 'summer'
            else:
                season = 'autumn'
            
            seasonal['sampling_season'] = season
            
            # Season-specific insights
            seasonal_insights = {
                'winter': {'expected_diversity': 'low', 'dominant_groups': ['Cold-adapted bacteria']},
                'spring': {'expected_diversity': 'increasing', 'dominant_groups': ['Bloom-forming algae']},
                'summer': {'expected_diversity': 'high', 'dominant_groups': ['Thermophilic bacteria']},
                'autumn': {'expected_diversity': 'moderate', 'dominant_groups': ['Decomposer bacteria']}
            }
            
            seasonal['expected_seasonal_variation'] = seasonal_insights.get(season, {})
        
        return seasonal
    
    def _analyze_depth_stratification(self, env_data: Dict[str, Any], 
                                    tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze depth-related stratification patterns."""
        depth = env_data.get('depth')
        
        # Default stratification for unknown depth
        stratification = {
            'depth_zone': 'unknown',
            'pressure_adaptations': [],
            'light_dependent_processes': {},
            'depth_specific_taxa': [],
            'vertical_migration_indicators': []
        }
        
        if depth is None:
            return stratification
        
        # Update with actual depth analysis
        stratification['depth_zone'] = self._classify_depth_zone(depth)
        
        # Depth zone classification
        if depth < 10:
            stratification['depth_zone'] = 'surface'
            stratification['light_dependent_processes'] = {
                'photosynthesis': 'high_potential',
                'primary_productivity': 'expected',
                'diel_vertical_migration': 'possible'
            }
        elif depth < 200:
            stratification['depth_zone'] = 'euphotic'
            stratification['light_dependent_processes'] = {
                'photosynthesis': 'moderate_potential',
                'primary_productivity': 'limited'
            }
        elif depth < 1000:
            stratification['depth_zone'] = 'dysphotic'
            stratification['pressure_adaptations'] = ['Piezotolerant_bacteria']
        else:
            stratification['depth_zone'] = 'aphotic'
            stratification['pressure_adaptations'] = ['Piezophilic_bacteria', 'Barophiles']
            stratification['depth_specific_taxa'] = ['Deep_sea_specialists', 'Pressure_adapted_archaea']
        
        return stratification
    
    def _analyze_biogeochemical_context(self, env_data: Dict[str, Any], 
                                      tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze biogeochemical cycling indicators."""
        biogeochem = {
            'carbon_cycling': {},
            'nitrogen_cycling': {},
            'sulfur_cycling': {},
            'metal_cycling': {},
            'biogeochemical_hotspots': [],
            'nutrient_limitations': []
        }
        
        # Carbon cycling indicators
        temp = env_data.get('temperature')
        depth = env_data.get('depth')
        
        if depth is not None and depth < 100:
            biogeochem['carbon_cycling'] = {
                'primary_production': 'likely',
                'co2_fixation': 'photosynthetic',
                'organic_matter_source': 'autochthonous'
            }
        else:
            biogeochem['carbon_cycling'] = {
                'primary_production': 'chemosynthetic',
                'co2_fixation': 'chemolithotrophic',
                'organic_matter_source': 'allochthonous'
            }
        
        # Nitrogen cycling
        ph = env_data.get('ph')
        if ph is not None and ph > 7.5:
            biogeochem['nitrogen_cycling'] = {
                'nitrification': 'favorable',
                'denitrification': 'possible',
                'nitrogen_fixation': 'potential'
            }
        
        return biogeochem
    
    def _generate_conservation_insights(self, env_data: Dict[str, Any], 
                                      tax_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conservation-relevant insights."""
        conservation = {
            'conservation_priority': 'medium',
            'threats_identified': [],
            'unique_features': [],
            'monitoring_recommendations': [],
            'protection_strategies': []
        }
        
        # Assess conservation priority based on diversity and uniqueness
        if isinstance(tax_data, dict) and 'alpha_diversity' in tax_data:
            alpha_div = tax_data['alpha_diversity']
            if isinstance(alpha_div, dict):
                genus_shannon = alpha_div.get('genus_shannon', 0)
                if genus_shannon > 3:
                    conservation['conservation_priority'] = 'high'
                    conservation['unique_features'].append('high_biodiversity')
        
        # Environmental uniqueness
        depth = env_data.get('depth')
        temp = env_data.get('temperature')
        
        if depth is not None and depth > 2000:
            conservation['unique_features'].append('deep_sea_ecosystem')
            conservation['protection_strategies'].append('deep_sea_protection_zones')
        
        if temp is not None and (temp > 40 or temp < 5):
            conservation['unique_features'].append('extreme_environment')
            conservation['monitoring_recommendations'].append('climate_change_monitoring')
        
        return conservation
    
    def _generate_sampling_recommendations(self, env_data: Dict[str, Any], 
                                         analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for future sampling."""
        recommendations = {
            'optimal_sampling_times': [],
            'additional_parameters': [],
            'sampling_frequency': 'quarterly',
            'spatial_recommendations': [],
            'methodological_improvements': []
        }
        
        # Seasonal recommendations
        if analysis_results.get('seasonal_patterns', {}).get('sampling_season') == 'winter':
            recommendations['optimal_sampling_times'] = ['spring', 'summer']
        
        # Parameter recommendations
        if 'temperature' not in env_data:
            recommendations['additional_parameters'].append('temperature')
        if 'salinity' not in env_data:
            recommendations['additional_parameters'].append('salinity')
        if 'ph' not in env_data:
            recommendations['additional_parameters'].append('pH')
        
        # Spatial recommendations
        depth = env_data.get('depth')
        if depth is not None and depth < 50:
            recommendations['spatial_recommendations'] = [
                'sample_at_multiple_depths',
                'include_benthic_samples',
                'consider_tidal_effects'
            ]
        
        return recommendations
    
    # Helper methods for classification
    def _classify_temperature(self, temp: float) -> str:
        """Classify temperature range."""
        if temp is None:
            return 'unknown'
        
        if temp < 0:
            return 'freezing'
        elif temp < 15:
            return 'cold'
        elif temp < 25:
            return 'moderate'
        elif temp < 35:
            return 'warm'
        else:
            return 'hot'
    
    def _classify_depth_zone(self, depth: float) -> str:
        """Classify depth zone."""
        if depth is None:
            return 'unknown'
        
        if depth < 10:
            return 'surface'
        elif depth < 200:
            return 'euphotic'
        elif depth < 1000:
            return 'dysphotic'
        elif depth < 4000:
            return 'bathypelagic'
        else:
            return 'abyssopelagic'
    
    def _classify_ph(self, ph: float) -> str:
        """Classify pH range."""
        if ph is None:
            return 'unknown'
        
        if ph < 3:
            return 'highly_acidic'
        elif ph < 6:
            return 'acidic'
        elif ph < 8:
            return 'neutral'
        elif ph < 10:
            return 'alkaline'
        else:
            return 'highly_alkaline'
    
    def _classify_salinity(self, salinity: float) -> str:
        """Classify salinity range."""
        if salinity is None:
            return 'unknown'
        
        if salinity < 0.5:
            return 'freshwater'
        elif salinity < 5:
            return 'oligohaline'
        elif salinity < 18:
            return 'mesohaline'
        elif salinity < 30:
            return 'polyhaline'
        else:
            return 'euhaline'
    
    def _get_temperature_implications(self, temp: float) -> List[str]:
        """Get biological implications of temperature."""
        implications = []
        if temp is None:
            return implications
        
        if temp < 5:
            implications.extend(['psychrophilic_adaptations', 'slow_metabolism', 'cold_shock_proteins'])
        elif temp > 30:
            implications.extend(['thermophilic_adaptations', 'heat_shock_response', 'protein_stability'])
        return implications
    
    def _predict_metabolic_activity(self, temp: float) -> str:
        """Predict metabolic activity level based on temperature."""
        if temp is None:
            return 'unknown'
        
        if temp < 5:
            return 'low'
        elif temp < 15:
            return 'moderate'
        elif temp < 30:
            return 'high'
        else:
            return 'extreme'
    
    def _estimate_light_availability(self, depth: float) -> str:
        """Estimate light availability at depth."""
        if depth is None:
            return 'unknown'
        
        if depth < 10:
            return 'high'
        elif depth < 100:
            return 'moderate'
        elif depth < 200:
            return 'low'
        else:
            return 'none'
    
    def _predict_depth_adaptations(self, depth: float) -> List[str]:
        """Predict adaptations needed for depth."""
        adaptations = []
        if depth is None:
            return adaptations
        
        if depth > 100:
            adaptations.append('pressure_tolerance')
        if depth > 200:
            adaptations.extend(['reduced_light_dependency', 'chemosynthesis'])
        if depth > 1000:
            adaptations.extend(['piezophilic_adaptations', 'specialized_membrane_composition'])
        return adaptations
    
    def _estimate_buffer_capacity(self, ph: float) -> str:
        """Estimate buffer capacity."""
        if ph is None:
            return 'unknown'
        
        if 6.5 <= ph <= 8.5:
            return 'high'
        elif 5.5 <= ph <= 9.5:
            return 'moderate'
        else:
            return 'low'
    
    def _get_ph_implications(self, ph: float) -> List[str]:
        """Get microbial implications of pH."""
        implications = []
        if ph is None:
            return implications
        
        if ph < 5:
            implications.extend(['acid_tolerance', 'acid_stress_response', 'specialized_pH_homeostasis'])
        elif ph > 9:
            implications.extend(['alkaline_tolerance', 'base_stress_response', 'alkaline_adaptations'])
        return implications
    
    def _calculate_osmotic_stress(self, salinity: float) -> str:
        """Calculate osmotic stress level."""
        if salinity is None:
            return 'unknown'
        
        if salinity < 0.5 or salinity > 40:
            return 'high'
        elif salinity < 5 or salinity > 35:
            return 'moderate'
        else:
            return 'low'
    
    def _get_salinity_adaptations(self, salinity: float) -> List[str]:
        """Get adaptations required for salinity level."""
        adaptations = []
        if salinity is None:
            return adaptations
        
        if salinity > 30:
            adaptations.extend(['osmoregulation', 'compatible_solutes', 'salt_tolerance'])
        elif salinity < 5:
            adaptations.extend(['freshwater_adaptations', 'ion_regulation'])
        return adaptations