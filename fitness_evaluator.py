import pandas as pd
import numpy as np
from datetime import datetime
import math

class FitnessEvaluator:
    def __init__(self, doctors_df, cabinets_df, appointments_df, revenue_df, demand_forecast, financial_metrics):
        self.doctors_df = doctors_df
        self.cabinets_df = cabinets_df
        self.appointments_df = appointments_df
        self.revenue_df = revenue_df
        self.demand_forecast = demand_forecast
        self.financial_metrics = financial_metrics
        
        # Pre-calculate lookup tables for performance
        self.doctor_lookup = self._create_doctor_lookup()
        self.cabinet_lookup = self._create_cabinet_lookup()
        self.service_costs = self._calculate_service_costs()
        
    def evaluate_fitness(self, chromosome, weights):
        """Evaluate fitness of a schedule chromosome"""
        
        if not chromosome:
            return 0.0
        
        # Calculate individual fitness components
        demand_score = self._evaluate_demand_coverage(chromosome) * weights.get('demand', 0.3)
        revenue_score = self._evaluate_revenue_potential(chromosome) * weights.get('revenue', 0.25)
        reliability_score = self._evaluate_reliability(chromosome) * weights.get('reliability', 0.2)
        strategy_score = self._evaluate_strategic_alignment(chromosome) * weights.get('strategy', 0.15)
        personnel_score = self._evaluate_personnel_balance(chromosome) * weights.get('personnel', 0.1)
        
        # Combine scores
        total_fitness = demand_score + revenue_score + reliability_score + strategy_score + personnel_score
        
        # Apply penalties for hard constraint violations
        penalty_factor = self._calculate_penalty_factor(chromosome)
        
        final_fitness = total_fitness * penalty_factor
        
        return max(0.0, final_fitness)  # Ensure non-negative fitness
    
    def _evaluate_demand_coverage(self, chromosome):
        """Evaluate how well the schedule covers predicted demand"""
        
        # Convert chromosome to schedule DataFrame
        schedule_df = pd.DataFrame(chromosome)
        
        if schedule_df.empty:
            return 0.0
        
        # Group by service and calculate coverage
        demand_coverage_scores = []
        
        for service in self.demand_forecast['service'].unique():
            predicted_demand = self.demand_forecast[
                self.demand_forecast['service'] == service
            ]['predicted_demand'].sum()
            
            # Calculate supplied capacity for this service
            service_assignments = schedule_df[
                schedule_df.get('service', '') == service
            ] if 'service' in schedule_df.columns else pd.DataFrame()
            
            supplied_capacity = len(service_assignments)
            
            # Calculate coverage ratio
            if predicted_demand > 0:
                coverage_ratio = min(1.0, supplied_capacity / predicted_demand)
                # Penalize both under and over-supply
                coverage_score = coverage_ratio - abs(coverage_ratio - 1.0) * 0.5
            else:
                coverage_score = 1.0 if supplied_capacity == 0 else 0.5
            
            demand_coverage_scores.append(max(0.0, coverage_score))
        
        return np.mean(demand_coverage_scores) if demand_coverage_scores else 0.0
    
    def _evaluate_revenue_potential(self, chromosome):
        """Evaluate predicted revenue generation potential"""
        
        schedule_df = pd.DataFrame(chromosome)
        
        if schedule_df.empty:
            return 0.0
        
        total_revenue_potential = 0.0
        
        for _, assignment in schedule_df.iterrows():
            doctor_id = assignment.get('doctor_id')
            service = assignment.get('service', '')
            is_dms = assignment.get('is_dms', False)
            
            # Get doctor's historical performance
            doctor_metrics = self.financial_metrics[
                self.financial_metrics['doctor_id'] == doctor_id
            ]
            
            if not doctor_metrics.empty:
                avg_appointment_cost = doctor_metrics.iloc[0]['avg_appointment_cost']
                fill_rate = doctor_metrics.iloc[0]['fill_rate']
            else:
                avg_appointment_cost = self.service_costs.get(service, 1000)  # Default cost
                fill_rate = 0.7  # Default fill rate
            
            # Calculate expected revenue for this assignment
            base_revenue = avg_appointment_cost * fill_rate
            
            # DMS bonus
            if is_dms:
                base_revenue *= 1.2  # 20% bonus for DMS
            
            total_revenue_potential += base_revenue
        
        # Normalize by theoretical maximum (all slots filled at highest rate)
        max_theoretical_revenue = len(chromosome) * max(self.service_costs.values(), default=2000)
        
        return min(1.0, total_revenue_potential / max_theoretical_revenue) if max_theoretical_revenue > 0 else 0.0
    
    def _evaluate_reliability(self, chromosome):
        """Evaluate reliability based on doctor performance history"""
        
        schedule_df = pd.DataFrame(chromosome)
        
        if schedule_df.empty:
            return 0.0
        
        reliability_scores = []
        
        for doctor_id in schedule_df['doctor_id'].unique():
            doctor_metrics = self.financial_metrics[
                self.financial_metrics['doctor_id'] == doctor_id
            ]
            
            if not doctor_metrics.empty:
                reliability_coef = doctor_metrics.iloc[0].get('reliability_coefficient', 0.5)
                fill_rate = doctor_metrics.iloc[0].get('fill_rate', 0.5)
                
                # Combine reliability coefficient and fill rate
                doctor_reliability = (reliability_coef * 0.6 + fill_rate * 0.4)
            else:
                # New doctor - give average score
                doctor_reliability = 0.5
            
            # Weight by number of assignments
            doctor_assignments = len(schedule_df[schedule_df['doctor_id'] == doctor_id])
            reliability_scores.extend([doctor_reliability] * doctor_assignments)
        
        return np.mean(reliability_scores) if reliability_scores else 0.0
    
    def _evaluate_strategic_alignment(self, chromosome):
        """Evaluate strategic goals (DMS, house calls, sick leave capability)"""
        
        schedule_df = pd.DataFrame(chromosome)
        
        if schedule_df.empty:
            return 0.0
        
        strategic_scores = []
        
        for _, assignment in schedule_df.iterrows():
            doctor_id = assignment.get('doctor_id')
            is_dms = assignment.get('is_dms', False)
            
            doctor_info = self.doctor_lookup.get(doctor_id, {})
            
            score = 0.0
            
            # DMS capability bonus
            if is_dms and doctor_info.get('dms_enabled', False):
                score += 0.4
            
            # House calls capability
            if doctor_info.get('house_calls', False):
                score += 0.2
            
            # Sick leave capability
            if doctor_info.get('sick_leave_enabled', False):
                score += 0.2
            
            # Service diversity bonus (doctors who can handle multiple services)
            doctor_metrics = self.financial_metrics[
                self.financial_metrics['doctor_id'] == doctor_id
            ]
            
            if not doctor_metrics.empty:
                service_diversity = doctor_metrics.iloc[0].get('service_diversity', 1)
                diversity_bonus = min(0.2, service_diversity / 10)  # Cap at 0.2
                score += diversity_bonus
            
            strategic_scores.append(min(1.0, score))
        
        return np.mean(strategic_scores) if strategic_scores else 0.0
    
    def _evaluate_personnel_balance(self, chromosome):
        """Evaluate personnel management factors"""
        
        schedule_df = pd.DataFrame(chromosome)
        
        if schedule_df.empty:
            return 0.0
        
        # Calculate workload distribution
        doctor_workloads = schedule_df['doctor_id'].value_counts()
        
        # Calculate balance metrics
        workload_std = doctor_workloads.std()
        workload_mean = doctor_workloads.mean()
        
        # Balance score (lower std deviation = better balance)
        balance_score = 1.0 / (1.0 + workload_std / workload_mean) if workload_mean > 0 else 0.0
        
        # New doctor preference (give bonus to new doctors)
        new_doctor_bonus = 0.0
        total_assignments = len(chromosome)
        
        for doctor_id in doctor_workloads.index:
            doctor_info = self.doctor_lookup.get(doctor_id, {})
            experience_years = doctor_info.get('experience_years', 0)
            
            if experience_years < 2:  # New doctor
                doctor_assignments = doctor_workloads[doctor_id]
                new_doctor_bonus += (doctor_assignments / total_assignments) * 0.3
        
        # Soft cabinet preferences
        preference_score = self._evaluate_cabinet_preferences(chromosome)
        
        # Combine personnel factors
        personnel_score = (balance_score * 0.5 + new_doctor_bonus * 0.3 + preference_score * 0.2)
        
        return min(1.0, personnel_score)
    
    def _evaluate_cabinet_preferences(self, chromosome):
        """Evaluate soft cabinet preferences"""
        
        preference_scores = []
        
        for gene in chromosome:
            doctor_id = gene.get('doctor_id')
            cabinet_id = gene.get('cabinet_id')
            
            doctor_info = self.doctor_lookup.get(doctor_id, {})
            preferred_cabinet = doctor_info.get('preferred_cabinet')
            
            if preferred_cabinet and preferred_cabinet == cabinet_id:
                preference_scores.append(1.0)
            else:
                preference_scores.append(0.5)  # Neutral score
        
        return np.mean(preference_scores) if preference_scores else 0.5
    
    def _calculate_penalty_factor(self, chromosome):
        """Calculate penalty factor for hard constraint violations"""
        
        penalty_factor = 1.0
        
        # Check for time conflicts
        time_conflicts = self._detect_time_conflicts(chromosome)
        if time_conflicts > 0:
            penalty_factor *= (1.0 - min(0.5, time_conflicts * 0.1))
        
        # Check for specialization violations
        specialization_violations = self._detect_specialization_violations(chromosome)
        if specialization_violations > 0:
            penalty_factor *= (1.0 - min(0.3, specialization_violations * 0.05))
        
        # Check for shift violations
        shift_violations = self._detect_shift_violations(chromosome)
        if shift_violations > 0:
            penalty_factor *= (1.0 - min(0.2, shift_violations * 0.03))
        
        return max(0.1, penalty_factor)  # Never go below 10% of original fitness
    
    def _detect_time_conflicts(self, chromosome):
        """Detect scheduling conflicts (same cabinet, same time)"""
        
        conflicts = 0
        time_slots = {}
        
        for gene in chromosome:
            day = gene.get('day')
            cabinet_id = gene.get('cabinet_id')
            start_time = gene.get('start_time')
            
            if day and cabinet_id and start_time:
                slot_key = f"{day}_{cabinet_id}_{start_time}"
                
                if slot_key in time_slots:
                    conflicts += 1
                else:
                    time_slots[slot_key] = gene
        
        return conflicts
    
    def _detect_specialization_violations(self, chromosome):
        """Detect specialization constraint violations"""
        
        violations = 0
        
        for gene in chromosome:
            doctor_id = gene.get('doctor_id')
            service = gene.get('service', '')
            
            doctor_info = self.doctor_lookup.get(doctor_id, {})
            doctor_specialty = doctor_info.get('specialty', '').lower()
            
            # Check if doctor can provide this service
            if not self._is_service_compatible(doctor_specialty, service):
                violations += 1
        
        return violations
    
    def _detect_shift_violations(self, chromosome):
        """Detect shift constraint violations"""
        
        violations = 0
        
        for gene in chromosome:
            doctor_id = gene.get('doctor_id')
            start_time = gene.get('start_time', '')
            
            doctor_info = self.doctor_lookup.get(doctor_id, {})
            preferred_shift = doctor_info.get('shift_type', 'day')
            
            if start_time and not self._is_time_in_shift(start_time, preferred_shift):
                violations += 1
        
        return violations
    
    def _create_doctor_lookup(self):
        """Create fast lookup dictionary for doctor information"""
        
        lookup = {}
        for _, doctor in self.doctors_df.iterrows():
            lookup[doctor['doctor_id']] = doctor.to_dict()
        return lookup
    
    def _create_cabinet_lookup(self):
        """Create fast lookup dictionary for cabinet information"""
        
        lookup = {}
        for _, cabinet in self.cabinets_df.iterrows():
            lookup[cabinet['cabinet_id']] = cabinet.to_dict()
        return lookup
    
    def _calculate_service_costs(self):
        """Calculate average costs by service type"""
        
        if self.appointments_df.empty:
            return {'default': 1000}
        
        service_costs = self.appointments_df.groupby('service_name')['cost'].mean().to_dict()
        
        # Add default cost
        service_costs['default'] = np.mean(list(service_costs.values())) if service_costs else 1000
        
        return service_costs
    
    def _is_service_compatible(self, doctor_specialty, service):
        """Check if doctor specialty is compatible with service"""
        
        # Simplified compatibility check
        service_lower = service.lower()
        
        specialty_mappings = {
            'терапевт': ['терапевт', 'общий', 'консультация'],
            'кардиолог': ['кардиолог', 'сердце', 'экг'],
            'педиатр': ['педиатр', 'детский', 'ребенок'],
            'гинеколог': ['гинеколог', 'женский'],
            'невролог': ['невролог', 'неврология']
        }
        
        if doctor_specialty in specialty_mappings:
            return any(keyword in service_lower for keyword in specialty_mappings[doctor_specialty])
        
        # Default: allow assignment (flexible)
        return True
    
    def _is_time_in_shift(self, time_str, shift_type):
        """Check if time falls within doctor's preferred shift"""
        
        try:
            hour = int(time_str.split(':')[0])
            
            if shift_type == 'morning' and hour < 14:
                return True
            elif shift_type == 'evening' and hour >= 14:
                return True
            elif shift_type == 'day' and 9 <= hour < 18:
                return True
            elif shift_type == 'night' and (hour >= 20 or hour < 8):
                return True
        except:
            pass
        
        return False
    
    def get_fitness_breakdown(self, chromosome, weights):
        """Get detailed breakdown of fitness components"""
        
        demand_score = self._evaluate_demand_coverage(chromosome)
        revenue_score = self._evaluate_revenue_potential(chromosome)
        reliability_score = self._evaluate_reliability(chromosome)
        strategy_score = self._evaluate_strategic_alignment(chromosome)
        personnel_score = self._evaluate_personnel_balance(chromosome)
        
        penalty_factor = self._calculate_penalty_factor(chromosome)
        
        weighted_scores = {
            'demand': demand_score * weights.get('demand', 0.3),
            'revenue': revenue_score * weights.get('revenue', 0.25),
            'reliability': reliability_score * weights.get('reliability', 0.2),
            'strategy': strategy_score * weights.get('strategy', 0.15),
            'personnel': personnel_score * weights.get('personnel', 0.1)
        }
        
        total_before_penalty = sum(weighted_scores.values())
        final_fitness = total_before_penalty * penalty_factor
        
        return {
            'raw_scores': {
                'demand': demand_score,
                'revenue': revenue_score,
                'reliability': reliability_score,
                'strategy': strategy_score,
                'personnel': personnel_score
            },
            'weighted_scores': weighted_scores,
            'penalty_factor': penalty_factor,
            'total_before_penalty': total_before_penalty,
            'final_fitness': final_fitness,
            'violations': {
                'time_conflicts': self._detect_time_conflicts(chromosome),
                'specialization_violations': self._detect_specialization_violations(chromosome),
                'shift_violations': self._detect_shift_violations(chromosome)
            }
        }
