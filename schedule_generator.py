import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from itertools import product

class ScheduleGenerator:
    def __init__(self, doctors_df, cabinets_df, demand_forecast):
        self.doctors_df = doctors_df
        self.cabinets_df = cabinets_df
        self.demand_forecast = demand_forecast
        
        # Pre-process compatibility matrices
        self.doctor_cabinet_compatibility = self._build_compatibility_matrix()
        self.shift_definitions = {
            'morning': {'start': '08:00', 'end': '14:00'},
            'evening': {'start': '14:00', 'end': '20:00'},
            'day': {'start': '09:00', 'end': '18:00'},
            'night': {'start': '20:00', 'end': '08:00'}
        }
    
    def generate_population(self, population_size, target_month, enforce_shifts=True, 
                          enforce_specializations=True, enforce_star_schedules=True, 
                          enforce_cabinet_bindings=True):
        """Generate a population of valid schedules (chromosomes)"""
        
        population = []
        attempts = 0
        max_attempts = population_size * 10  # Prevent infinite loops
        
        # Get working days for the target month
        working_days = self._get_working_days(target_month)
        
        while len(population) < population_size and attempts < max_attempts:
            attempts += 1
            
            try:
                chromosome = self._generate_single_schedule(
                    working_days, enforce_shifts, enforce_specializations,
                    enforce_star_schedules, enforce_cabinet_bindings
                )
                
                if self._is_valid_schedule(chromosome):
                    population.append(chromosome)
                    
            except Exception as e:
                # Log error but continue trying
                print(f"Error generating schedule (attempt {attempts}): {str(e)}")
                continue
        
        if len(population) < population_size:
            print(f"Warning: Only generated {len(population)} valid schedules out of {population_size} requested")
        
        return population
    
    def _generate_single_schedule(self, working_days, enforce_shifts, enforce_specializations,
                                enforce_star_schedules, enforce_cabinet_bindings):
        """Generate a single valid schedule (chromosome)"""
        
        schedule = []  # List of genes
        
        for day in working_days:
            # Get demand for this day
            day_demand = self._get_day_demand(day)
            
            # Generate assignments for this day
            day_assignments = self._generate_day_assignments(
                day, day_demand, enforce_shifts, enforce_specializations,
                enforce_star_schedules, enforce_cabinet_bindings
            )
            
            schedule.extend(day_assignments)
        
        return schedule
    
    def _generate_day_assignments(self, day, day_demand, enforce_shifts, enforce_specializations,
                                enforce_star_schedules, enforce_cabinet_bindings):
        """Generate doctor-cabinet assignments for a single day"""
        
        day_assignments = []
        
        # Group demand by service and time slot
        demand_slots = self._create_demand_slots(day, day_demand)
        
        # Get available doctors for this day
        available_doctors = self._get_available_doctors(day)
        
        # Handle star doctors first (fixed schedules)
        if enforce_star_schedules:
            star_assignments = self._assign_star_doctors(day, available_doctors)
            day_assignments.extend(star_assignments)
            
            # Remove assigned doctors from available pool
            assigned_doctor_ids = {gene['doctor_id'] for gene in star_assignments}
            available_doctors = [d for d in available_doctors if d['doctor_id'] not in assigned_doctor_ids]
        
        # Assign remaining slots
        remaining_slots = self._calculate_remaining_slots(demand_slots, day_assignments)
        
        for slot in remaining_slots:
            if available_doctors:
                assignment = self._assign_doctor_to_slot(
                    slot, available_doctors, enforce_shifts, enforce_specializations, enforce_cabinet_bindings
                )
                
                if assignment:
                    day_assignments.append(assignment)
                    
                    # Update available doctors (consider daily limits)
                    available_doctors = self._update_available_doctors(available_doctors, assignment)
        
        return day_assignments
    
    def _get_working_days(self, target_month):
        """Get list of working days for the target month"""
        
        # Start from first day of month
        start_date = target_month.replace(day=1)
        
        # Find last day of month
        if start_date.month == 12:
            end_date = start_date.replace(year=start_date.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end_date = start_date.replace(month=start_date.month + 1, day=1) - timedelta(days=1)
        
        working_days = []
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends (assuming Mon-Fri work schedule)
            if current_date.weekday() < 5:  # 0=Monday, 6=Sunday
                working_days.append(current_date)
            current_date += timedelta(days=1)
        
        return working_days
    
    def _get_day_demand(self, day):
        """Get demand forecast for a specific day"""
        
        day_str = day.strftime('%Y-%m-%d')
        day_demand = self.demand_forecast[
            pd.to_datetime(self.demand_forecast['date']).dt.strftime('%Y-%m-%d') == day_str
        ].copy()
        
        return day_demand if not day_demand.empty else pd.DataFrame()
    
    def _create_demand_slots(self, day, day_demand):
        """Convert daily demand into time slots"""
        
        slots = []
        
        for _, demand_row in day_demand.iterrows():
            service = demand_row['service']
            total_demand = int(demand_row['predicted_demand'])
            dms_demand = int(demand_row['dms_demand'])
            
            # Create hourly slots (simplified - 8 working hours)
            slots_per_hour = max(1, total_demand // 8)
            
            for hour in range(8, 18):  # 8 AM to 6 PM
                for slot_num in range(slots_per_hour):
                    slots.append({
                        'day': day,
                        'hour': hour,
                        'service': service,
                        'is_dms': random.random() < (dms_demand / total_demand) if total_demand > 0 else False,
                        'slot_id': f"{day.strftime('%Y%m%d')}_{hour:02d}_{slot_num:02d}"
                    })
        
        return slots
    
    def _get_available_doctors(self, day):
        """Get doctors available for a specific day"""
        
        available = []
        
        for _, doctor in self.doctors_df.iterrows():
            # Check if doctor is available this day (simplified logic)
            if self._is_doctor_available_on_day(doctor, day):
                available.append(doctor.to_dict())
        
        return available
    
    def _is_doctor_available_on_day(self, doctor, day):
        """Check if doctor is available on a specific day"""
        
        # Simplified availability check
        # In real implementation, this would check vacation schedules, 
        # personal calendars, etc.
        
        weekday = day.weekday()
        
        # Assume most doctors work Mon-Fri
        if weekday >= 5:  # Weekend
            return False
        
        # Star doctors might have fixed schedules
        if doctor.get('is_star', False):
            # Check their fixed schedule (simplified)
            return weekday in [0, 1, 2, 3, 4]  # Mon-Fri
        
        return True
    
    def _assign_star_doctors(self, day, available_doctors):
        """Assign star doctors to their fixed schedules"""
        
        star_assignments = []
        star_doctors = [d for d in available_doctors if d.get('is_star', False)]
        
        for doctor in star_doctors:
            # Get their fixed schedule for this day
            fixed_schedule = self._get_star_doctor_schedule(doctor, day)
            
            for assignment in fixed_schedule:
                gene = {
                    'day': day,
                    'doctor_id': doctor['doctor_id'],
                    'cabinet_id': assignment['cabinet_id'],
                    'shift': assignment['shift'],
                    'start_time': assignment['start_time'],
                    'end_time': assignment['end_time'],
                    'is_star_assignment': True
                }
                star_assignments.append(gene)
        
        return star_assignments
    
    def _get_star_doctor_schedule(self, doctor, day):
        """Get fixed schedule for a star doctor"""
        
        # Simplified implementation - in reality this would come from database
        doctor_id = doctor['doctor_id']
        shift = doctor.get('shift_type', 'day')
        
        # Find suitable cabinet (consider bindings)
        suitable_cabinet = self._find_suitable_cabinet_for_doctor(doctor)
        
        if suitable_cabinet:
            shift_times = self.shift_definitions.get(shift, self.shift_definitions['day'])
            
            return [{
                'cabinet_id': suitable_cabinet['cabinet_id'],
                'shift': shift,
                'start_time': shift_times['start'],
                'end_time': shift_times['end']
            }]
        
        return []
    
    def _calculate_remaining_slots(self, demand_slots, existing_assignments):
        """Calculate slots that still need to be filled"""
        
        # Simplified - return all demand slots minus those already covered
        # In real implementation, this would be more sophisticated
        
        return demand_slots  # For now, return all slots
    
    def _assign_doctor_to_slot(self, slot, available_doctors, enforce_shifts, 
                             enforce_specializations, enforce_cabinet_bindings):
        """Assign a doctor to a specific time slot"""
        
        # Filter doctors by constraints
        suitable_doctors = self._filter_suitable_doctors(
            available_doctors, slot, enforce_shifts, enforce_specializations
        )
        
        if not suitable_doctors:
            return None
        
        # Select doctor (random selection for genetic diversity)
        selected_doctor = random.choice(suitable_doctors)
        
        # Find suitable cabinet
        suitable_cabinet = self._find_suitable_cabinet_for_doctor(
            selected_doctor, enforce_cabinet_bindings
        )
        
        if not suitable_cabinet:
            return None
        
        # Create gene
        gene = {
            'day': slot['day'],
            'doctor_id': selected_doctor['doctor_id'],
            'cabinet_id': suitable_cabinet['cabinet_id'],
            'shift': selected_doctor.get('shift_type', 'day'),
            'start_time': f"{slot['hour']:02d}:00",
            'end_time': f"{slot['hour']+1:02d}:00",
            'service': slot['service'],
            'is_dms': slot['is_dms'],
            'slot_id': slot['slot_id']
        }
        
        return gene
    
    def _filter_suitable_doctors(self, available_doctors, slot, enforce_shifts, enforce_specializations):
        """Filter doctors based on constraints"""
        
        suitable = []
        
        for doctor in available_doctors:
            # Check shift compatibility
            if enforce_shifts:
                if not self._is_shift_compatible(doctor, slot):
                    continue
            
            # Check specialization
            if enforce_specializations:
                if not self._is_specialization_compatible(doctor, slot):
                    continue
            
            suitable.append(doctor)
        
        return suitable
    
    def _is_shift_compatible(self, doctor, slot):
        """Check if doctor's shift is compatible with slot time"""
        
        doctor_shift = doctor.get('shift_type', 'day')
        slot_hour = slot['hour']
        
        if doctor_shift == 'morning' and slot_hour >= 14:
            return False
        elif doctor_shift == 'evening' and slot_hour < 14:
            return False
        elif doctor_shift == 'night' and 8 <= slot_hour < 20:
            return False
        
        return True
    
    def _is_specialization_compatible(self, doctor, slot):
        """Check if doctor's specialization matches required service"""
        
        doctor_specialty = doctor.get('specialty', '').lower()
        service = slot.get('service', '').lower()
        
        # Simplified mapping - in reality this would be more comprehensive
        specialty_service_mapping = {
            'терапевт': ['прием терапевта', 'консультация терапевта', 'общий осмотр'],
            'кардиолог': ['прием кардиолога', 'кардиологический осмотр', 'экг'],
            'педиатр': ['прием педиатра', 'детский осмотр'],
            'гинеколог': ['прием гинеколога', 'гинекологический осмотр'],
            'невролог': ['прием невролога', 'неврологический осмотр']
        }
        
        allowed_services = specialty_service_mapping.get(doctor_specialty, [])
        
        # If no specific mapping, allow (flexible assignment)
        if not allowed_services:
            return True
        
        return any(allowed in service for allowed in allowed_services)
    
    def _find_suitable_cabinet_for_doctor(self, doctor, enforce_cabinet_bindings=True):
        """Find suitable cabinet for doctor"""
        
        doctor_specialty = doctor.get('specialty', '')
        cabinet_binding = doctor.get('cabinet_binding')
        
        # Check hard binding first
        if enforce_cabinet_bindings and cabinet_binding:
            bound_cabinet = self.cabinets_df[
                self.cabinets_df['cabinet_id'] == cabinet_binding
            ]
            if not bound_cabinet.empty:
                return bound_cabinet.iloc[0].to_dict()
        
        # Find cabinets that allow this specialty
        suitable_cabinets = []
        
        for _, cabinet in self.cabinets_df.iterrows():
            allowed_specialties = cabinet.get('specialty_allowed', [])
            
            if isinstance(allowed_specialties, list):
                if doctor_specialty in allowed_specialties or len(allowed_specialties) == 0:
                    suitable_cabinets.append(cabinet.to_dict())
        
        return random.choice(suitable_cabinets) if suitable_cabinets else None
    
    def _update_available_doctors(self, available_doctors, assignment):
        """Update available doctors after assignment (consider daily limits)"""
        
        # Simplified - in real implementation, track daily work hours
        # For now, keep all doctors available (allows multiple assignments per day)
        
        return available_doctors
    
    def _build_compatibility_matrix(self):
        """Pre-build doctor-cabinet compatibility matrix"""
        
        compatibility = {}
        
        for _, doctor in self.doctors_df.iterrows():
            doctor_id = doctor['doctor_id']
            doctor_specialty = doctor.get('specialty', '')
            
            compatible_cabinets = []
            
            for _, cabinet in self.cabinets_df.iterrows():
                allowed_specialties = cabinet.get('specialty_allowed', [])
                
                if isinstance(allowed_specialties, list):
                    if doctor_specialty in allowed_specialties or len(allowed_specialties) == 0:
                        compatible_cabinets.append(cabinet['cabinet_id'])
            
            compatibility[doctor_id] = compatible_cabinets
        
        return compatibility
    
    def _is_valid_schedule(self, chromosome):
        """Validate that a chromosome represents a valid schedule"""
        
        if not chromosome:
            return False
        
        # Check for basic consistency
        for gene in chromosome:
            required_fields = ['day', 'doctor_id', 'cabinet_id', 'shift', 'start_time', 'end_time']
            if not all(field in gene for field in required_fields):
                return False
        
        # Check for conflicts (same cabinet, same time)
        time_slots = {}
        
        for gene in chromosome:
            day = gene['day']
            cabinet_id = gene['cabinet_id']
            start_time = gene['start_time']
            
            slot_key = f"{day}_{cabinet_id}_{start_time}"
            
            if slot_key in time_slots:
                return False  # Conflict detected
            
            time_slots[slot_key] = gene
        
        return True
    
    def validate_schedule(self, chromosome):
        """Detailed validation of schedule constraints"""
        
        validation_results = {
            'basic_structure': True,
            'no_time_conflicts': True,
            'specialty_compliance': True,
            'shift_compliance': True,
            'cabinet_availability': True
        }
        
        if not chromosome:
            return {key: False for key in validation_results.keys()}
        
        # Basic structure validation
        for gene in chromosome:
            required_fields = ['day', 'doctor_id', 'cabinet_id', 'shift', 'start_time', 'end_time']
            if not all(field in gene for field in required_fields):
                validation_results['basic_structure'] = False
                break
        
        # Time conflict validation
        time_slots = {}
        for gene in chromosome:
            slot_key = f"{gene['day']}_{gene['cabinet_id']}_{gene['start_time']}"
            if slot_key in time_slots:
                validation_results['no_time_conflicts'] = False
                break
            time_slots[slot_key] = gene
        
        # Add more detailed validations as needed
        
        return validation_results
