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
            
            # Remove assigned doctors from available pool for overlapping times
            available_doctors = self._filter_available_doctors_by_existing_assignments(
                available_doctors, star_assignments
            )
        
        # Sort slots by time to assign them in chronological order
        demand_slots.sort(key=lambda x: (x['hour'], x.get('minute', 0)))
        
        # Assign remaining slots
        for slot in demand_slots:
            if available_doctors:
                assignment = self._assign_doctor_to_slot_with_conflict_check(
                    slot, available_doctors, day_assignments, enforce_shifts, 
                    enforce_specializations, enforce_cabinet_bindings
                )
                
                if assignment:
                    day_assignments.append(assignment)
        
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
        
        # For now, use the same demand pattern for all working days
        # In a real implementation, this would be more sophisticated
        if not self.demand_forecast.empty:
            # Use the first forecast entry as a template for all days
            day_demand = self.demand_forecast.copy()
            day_demand['date'] = day.strftime('%Y-%m-%d')
            return day_demand
        
        return pd.DataFrame()
    
    def _create_demand_slots(self, day, day_demand):
        """Convert daily demand into time slots"""
        
        slots = []
        
        for _, demand_row in day_demand.iterrows():
            service = demand_row['service']
            total_demand = int(demand_row['predicted_demand'])
            dms_demand = int(demand_row['dms_demand'])
            
            # Create individual appointment slots based on actual demand
            for slot_num in range(total_demand):
                # Distribute slots across working hours (8 AM to 6 PM)
                hour = 8 + (slot_num % 10)  # 10 working hours
                minute = (slot_num // 10) * 30  # 30-minute slots
                
                if minute >= 60:
                    hour += minute // 60
                    minute = minute % 60
                
                # Skip lunch break (12-13)
                if hour == 12:
                    hour = 13
                
                # Don't exceed working hours
                if hour >= 18:
                    hour = 8 + (slot_num % 9)  # Skip lunch hour
                    if hour >= 12:
                        hour += 1
                
                slots.append({
                    'day': day,
                    'hour': hour,
                    'minute': minute,
                    'service': service,
                    'is_dms': slot_num < dms_demand,  # First N slots are DMS
                    'slot_id': f"{day.strftime('%Y%m%d')}_{hour:02d}_{minute:02d}_{slot_num:03d}",
                    'duration_minutes': 30  # Standard appointment duration
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
        
        if not suitable_cabinet:
            return []
        
        # Create multiple time slots instead of one long shift
        appointments = []
        shift_ranges = {
            'morning': (8, 14),   # 8:00 - 14:00
            'evening': (14, 20),  # 14:00 - 20:00
            'day': (9, 18),       # 9:00 - 18:00
            'night': (20, 8)      # 20:00 - 8:00 (next day)
        }
        
        if shift in shift_ranges:
            start_hour, end_hour = shift_ranges[shift]
            
            # Create 30-minute appointment slots
            current_hour = start_hour
            while current_hour < end_hour:
                # Skip lunch break for day shifts
                if shift == 'day' and current_hour == 12:
                    current_hour = 13
                    continue
                
                appointments.append({
                    'cabinet_id': suitable_cabinet['cabinet_id'],
                    'shift': shift,
                    'start_time': f"{current_hour:02d}:00",
                    'end_time': f"{current_hour:02d}:30"
                })
                
                appointments.append({
                    'cabinet_id': suitable_cabinet['cabinet_id'],
                    'shift': shift,
                    'start_time': f"{current_hour:02d}:30",
                    'end_time': f"{current_hour+1:02d}:00"
                })
                
                current_hour += 1
        
        # Limit to reasonable number of appointments (e.g., 4-6 per day)
        return appointments[:6]
    
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
        
        # Calculate precise time slots
        start_hour = slot['hour']
        start_minute = slot.get('minute', 0)
        duration = slot.get('duration_minutes', 30)
        
        end_minute = start_minute + duration
        end_hour = start_hour
        if end_minute >= 60:
            end_hour += end_minute // 60
            end_minute = end_minute % 60
        
        # Create gene
        gene = {
            'day': slot['day'],
            'doctor_id': selected_doctor['doctor_id'],
            'cabinet_id': suitable_cabinet['cabinet_id'],
            'shift': selected_doctor.get('shift_type', 'day'),
            'start_time': f"{start_hour:02d}:{start_minute:02d}",
            'end_time': f"{end_hour:02d}:{end_minute:02d}",
            'service': slot['service'],
            'is_dms': slot['is_dms'],
            'slot_id': slot['slot_id'],
            'duration_minutes': duration
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
        
        # Define shift time ranges
        shift_ranges = {
            'morning': (8, 14),   # 8:00 - 14:00
            'evening': (14, 20),  # 14:00 - 20:00
            'day': (9, 18),       # 9:00 - 18:00
            'night': (20, 8)      # 20:00 - 8:00 (next day)
        }
        
        if doctor_shift in shift_ranges:
            start_hour, end_hour = shift_ranges[doctor_shift]
            
            if doctor_shift == 'night':  # Special case for night shift
                return slot_hour >= start_hour or slot_hour < end_hour
            else:
                return start_hour <= slot_hour < end_hour
        
        return True  # Default: allow if shift not defined
    
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
        
        # Check for time conflicts (same cabinet, overlapping times)
        if not self._check_cabinet_availability(chromosome):
            return False
        
        # Check for doctor overload (same doctor, overlapping times)
        if not self._check_doctor_availability(chromosome):
            return False
        
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
        
        # Cabinet availability validation
        if not self._check_cabinet_availability(chromosome):
            validation_results['cabinet_availability'] = False
        
        # Doctor availability validation
        if not self._check_doctor_availability(chromosome):
            validation_results['doctor_availability'] = True  # Add this field
        
        return validation_results
    
    def _check_cabinet_availability(self, chromosome):
        """Check for cabinet time conflicts"""
        
        cabinet_schedule = {}  # cabinet_id -> list of (start_time, end_time, day)
        
        for gene in chromosome:
            cabinet_id = gene['cabinet_id']
            day = gene['day']
            start_time = gene['start_time']
            end_time = gene['end_time']
            
            # Convert time strings to minutes for easier comparison
            start_minutes = self._time_to_minutes(start_time)
            end_minutes = self._time_to_minutes(end_time)
            
            if cabinet_id not in cabinet_schedule:
                cabinet_schedule[cabinet_id] = []
            
            # Check for conflicts with existing appointments
            for existing_start, existing_end, existing_day in cabinet_schedule[cabinet_id]:
                if day == existing_day:  # Same day
                    # Check for time overlap
                    if not (end_minutes <= existing_start or start_minutes >= existing_end):
                        return False  # Conflict found
            
            cabinet_schedule[cabinet_id].append((start_minutes, end_minutes, day))
        
        return True
    
    def _check_doctor_availability(self, chromosome):
        """Check for doctor time conflicts"""
        
        doctor_schedule = {}  # doctor_id -> list of (start_time, end_time, day)
        
        for gene in chromosome:
            doctor_id = gene['doctor_id']
            day = gene['day']
            start_time = gene['start_time']
            end_time = gene['end_time']
            
            # Convert time strings to minutes for easier comparison
            start_minutes = self._time_to_minutes(start_time)
            end_minutes = self._time_to_minutes(end_time)
            
            if doctor_id not in doctor_schedule:
                doctor_schedule[doctor_id] = []
            
            # Check for conflicts with existing appointments
            for existing_start, existing_end, existing_day in doctor_schedule[doctor_id]:
                if day == existing_day:  # Same day
                    # Check for time overlap
                    if not (end_minutes <= existing_start or start_minutes >= existing_end):
                        return False  # Conflict found
            
            doctor_schedule[doctor_id].append((start_minutes, end_minutes, day))
        
        return True
    
    def _time_to_minutes(self, time_str):
        """Convert time string (HH:MM) to minutes since midnight"""
        try:
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        except:
            return 0
    
    def _filter_available_doctors_by_existing_assignments(self, available_doctors, existing_assignments):
        """Filter doctors based on existing time assignments"""
        
        # For now, keep all doctors available but we'll check conflicts during assignment
        # In a more sophisticated implementation, we could pre-filter based on time conflicts
        return available_doctors
    
    def _assign_doctor_to_slot_with_conflict_check(self, slot, available_doctors, existing_assignments,
                                                 enforce_shifts, enforce_specializations, enforce_cabinet_bindings):
        """Assign doctor to slot with conflict checking"""
        
        # Filter doctors by constraints
        suitable_doctors = self._filter_suitable_doctors(
            available_doctors, slot, enforce_shifts, enforce_specializations
        )
        
        if not suitable_doctors:
            return None
        
        # Try each suitable doctor until we find one without conflicts
        random.shuffle(suitable_doctors)  # Randomize for genetic diversity
        
        for doctor in suitable_doctors:
            # Find suitable cabinet
            suitable_cabinet = self._find_suitable_cabinet_for_doctor(
                doctor, enforce_cabinet_bindings
            )
            
            if not suitable_cabinet:
                continue
            
            # Create potential assignment
            potential_gene = self._create_gene_from_slot_and_doctor(slot, doctor, suitable_cabinet)
            
            # Check for conflicts with existing assignments
            if self._has_conflicts_with_existing(potential_gene, existing_assignments):
                continue
            
            return potential_gene
        
        return None  # No suitable doctor found without conflicts
    
    def _create_gene_from_slot_and_doctor(self, slot, doctor, cabinet):
        """Create a gene from slot, doctor, and cabinet information"""
        
        # Calculate precise time slots
        start_hour = slot['hour']
        start_minute = slot.get('minute', 0)
        duration = slot.get('duration_minutes', 30)
        
        end_minute = start_minute + duration
        end_hour = start_hour
        if end_minute >= 60:
            end_hour += end_minute // 60
            end_minute = end_minute % 60
        
        return {
            'day': slot['day'],
            'doctor_id': doctor['doctor_id'],
            'cabinet_id': cabinet['cabinet_id'],
            'shift': doctor.get('shift_type', 'day'),
            'start_time': f"{start_hour:02d}:{start_minute:02d}",
            'end_time': f"{end_hour:02d}:{end_minute:02d}",
            'service': slot['service'],
            'is_dms': slot['is_dms'],
            'slot_id': slot['slot_id'],
            'duration_minutes': duration
        }
    
    def _has_conflicts_with_existing(self, potential_gene, existing_assignments):
        """Check if potential gene conflicts with existing assignments"""
        
        doctor_id = potential_gene['doctor_id']
        cabinet_id = potential_gene['cabinet_id']
        day = potential_gene['day']
        start_time = potential_gene['start_time']
        end_time = potential_gene['end_time']
        
        start_minutes = self._time_to_minutes(start_time)
        end_minutes = self._time_to_minutes(end_time)
        
        for existing_gene in existing_assignments:
            existing_day = existing_gene['day']
            
            if day != existing_day:
                continue  # Different day, no conflict
            
            existing_start = self._time_to_minutes(existing_gene['start_time'])
            existing_end = self._time_to_minutes(existing_gene['end_time'])
            
            # Check for time overlap
            time_overlap = not (end_minutes <= existing_start or start_minutes >= existing_end)
            
            if time_overlap:
                # Check if same doctor or same cabinet
                if (doctor_id == existing_gene['doctor_id'] or 
                    cabinet_id == existing_gene['cabinet_id']):
                    return True  # Conflict found
        
        return False  # No conflicts
