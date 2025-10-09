import random
import numpy as np
from deap import base, creator, tools, algorithms
import pandas as pd
from fitness_evaluator import FitnessEvaluator
import copy

class ScheduleOptimizer:
    def __init__(self, doctors_df, cabinets_df, appointments_df, revenue_df, demand_forecast, financial_metrics):
        self.doctors_df = doctors_df
        self.cabinets_df = cabinets_df
        self.appointments_df = appointments_df
        self.revenue_df = revenue_df
        self.demand_forecast = demand_forecast
        self.financial_metrics = financial_metrics
        
        # Initialize fitness evaluator
        self.fitness_evaluator = FitnessEvaluator(
            doctors_df, cabinets_df, appointments_df, 
            revenue_df, demand_forecast, financial_metrics
        )
        
        # Set up DEAP framework
        self._setup_deap()
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework"""
        
        # Create fitness class (maximize fitness)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
    
    def optimize(self, initial_population, generations, mutation_rate, crossover_rate, weights, callback=None):
        """Run genetic algorithm optimization"""
        
        if not initial_population:
            raise ValueError("Initial population cannot be empty")
        
        # Convert initial population to DEAP format
        population = []
        for chromosome in initial_population:
            individual = creator.Individual(chromosome)
            population.append(individual)
        
        # Store weights for fitness evaluation
        self.current_weights = weights
        
        # Register genetic operators
        self._register_operators(mutation_rate, crossover_rate)
        
        # Evaluate initial population
        self._evaluate_population(population)
        
        # Evolution statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        
        # Hall of Fame to keep track of best individuals
        hall_of_fame = tools.HallOfFame(1)
        
        # Evolution history
        evolution_history = []
        
        # Run evolution
        for generation in range(generations):
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            self._evaluate_population(invalid_individuals)
            
            # Replace population
            population[:] = offspring
            
            # Update hall of fame
            hall_of_fame.update(population)
            
            # Record statistics
            record = stats.compile(population)
            evolution_history.append({
                'generation': generation,
                'best_fitness': record['max'],
                'avg_fitness': record['avg'],
                'min_fitness': record['min'],
                'std_fitness': record['std']
            })
            
            # Callback for progress updates
            if callback:
                callback(generation, record)
        
        # Return best solution and evolution history
        best_individual = hall_of_fame[0]
        return list(best_individual), evolution_history
    
    def _register_operators(self, mutation_rate, crossover_rate):
        """Register genetic operators with DEAP toolbox"""
        
        # Selection: Tournament selection
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Crossover: Custom schedule crossover
        self.toolbox.register("mate", self._crossover_schedules)
        
        # Mutation: Custom schedule mutation
        self.toolbox.register("mutate", self._mutate_schedule)
        
        # Cloning
        self.toolbox.register("clone", copy.deepcopy)
    
    def _evaluate_population(self, population):
        """Evaluate fitness for a population"""
        
        for individual in population:
            if not individual.fitness.valid:
                fitness_value = self.fitness_evaluator.evaluate_fitness(individual, self.current_weights)
                individual.fitness.values = (fitness_value,)
    
    def _crossover_schedules(self, parent1, parent2):
        """Custom crossover operator for schedules"""
        
        if not parent1 or not parent2:
            return parent1, parent2
        
        # Uniform crossover with schedule-aware logic
        child1, child2 = [], []
        
        # Create day-based crossover
        parent1_by_day = self._group_by_day(parent1)
        parent2_by_day = self._group_by_day(parent2)
        
        all_days = set(parent1_by_day.keys()) | set(parent2_by_day.keys())
        
        for day in all_days:
            day1_genes = parent1_by_day.get(day, [])
            day2_genes = parent2_by_day.get(day, [])
            
            # Random choice of which parent to take this day from
            if random.random() < 0.5:
                child1.extend(day1_genes)
                child2.extend(day2_genes)
            else:
                child1.extend(day2_genes)
                child2.extend(day1_genes)
        
        # Update parents in place
        parent1[:] = child1
        parent2[:] = child2
        
        return parent1, parent2
    
    def _mutate_schedule(self, individual):
        """Custom mutation operator for schedules"""
        
        if not individual:
            return individual,
        
        mutation_types = ['swap_doctors', 'swap_cabinets', 'change_times', 'swap_days']
        mutation_type = random.choice(mutation_types)
        
        if mutation_type == 'swap_doctors':
            self._mutate_swap_doctors(individual)
        elif mutation_type == 'swap_cabinets':
            self._mutate_swap_cabinets(individual)
        elif mutation_type == 'change_times':
            self._mutate_change_times(individual)
        elif mutation_type == 'swap_days':
            self._mutate_swap_days(individual)
        
        return individual,
    
    def _mutate_swap_doctors(self, individual):
        """Swap doctors between two random assignments"""
        
        if len(individual) < 2:
            return
        
        idx1, idx2 = random.sample(range(len(individual)), 2)
        gene1, gene2 = individual[idx1], individual[idx2]
        
        # Check if swap is valid (same specialty requirements)
        if self._is_doctor_swap_valid(gene1, gene2):
            gene1['doctor_id'], gene2['doctor_id'] = gene2['doctor_id'], gene1['doctor_id']
    
    def _mutate_swap_cabinets(self, individual):
        """Swap cabinets between two assignments of the same doctor"""
        
        if len(individual) < 2:
            return
        
        # Find assignments by the same doctor
        doctor_assignments = {}
        for idx, gene in enumerate(individual):
            doctor_id = gene['doctor_id']
            if doctor_id not in doctor_assignments:
                doctor_assignments[doctor_id] = []
            doctor_assignments[doctor_id].append(idx)
        
        # Find a doctor with multiple assignments
        for doctor_id, assignments in doctor_assignments.items():
            if len(assignments) >= 2:
                idx1, idx2 = random.sample(assignments, 2)
                gene1, gene2 = individual[idx1], individual[idx2]
                
                if self._is_cabinet_swap_valid(gene1, gene2):
                    gene1['cabinet_id'], gene2['cabinet_id'] = gene2['cabinet_id'], gene1['cabinet_id']
                break
    
    def _mutate_change_times(self, individual):
        """Change time slot for a random assignment"""
        
        if not individual:
            return
        
        gene = random.choice(individual)
        
        # Generate new time within doctor's shift
        doctor_id = gene['doctor_id']
        doctor_info = self.fitness_evaluator.doctor_lookup.get(doctor_id, {})
        shift_type = doctor_info.get('shift_type', 'day')
        
        new_times = self._generate_valid_times(shift_type)
        if new_times:
            start_time, end_time = random.choice(new_times)
            gene['start_time'] = start_time
            gene['end_time'] = end_time
    
    def _mutate_swap_days(self, individual):
        """Swap assignments between two different days"""
        
        if len(individual) < 2:
            return
        
        # Group by day
        by_day = self._group_by_day(individual)
        days = list(by_day.keys())
        
        if len(days) >= 2:
            day1, day2 = random.sample(days, 2)
            
            if by_day[day1] and by_day[day2]:
                gene1 = random.choice(by_day[day1])
                gene2 = random.choice(by_day[day2])
                
                # Swap days
                gene1['day'], gene2['day'] = gene2['day'], gene1['day']
    
    def _group_by_day(self, individual):
        """Group schedule genes by day"""
        
        by_day = {}
        for gene in individual:
            day = gene.get('day')
            if day:
                if day not in by_day:
                    by_day[day] = []
                by_day[day].append(gene)
        return by_day
    
    def _is_doctor_swap_valid(self, gene1, gene2):
        """Check if swapping doctors between assignments is valid"""
        
        doctor1_id = gene1['doctor_id']
        doctor2_id = gene2['doctor_id']
        service1 = gene1.get('service', '')
        service2 = gene2.get('service', '')
        
        doctor1_info = self.fitness_evaluator.doctor_lookup.get(doctor1_id, {})
        doctor2_info = self.fitness_evaluator.doctor_lookup.get(doctor2_id, {})
        
        specialty1 = doctor1_info.get('specialty', '').lower()
        specialty2 = doctor2_info.get('specialty', '').lower()
        
        # Check if doctors can handle each other's services
        can_do_service1 = self.fitness_evaluator._is_service_compatible(specialty2, service1)
        can_do_service2 = self.fitness_evaluator._is_service_compatible(specialty1, service2)
        
        return can_do_service1 and can_do_service2
    
    def _is_cabinet_swap_valid(self, gene1, gene2):
        """Check if swapping cabinets is valid"""
        
        doctor_id = gene1['doctor_id']  # Same doctor for both
        cabinet1_id = gene1['cabinet_id']
        cabinet2_id = gene2['cabinet_id']
        
        doctor_info = self.fitness_evaluator.doctor_lookup.get(doctor_id, {})
        specialty = doctor_info.get('specialty', '')
        
        # Check if both cabinets allow this specialty
        cabinet1_info = self.fitness_evaluator.cabinet_lookup.get(cabinet1_id, {})
        cabinet2_info = self.fitness_evaluator.cabinet_lookup.get(cabinet2_id, {})
        
        allowed1 = cabinet1_info.get('specialty_allowed', [])
        allowed2 = cabinet2_info.get('specialty_allowed', [])
        
        can_use_cabinet1 = not allowed1 or specialty in allowed1
        can_use_cabinet2 = not allowed2 or specialty in allowed2
        
        return can_use_cabinet1 and can_use_cabinet2
    
    def _generate_valid_times(self, shift_type):
        """Generate valid time slots for a shift type"""
        
        shift_definitions = {
            'morning': [('08:00', '09:00'), ('09:00', '10:00'), ('10:00', '11:00'), 
                       ('11:00', '12:00'), ('12:00', '13:00'), ('13:00', '14:00')],
            'evening': [('14:00', '15:00'), ('15:00', '16:00'), ('16:00', '17:00'),
                       ('17:00', '18:00'), ('18:00', '19:00'), ('19:00', '20:00')],
            'day': [('09:00', '10:00'), ('10:00', '11:00'), ('11:00', '12:00'),
                   ('12:00', '13:00'), ('14:00', '15:00'), ('15:00', '16:00'),
                   ('16:00', '17:00'), ('17:00', '18:00')],
            'night': [('20:00', '21:00'), ('21:00', '22:00'), ('22:00', '23:00'),
                     ('06:00', '07:00'), ('07:00', '08:00')]
        }
        
        return shift_definitions.get(shift_type, shift_definitions['day'])
    
    def get_optimization_insights(self, evolution_history):
        """Generate insights from optimization process"""
        
        if not evolution_history:
            return {}
        
        # Convergence analysis
        best_fitnesses = [gen['best_fitness'] for gen in evolution_history]
        avg_fitnesses = [gen['avg_fitness'] for gen in evolution_history]
        
        # Calculate improvement rate
        if len(best_fitnesses) >= 2:
            improvement_rate = (best_fitnesses[-1] - best_fitnesses[0]) / len(best_fitnesses)
        else:
            improvement_rate = 0
        
        # Find plateau detection
        plateau_threshold = 0.001
        plateau_start = None
        
        for i in range(1, len(best_fitnesses)):
            if abs(best_fitnesses[i] - best_fitnesses[i-1]) < plateau_threshold:
                if plateau_start is None:
                    plateau_start = i
            else:
                plateau_start = None
        
        # Diversity analysis (std of fitness values)
        diversity_trend = [gen['std_fitness'] for gen in evolution_history]
        
        return {
            'total_generations': len(evolution_history),
            'initial_fitness': best_fitnesses[0] if best_fitnesses else 0,
            'final_fitness': best_fitnesses[-1] if best_fitnesses else 0,
            'improvement_rate': improvement_rate,
            'plateau_start': plateau_start,
            'convergence_achieved': plateau_start is not None and plateau_start < len(best_fitnesses) * 0.8,
            'diversity_maintained': np.mean(diversity_trend) > 0.01 if diversity_trend else False,
            'fitness_trend': best_fitnesses,
            'avg_fitness_trend': avg_fitnesses,
            'diversity_trend': diversity_trend
        }
