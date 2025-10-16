#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функциональности системы расписаний
Запуск: python3 test_app.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from schedule_generator import ScheduleGenerator
from genetic_algorithm import ScheduleOptimizer
from fitness_evaluator import FitnessEvaluator

def create_test_data():
    """Создание тестовых данных"""
    
    # Врачи
    doctors_df = pd.DataFrame({
        'doctor_id': [1, 2, 3, 4, 5],
        'name': ['Иванов И.И.', 'Петров П.П.', 'Сидоров С.С.', 'Козлова К.К.', 'Морозова М.М.'],
        'specialty': ['терапевт', 'кардиолог', 'педиатр', 'терапевт', 'гинеколог'],
        'shift_type': ['morning', 'day', 'evening', 'day', 'morning'],
        'is_star': [True, False, False, False, True],
        'experience_years': [10, 5, 8, 3, 12]
    })
    
    # Кабинеты
    cabinets_df = pd.DataFrame({
        'cabinet_id': [101, 102, 103, 104, 105],
        'cabinet_name': ['Терапия 1', 'Кардиология', 'Педиатрия', 'Терапия 2', 'Гинекология'],
        'specialty_allowed': [['терапевт'], ['кардиолог'], ['педиатр'], ['терапевт', 'гинеколог'], ['гинеколог']]
    })
    
    # Исторические записи
    appointments_df = pd.DataFrame({
        'doctor_id': [1, 2, 3, 1, 2, 3, 4, 5] * 20,
        'service_name': ['прием терапевта', 'прием кардиолога', 'прием педиатра', 
                        'консультация терапевта', 'кардиологический осмотр', 'детский осмотр',
                        'прием терапевта', 'прием гинеколога'] * 20,
        'cost': [1500, 2000, 1800, 1200, 2200, 1600, 1500, 2500] * 20,
        'date': [datetime.now() - timedelta(days=x) for x in range(160)]
    })
    
    # Доходы
    revenue_df = pd.DataFrame({
        'doctor_id': [1, 2, 3, 4, 5],
        'total_revenue': [200000, 150000, 180000, 120000, 160000],
        'appointments_count': [120, 80, 100, 90, 70]
    })
    
    # Прогноз спроса
    demand_forecast = pd.DataFrame({
        'date': [datetime.now().strftime('%Y-%m-%d')] * 5,
        'service': ['прием терапевта', 'прием кардиолога', 'прием педиатра', 
                   'консультация терапевта', 'прием гинеколога'],
        'predicted_demand': [12, 8, 10, 6, 5],
        'dms_demand': [4, 3, 3, 2, 2]
    })
    
    # Финансовые метрики
    financial_metrics = pd.DataFrame({
        'doctor_id': [1, 2, 3, 4, 5],
        'avg_appointment_cost': [1500, 2000, 1800, 1350, 2300],
        'fill_rate': [0.85, 0.75, 0.90, 0.70, 0.80],
        'reliability_coefficient': [0.95, 0.85, 0.92, 0.78, 0.88],
        'service_diversity': [4, 3, 2, 3, 2]
    })
    
    return doctors_df, cabinets_df, appointments_df, revenue_df, demand_forecast, financial_metrics

def test_schedule_generation():
    """Тестирование генерации расписаний"""
    
    print("🏥 Тестирование системы умного расписания врачей")
    print("=" * 60)
    
    # Создание данных
    print("📊 Создание тестовых данных...")
    doctors_df, cabinets_df, appointments_df, revenue_df, demand_forecast, financial_metrics = create_test_data()
    
    print(f"   Врачи: {len(doctors_df)}")
    print(f"   Кабинеты: {len(cabinets_df)}")
    print(f"   Исторические записи: {len(appointments_df)}")
    print(f"   Дневной спрос: {demand_forecast['predicted_demand'].sum()}")
    
    # Генерация расписаний
    print("\n🔄 Генерация начальной популяции...")
    generator = ScheduleGenerator(doctors_df, cabinets_df, demand_forecast)
    
    target_month = datetime.now().replace(day=1)
    population_size = 10
    
    population = generator.generate_population(
        population_size, target_month,
        enforce_shifts=True,
        enforce_specializations=True,
        enforce_star_schedules=True,
        enforce_cabinet_bindings=True
    )
    
    print(f"   Сгенерировано: {len(population)}/{population_size} расписаний")
    
    if not population:
        print("❌ Не удалось сгенерировать расписания!")
        return
    
    # Валидация
    print("\n✅ Проверка валидности...")
    valid_count = 0
    for i, schedule in enumerate(population):
        validation = generator.validate_schedule(schedule)
        is_valid = all(validation.values())
        if is_valid:
            valid_count += 1
        print(f"   Расписание {i+1}: {'✅' if is_valid else '❌'} ({len(schedule)} назначений)")
    
    print(f"   Валидных расписаний: {valid_count}/{len(population)}")
    
    # Оценка приспособленности
    print("\n📈 Оценка приспособленности...")
    evaluator = FitnessEvaluator(
        doctors_df, cabinets_df, appointments_df,
        revenue_df, demand_forecast, financial_metrics
    )
    
    weights = {
        'demand': 0.3,
        'revenue': 0.25,
        'reliability': 0.2,
        'strategy': 0.15,
        'personnel': 0.1
    }
    
    fitness_scores = []
    for i, schedule in enumerate(population):
        fitness = evaluator.evaluate_fitness(schedule, weights)
        fitness_scores.append(fitness)
        print(f"   Расписание {i+1}: fitness = {fitness:.4f}")
    
    best_idx = np.argmax(fitness_scores)
    print(f"   Лучшее расписание: #{best_idx+1} (fitness = {fitness_scores[best_idx]:.4f})")
    
    # Генетическая оптимизация
    print("\n🧬 Генетическая оптимизация...")
    optimizer = ScheduleOptimizer(
        doctors_df, cabinets_df, appointments_df,
        revenue_df, demand_forecast, financial_metrics
    )
    
    try:
        best_schedule, evolution_history = optimizer.optimize(
            population[:5],  # Используем только 5 лучших
            generations=10,
            mutation_rate=0.1,
            crossover_rate=0.8,
            weights=weights
        )
        
        final_fitness = evaluator.evaluate_fitness(best_schedule, weights)
        print(f"   Оптимизация завершена!")
        print(f"   Поколений: {len(evolution_history)}")
        print(f"   Финальная приспособленность: {final_fitness:.4f}")
        print(f"   Улучшение: {final_fitness - fitness_scores[best_idx]:.4f}")
        
        # Детальный анализ
        print("\n📋 Детальный анализ лучшего расписания:")
        breakdown = evaluator.get_fitness_breakdown(best_schedule, weights)
        
        print("   Компоненты приспособленности:")
        for component, score in breakdown['raw_scores'].items():
            weighted = breakdown['weighted_scores'][component]
            print(f"     {component:12}: {score:.3f} (взвешенный: {weighted:.3f})")
        
        print(f"   Штрафной коэффициент: {breakdown['penalty_factor']:.3f}")
        
        violations = breakdown['violations']
        total_violations = sum(violations.values())
        if total_violations > 0:
            print(f"   ⚠️  Нарушения: {total_violations}")
            for violation_type, count in violations.items():
                if count > 0:
                    print(f"     {violation_type}: {count}")
        else:
            print("   ✅ Нарушений не найдено")
        
        # Статистика по врачам
        print("\n👨‍⚕️ Загрузка врачей:")
        doctor_workload = {}
        for gene in best_schedule:
            doctor_id = gene['doctor_id']
            doctor_workload[doctor_id] = doctor_workload.get(doctor_id, 0) + 1
        
        for doctor_id, appointments in doctor_workload.items():
            doctor_info = doctors_df[doctors_df['doctor_id'] == doctor_id].iloc[0]
            name = doctor_info['name']
            specialty = doctor_info['specialty']
            is_star = '⭐' if doctor_info['is_star'] else ''
            print(f"   {name} ({specialty}){is_star}: {appointments} назначений")
        
        print(f"\n🎉 Тестирование завершено успешно!")
        print(f"   Общее количество назначений: {len(best_schedule)}")
        print(f"   Рабочих дней в месяце: {len(set(gene['day'] for gene in best_schedule))}")
        
    except Exception as e:
        print(f"❌ Ошибка оптимизации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_schedule_generation()