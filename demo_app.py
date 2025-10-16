#!/usr/bin/env python3
"""
Демонстрационное приложение системы расписаний врачей
Запуск: python3 demo_app.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from schedule_generator import ScheduleGenerator
from genetic_algorithm import ScheduleOptimizer
from fitness_evaluator import FitnessEvaluator

def main():
    print("🏥 ДЕМО: Система умного расписания врачей")
    print("=" * 50)
    
    # Простые тестовые данные
    doctors_df = pd.DataFrame({
        'doctor_id': [1, 2, 3],
        'name': ['Иванов И.И.', 'Петров П.П.', 'Сидоров С.С.'],
        'specialty': ['терапевт', 'кардиолог', 'педиатр'],
        'shift_type': ['day', 'day', 'day'],
        'is_star': [False, False, False],  # Без звездных врачей для стабильности
        'experience_years': [5, 7, 3]
    })
    
    cabinets_df = pd.DataFrame({
        'cabinet_id': [101, 102, 103],
        'cabinet_name': ['Терапия', 'Кардиология', 'Педиатрия'],
        'specialty_allowed': [['терапевт'], ['кардиолог'], ['педиатр']]
    })
    
    appointments_df = pd.DataFrame({
        'doctor_id': [1, 2, 3] * 30,
        'service_name': ['прием терапевта', 'прием кардиолога', 'прием педиатра'] * 30,
        'cost': [1500, 2000, 1800] * 30
    })
    
    revenue_df = pd.DataFrame({
        'doctor_id': [1, 2, 3],
        'total_revenue': [150000, 180000, 120000],
        'appointments_count': [100, 90, 80]
    })
    
    demand_forecast = pd.DataFrame({
        'date': [datetime.now().strftime('%Y-%m-%d')] * 3,
        'service': ['прием терапевта', 'прием кардиолога', 'прием педиатра'],
        'predicted_demand': [6, 4, 5],
        'dms_demand': [2, 2, 1]
    })
    
    financial_metrics = pd.DataFrame({
        'doctor_id': [1, 2, 3],
        'avg_appointment_cost': [1500, 2000, 1800],
        'fill_rate': [0.8, 0.85, 0.75],
        'reliability_coefficient': [0.9, 0.95, 0.85],
        'service_diversity': [3, 2, 3]
    })
    
    print("📊 Данные:")
    print(f"   Врачи: {len(doctors_df)}")
    print(f"   Кабинеты: {len(cabinets_df)}")
    print(f"   Дневной спрос: {demand_forecast['predicted_demand'].sum()}")
    
    # Генерация
    print("\n🔄 Генерация расписаний...")
    generator = ScheduleGenerator(doctors_df, cabinets_df, demand_forecast)
    population = generator.generate_population(5, datetime.now().replace(day=1))
    
    if not population:
        print("❌ Не удалось сгенерировать расписания")
        return
    
    print(f"   ✅ Сгенерировано: {len(population)} расписаний")
    print(f"   📋 Размер расписания: {len(population[0])} назначений")
    
    # Валидация
    print("\n✅ Валидация:")
    validation = generator.validate_schedule(population[0])
    for constraint, status in validation.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {constraint}")
    
    # Оценка приспособленности
    print("\n📈 Оценка приспособленности:")
    evaluator = FitnessEvaluator(
        doctors_df, cabinets_df, appointments_df,
        revenue_df, demand_forecast, financial_metrics
    )
    
    weights = {'demand': 0.3, 'revenue': 0.25, 'reliability': 0.2, 'strategy': 0.15, 'personnel': 0.1}
    
    for i, schedule in enumerate(population):
        fitness = evaluator.evaluate_fitness(schedule, weights)
        print(f"   Расписание {i+1}: {fitness:.4f}")
    
    # Оптимизация
    print("\n🧬 Генетическая оптимизация (5 поколений)...")
    optimizer = ScheduleOptimizer(
        doctors_df, cabinets_df, appointments_df,
        revenue_df, demand_forecast, financial_metrics
    )
    
    best_schedule, evolution_history = optimizer.optimize(
        population, generations=5, mutation_rate=0.1, crossover_rate=0.8, weights=weights
    )
    
    final_fitness = evaluator.evaluate_fitness(best_schedule, weights)
    print(f"   ✅ Финальная приспособленность: {final_fitness:.4f}")
    
    # Анализ результатов
    print("\n📋 Анализ лучшего расписания:")
    breakdown = evaluator.get_fitness_breakdown(best_schedule, weights)
    
    for component, score in breakdown['raw_scores'].items():
        print(f"   {component:12}: {score:.3f}")
    
    # Загрузка врачей
    print("\n👨‍⚕️ Загрузка врачей:")
    doctor_workload = {}
    for gene in best_schedule:
        doctor_id = gene['doctor_id']
        doctor_workload[doctor_id] = doctor_workload.get(doctor_id, 0) + 1
    
    for doctor_id, count in doctor_workload.items():
        doctor_name = doctors_df[doctors_df['doctor_id'] == doctor_id]['name'].iloc[0]
        specialty = doctors_df[doctors_df['doctor_id'] == doctor_id]['specialty'].iloc[0]
        print(f"   {doctor_name} ({specialty}): {count} назначений")
    
    # Покрытие спроса
    print("\n📊 Покрытие спроса (месячное):")
    services_count = {}
    for gene in best_schedule:
        service = gene.get('service', 'unknown')
        services_count[service] = services_count.get(service, 0) + 1
    
    working_days = len(set(gene['day'] for gene in best_schedule))
    
    for _, row in demand_forecast.iterrows():
        service = row['service']
        daily_demand = row['predicted_demand']
        monthly_demand = daily_demand * working_days
        supplied = services_count.get(service, 0)
        coverage = (supplied / monthly_demand * 100) if monthly_demand > 0 else 0
        print(f"   {service}: {supplied}/{monthly_demand} ({coverage:.1f}%)")
    
    print(f"\n🎉 Демонстрация завершена!")
    print(f"   Рабочих дней: {working_days}")
    print(f"   Общих назначений: {len(best_schedule)}")
    print(f"   Средняя загрузка в день: {len(best_schedule) / working_days:.1f}")

if __name__ == "__main__":
    main()