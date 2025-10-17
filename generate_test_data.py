import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Инициализация Faker для русских имен
fake = Faker('ru_RU')
Faker.seed(12345)  # Для воспроизводимости результатов
np.random.seed(12345)
random.seed(12345)

def generate_doctors(num_doctors=100):
    """Генерация данных о врачах"""
    specialties = [
        'Терапия', 'Хирургия', 'Кардиология', 'Неврология', 'Офтальмология',
        'Оториноларингология', 'Дерматология', 'Педиатрия', 'Гинекология', 'Урология'
    ]
    
    shift_types = ['morning', 'evening', 'day']
    
    doctors = []
    for i in range(num_doctors):
        doctor = {
            'doctor_id': f'doc_{i+1}',
            'name': f"{fake.last_name()} {fake.first_name()} {fake.middle_name()}",
            'specialty': random.choice(specialties),
            'shift_type': random.choice(shift_types),
            'experience_years': random.randint(1, 40),
            'is_star': random.random() < 0.1,  # 10% врачей - "звезды"
            'dms_enabled': random.random() < 0.7,  # 70% врачей работают с ДМС
            'house_calls': random.random() < 0.3,  # 30% врачей делают выезды на дом
            'sick_leave_enabled': random.random() < 0.8,  # 80% врачей могут выдавать больничные
            'cabinet_binding': 'пусто'
        }
        doctors.append(doctor)
    
    return pd.DataFrame(doctors)

def generate_cabinets(num_cabinets=60):
    """Генерация данных о кабинетах"""
    specialties_allowed = [
        'Общая консультация',
        'Хирургия',
        'Диагностика',
        'Процедурный кабинет',
        'Физиотерапия'
    ]
    
    branches = ['branch_A', 'branch_B', 'branch_C']
    
    cabinets = []
    for i in range(num_cabinets):
        cabinet = {
            'cabinet_id': f'cab_{i+1}',
            'name': str(random.randint(100, 999)),  # номер кабинета
            'specialty_allowed': random.choice(specialties_allowed),
            'working_hours': '08:00-21:00',
            'branch_id': random.choice(branches)
        }
        cabinets.append(cabinet)
    
    return pd.DataFrame(cabinets)

def generate_seasonal_coefficients():
    """Генерация сезонных коэффициентов"""
    # Берем текущий месяц для прогноза
    current_date = datetime.now()
    forecast_month = current_date.month
    
    seasonal = []
    
    # Добавляем коэффициент 1.1 для терапевтов на прогнозируемый месяц
    seasonal.append({
        'season_id': 'season_1',
        'month_number': forecast_month,
        'specialty': 'Терапия',
        'seasonal_factor': 1.1,
        'comment': 'Повышенный спрос на терапевтов'
    })
    
    # Добавляем нейтральные коэффициенты для остальных специальностей
    specialties = ['Хирургия', 'Кардиология', 'Неврология', 'Офтальмология',
                  'Оториноларингология', 'Дерматология', 'Педиатрия', 'Гинекология', 'Урология']
    
    for i, specialty in enumerate(specialties, 2):
        seasonal.append({
            'season_id': f'season_{i}',
            'month_number': forecast_month,
            'specialty': specialty,
            'seasonal_factor': 1.0,
            'comment': 'Стандартный сезонный коэффициент'
        })
    
    return pd.DataFrame(seasonal)

def generate_promo_calendar():
    """Генерация календаря маркетинговых акций"""
    # Берем текущий месяц для прогноза
    current_date = datetime.now()
    start_date = current_date.replace(day=1)  # Начало месяца
    end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)  # Конец месяца
    
    promo = []
    
    # Добавляем акцию на услуги офтальмолога
    promo.append({
        'promo_id': 'promo_1',
        'start_date': start_date.strftime("%Y-%m-%d %H:%M:%S"),
        'end_date': end_date.strftime("%Y-%m-%d %H:%M:%S"),
        'specialty': 'Офтальмология',
        'promo_factor': 1.01,
        'promo_name': 'Акция на офтальмологические услуги'
    })
    
    return pd.DataFrame(promo)

def generate_appointments(doctors_df, cabinets_df, num_appointments=100000):
    """Генерация данных о приемах"""
    # Настройка диапазона дат (3 месяца назад от текущей даты)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Услуги и их стоимость
    services = {
        'B01.047.001 Прием терапевта первичный': (2000, 3000),
        'B01.047.002 Прием терапевта повторный': (1500, 2500),
        'B01.057.001 Прием хирурга первичный': (2500, 3500),
        'B01.057.002 Прием хирурга повторный': (2000, 3000),
        'B01.023.001 Прием невролога первичный': (2500, 3500),
        'B01.023.002 Прием невролога повторный': (2000, 3000),
        'B01.029.001 Прием офтальмолога первичный': (2000, 3000),
        'B01.029.002 Прием офтальмолога повторный': (1500, 2500)
    }
    
    appointments = []
    doctor_ids = doctors_df['doctor_id'].tolist()
    cabinet_ids = cabinets_df['cabinet_id'].tolist()
    
    for i in range(num_appointments):
        service_name = random.choice(list(services.keys()))
        min_cost, max_cost = services[service_name]
        
        # Генерация случайной даты и времени в рабочие часы (с 8:00 до 21:00)
        random_date = start_date + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(8, 20),
            minutes=random.choice([0, 15, 30, 45])
        )
        
        # Форматируем дату в строку с пробелом между датой и временем
        formatted_date = random_date.strftime("%Y-%m-%d %H:%M:%S")
        
        appointment = {
            'appointment_id': f'app_{i+1}',
            'doctor_id': random.choice(doctor_ids),
            'cabinet_id': random.choice(cabinet_ids),
            'service_name': service_name,
            'appointment_date': formatted_date,
            'cost': random.randint(min_cost, max_cost),
            'is_dms': random.random() < 0.4  # 40% приемов по ДМС
        }
        appointments.append(appointment)
    
    appointments_df = pd.DataFrame(appointments)
    appointments_df = appointments_df.sort_values('appointment_date')
    return appointments_df

def generate_revenue(appointments_df):
    """Генерация отчета по доходам врачей на основе приемов"""
    # Группировка по врачу и месяцу
    revenue_data = []
    
    # Преобразуем дату в начало месяца для группировки
    appointments_df['month'] = appointments_df['appointment_date'].dt.to_period('M').dt.to_timestamp()
    
    # Группируем данные
    grouped = appointments_df.groupby(['doctor_id', 'month']).agg({
        'cost': 'sum',
        'appointment_id': 'count'
    }).reset_index()
    
    # Формируем отчет
    revenue_data = grouped.rename(columns={
        'cost': 'total_revenue',
        'appointment_id': 'appointments_count'
    })
    
    return revenue_data

def main():
    """Основная функция для генерации всех данных"""
    print("Генерация тестовых данных...")
    
    # Генерация данных
    doctors_df = generate_doctors(100)
    print(f"Сгенерировано {len(doctors_df)} записей о врачах")
    
    cabinets_df = generate_cabinets(60)
    print(f"Сгенерировано {len(cabinets_df)} записей о кабинетах")
    
    seasonal_df = generate_seasonal_coefficients()
    print(f"Сгенерировано {len(seasonal_df)} сезонных коэффициентов")
    
    promo_df = generate_promo_calendar()
    print(f"Сгенерировано {len(promo_df)} маркетинговых акций")
    
    appointments_df = generate_appointments(doctors_df, cabinets_df, 100000)
    print(f"Сгенерировано {len(appointments_df)} записей о приемах")
    
    # Преобразуем дату обратно в datetime для группировки
    appointments_df['appointment_date'] = pd.to_datetime(appointments_df['appointment_date'])
    
    revenue_df = generate_revenue(appointments_df)
    print(f"Сгенерировано {len(revenue_df)} записей о доходах")
    
    # Приводим булевы значения к нижнему регистру
    bool_columns = ['is_dms', 'is_star', 'dms_enabled', 'house_calls', 'sick_leave_enabled']
    for df in [doctors_df, appointments_df]:
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower()

    # Удаляем лишние колонки
    if 'month' in appointments_df.columns:
        appointments_df = appointments_df.drop(columns=['month'])

    # Заключаем строковые значения в кавычки
    for df in [doctors_df, cabinets_df, appointments_df]:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(lambda x: f'"{str(x)}"' if ',' in str(x) or ' ' in str(x) else x)

    # Сохранение данных в CSV
    doctors_df.to_csv('test_doctors.csv', index=False, quoting=1)  # quoting=1 для всех строковых значений
    cabinets_df.to_csv('test_cabinets.csv', index=False, quoting=1)
    appointments_df.to_csv('test_appointments.csv', index=False, quoting=1)
    revenue_df.to_csv('test_revenue.csv', index=False)
    seasonal_df.to_csv('test_seasonal.csv', index=False)
    promo_df.to_csv('test_promo.csv', index=False)
    
    print("\nТестовые данные успешно сохранены в файлы:")
    print("- test_doctors.csv")
    print("- test_cabinets.csv")
    print("- test_appointments.csv")
    print("- test_revenue.csv")
    print("- test_seasonal.csv")
    print("- test_promo.csv")

if __name__ == '__main__':
    main()