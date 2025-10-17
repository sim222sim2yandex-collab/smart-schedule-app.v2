import pandas as pd
import numpy as np
from datetime import datetime
import re
import io

def parse_csv_line(line):
    """Parse a CSV line handling quotes correctly"""
    result = []
    current = ''
    in_quotes = False
    i = 0
    
    while i < len(line):
        char = line[i]
        
        # Пропускаем пробелы вне кавычек
        if char.isspace() and not in_quotes:
            i += 1
            continue
            
        if char == '"':
            # Проверяем на экранированные кавычки (две кавычки подряд)
            if i + 1 < len(line) and line[i + 1] == '"':
                current += '"'
                i += 2
                continue
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            # Очищаем значение от кавычек и пробелов
            cleaned = current.strip()
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1]
            result.append(cleaned.strip())
            current = ''
        else:
            current += char
        i += 1
    
    if current:
        # Очищаем последнее значение
        cleaned = current.strip()
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        result.append(cleaned.strip())
    
    return result

class DataProcessor:
    def __init__(self):
        self.required_columns = {
            'doctors': ['doctor_id', 'name', 'specialty', 'shift_type', 'experience_years'],
            'cabinets': ['cabinet_id', 'name', 'specialty_allowed', 'working_hours'],
            'appointments': ['appointment_id', 'doctor_id', 'cabinet_id', 'service_name', 'appointment_date', 'cost', 'is_dms'],
            'revenue': ['doctor_id', 'month', 'total_revenue', 'appointments_count'],
            'seasonal': ['season_id', 'month_number', 'specialty', 'seasonal_factor'],
            'promo': ['promo_id', 'start_date', 'end_date', 'specialty', 'promo_factor', 'promo_name']
        }
    
    def load_file(self, uploaded_file):
        """Load CSV or Excel file into DataFrame"""
        if uploaded_file is None:
            return pd.DataFrame()

        try:
            if uploaded_file.name.endswith('.csv'):
                # Читаем файл
                uploaded_file.seek(0)
                content = uploaded_file.read().decode('utf-8-sig')  # используем utf-8-sig для автоматического удаления BOM
                lines = content.strip().split('\n')
                
                if not lines or not lines[0].strip():
                    return pd.DataFrame() # Возвращаем пустой DataFrame, если файл пуст или содержит только пустые строки
                
                # Парсим заголовки всегда через parse_csv_line
                headers = parse_csv_line(lines[0])
                headers = [h.strip().lower() for h in headers]
                print("Обработанные заголовки:", headers)
                
                # Парсим данные
                data = []
                for line in lines[1:]:
                    if not line.strip():  # Пропускаем пустые строки
                        continue
                    
                    # Всегда используем parse_csv_line для корректной обработки кавычек
                    parts = parse_csv_line(line)
                    parts = [p.strip() for p in parts]
                    
                    if len(parts) == len(headers):
                        data.append(parts)
                    else:
                        print(f"Пропущена строка с неверным количеством полей: {line}")
                        print(f"Ожидалось {len(headers)} полей, получено {len(parts)}")
                
                # Создаем DataFrame
                df = pd.DataFrame(data, columns=headers)
                
                # Преобразуем числовые колонки
                pure_numeric_columns = ['experience_years', 'total_revenue', 'appointments_count', 'cost', 'seasonal_factor', 'promo_factor']
                for col in df.columns:
                    if col in pure_numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Преобразуем даты
                date_columns = ['appointment_date', 'start_date', 'end_date', 'month']
                for col in df.columns:
                    if col in date_columns:
                        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                
            else:
                df = pd.read_excel(uploaded_file)
            
            # Преобразование типов данных
            if 'cost' in df.columns:
                df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
            
            # Обработка булевых колонок
            bool_columns = ['is_dms', 'is_star', 'dms_enabled', 'house_calls', 'sick_leave_enabled']
            for col in df.columns:
                if col in bool_columns:
                    df[col] = df[col].map({
                        'True': True, 'False': False,
                        'true': True, 'false': False,
                        '1': True, '0': False,
                        1: True, 0: False,
                        True: True, False: False,
                        'да': True, 'нет': False,
                        'y': True, 'n': False,
                        'yes': True, 'no': False
                    }).fillna(False)
            
            if 'appointment_date' in df.columns:
                df['appointment_date'] = pd.to_datetime(df['appointment_date'])
                
            # Приведение пустых значений к None
            for col in df.columns:
                df[col] = df[col].replace({'пусто': None, '': None})
            
            print(f"\nЗагруженные данные:")
            print(f"Колонки (с учетом регистра): {list(df.columns)}")
            print(f"Колонки (нижний регистр): {[col.lower() for col in df.columns]}")
            print(f"Первые строки:\n{df.head()}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Ошибка при загрузке файла {uploaded_file.name}: {str(e)}")
    
    def validate_data_structure(self, doctors_df, cabinets_df, appointments_df, revenue_df, seasonal_df=None, promo_df=None):
        """Validate that all required columns are present"""
        validation_results = {'valid': True, 'errors': []}
        
        datasets = {
            'seasonal': seasonal_df if seasonal_df is not None else pd.DataFrame(),
            'promo': promo_df if promo_df is not None else pd.DataFrame()
        }
        
        # Проверяем только непустые датафреймы
        datasets = {k: v for k, v in datasets.items() if not v.empty}
        
        if not datasets:
            return validation_results
            
        # Словарь соответствия вариантов написания колонок
        column_mappings = {
            'appointment_id': ['appointment_id', 'appointmentid', 'appointment id', 'id'],
            'doctor_id': ['doctor_id', 'doctorid', 'doctor id', 'doc_id'],
            'cabinet_id': ['cabinet_id', 'cabinetid', 'cabinet id', 'cab_id'],
            'service_name': ['service_name', 'servicename', 'service', 'service id'],
            'appointment_date': ['appointment_date', 'appointmentdate', 'date', 'datetime'],
            'cost': ['cost', 'price', 'service_cost', 'servicecost'],
            'is_dms': ['is_dms', 'isdms', 'dms', 'insurance'],
            'name': ['name', 'full_name', 'doctor_name'],
            'specialty': ['specialty', 'specialization', 'spec'],
            'shift_type': ['shift_type', 'shift', 'work_shift'],
            'experience_years': ['experience_years', 'experience', 'years'],
            'specialty_allowed': ['specialty_allowed', 'allowed_specialties', 'specialties'],
            'working_hours': ['working_hours', 'work_hours', 'hours'],
            'month': ['month', 'date', 'period'],
            'total_revenue': ['total_revenue', 'revenue', 'income'],
            'appointments_count': ['appointments_count', 'count', 'visits']
        }
        
        for dataset_name, df in datasets.items():
            print(f"\nПроверка структуры таблицы '{dataset_name}':")
            print(f"Исходные колонки: {list(df.columns)}")
            
            # Проверяем наличие требуемых колонок
            missing_cols = []
            df_columns_lower = [col.lower() for col in df.columns]
            
            for required_col in self.required_columns[dataset_name]:
                required_col_lower = required_col.lower()
                variants = column_mappings.get(required_col_lower, [required_col_lower])
                variants = [v.lower() for v in variants]
                
                if not any(variant in df_columns_lower for variant in variants):
                    missing_cols.append(required_col)
            
            if missing_cols:
                validation_results['valid'] = False
                validation_results['errors'].append(
                    f"В справочнике '{dataset_name}' отсутствуют колонки: {', '.join(missing_cols)}"
                )
                
                print(f"\nПодсказка для '{dataset_name}':")
                print(f"Ожидаемые колонки: {self.required_columns[dataset_name]}")
                print(f"Найденные колонки: {list(df.columns)}")

        return validation_results
    
    def clean_data(self, doctors_df, cabinets_df, appointments_df, revenue_df, seasonal_df, promo_df):
        """Clean and preprocess all datasets"""
        cleaned_data = {
            'doctors': doctors_df,
            'cabinets': cabinets_df,
            'appointments': appointments_df,
            'revenue': revenue_df
        }
        
        # Очистка сезонных коэффициентов
        if seasonal_df is not None and not seasonal_df.empty:
            seasonal_df.columns = seasonal_df.columns.str.lower()
            for col in seasonal_df.select_dtypes(include=['object']).columns:
                seasonal_df[col] = seasonal_df[col].str.strip()
            if 'seasonal_factor' in seasonal_df.columns:
                seasonal_df['seasonal_factor'] = pd.to_numeric(seasonal_df['seasonal_factor'], errors='coerce')
            if 'month_number' in seasonal_df.columns:
                seasonal_df['month_number'] = pd.to_numeric(seasonal_df['month_number'], errors='coerce')
            cleaned_data['seasonal'] = seasonal_df

        # Очистка календаря акций
        if promo_df is not None and not promo_df.empty:
            promo_df.columns = promo_df.columns.str.lower()
            for col in promo_df.select_dtypes(include=['object']).columns:
                promo_df[col] = promo_df[col].str.strip()
            if 'start_date' in promo_df.columns:
                promo_df['start_date'] = pd.to_datetime(promo_df['start_date'])
            if 'end_date' in promo_df.columns:
                promo_df['end_date'] = pd.to_datetime(promo_df['end_date'])
            if 'promo_factor' in promo_df.columns:
                promo_df['promo_factor'] = pd.to_numeric(promo_df['promo_factor'], errors='coerce')
            cleaned_data['promo'] = promo_df

        return cleaned_data