import pandas as pd
import numpy as np
from datetime import datetime
import re
import io

class DataProcessor:
    def __init__(self):
        self.required_columns = {
            'doctors': ['doctor_id', 'name', 'specialty', 'shift_type', 'experience_years'],
            'cabinets': ['cabinet_id', 'name', 'specialty_allowed', 'working_hours'],
            'appointments': ['appointment_id', 'doctor_id', 'cabinet_id', 'service_name', 'appointment_date', 'cost', 'is_dms'],
            'revenue': ['doctor_id', 'month', 'total_revenue', 'appointments_count']
        }
    
    def load_file(self, uploaded_file):
        """Load CSV or Excel file into DataFrame"""
        try:
            if uploaded_file.name.endswith('.csv'):
                # Читаем файл
                uploaded_file.seek(0)
                content = uploaded_file.read().decode('utf-8')
                lines = content.strip().split('\n')
                
                if not lines:
                    raise Exception("Файл пуст")
                
                # Функция для парсинга строк CSV
                def parse_csv_line(line):
                    result = []
                    current = ''
                    in_quotes = False
                    
                    for char in line:
                        if char == '"':
                            in_quotes = not in_quotes
                        elif char == ',' and not in_quotes:
                            result.append(current.strip().strip('"').strip("'"))
                            current = ''
                        else:
                            current += char
                    
                    if current:
                        result.append(current.strip().strip('"').strip("'"))
                    
                    return result
                
                # Парсим заголовки
                headers = parse_csv_line(lines[0])
                headers = [h.strip().strip('"').strip("'").lower() for h in headers]
                print("Обработанные заголовки:", headers)
                
                # Парсим данные
                data = []
                for line in lines[1:]:
                    if not line.strip():  # Пропускаем пустые строки
                        continue
                    
                    parts = parse_csv_line(line)
                    if len(parts) == len(headers):
                        data.append(parts)
                    else:
                        print(f"Пропущена строка с неверным количеством полей: {line}")
                        print(f"Ожидалось {len(headers)} полей, получено {len(parts)}")
                
                # Создаем DataFrame
                df = pd.DataFrame(data, columns=headers)
                
            else:
                df = pd.read_excel(uploaded_file)
            
            # Преобразование типов данных
            if 'cost' in df.columns:
                df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
            if 'is_dms' in df.columns:
                df['is_dms'] = df['is_dms'].map({'True': True, 'False': False, 
                                                'true': True, 'false': False,
                                                '1': True, '0': False,
                                                1: True, 0: False,
                                                True: True, False: False})
            if 'appointment_date' in df.columns:
                df['appointment_date'] = pd.to_datetime(df['appointment_date'])
            
            print(f"\nЗагруженные данные:")
            print(f"Колонки: {list(df.columns)}")
            print(f"Первые строки:\n{df.head()}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Ошибка при загрузке файла {uploaded_file.name}: {str(e)}")
    
    def validate_data_structure(self, doctors_df, cabinets_df, appointments_df, revenue_df):
        """Validate that all required columns are present"""
        validation_results = {'valid': True, 'errors': []}
        
        datasets = {
            'doctors': doctors_df,
            'cabinets': cabinets_df,
            'appointments': appointments_df,
            'revenue': revenue_df
        }
        
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
            for required_col in self.required_columns[dataset_name]:
                required_col_lower = required_col.lower()
                variants = column_mappings.get(required_col_lower, [required_col_lower])
                
                if not any(variant in df.columns for variant in variants):
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