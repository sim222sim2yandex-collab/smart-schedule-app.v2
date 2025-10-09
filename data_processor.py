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
                # Try different encodings
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='cp1251')
            else:
                df = pd.read_excel(uploaded_file)
            
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
        
        for dataset_name, df in datasets.items():
            required_cols = self.required_columns[dataset_name]
            missing_cols = set(required_cols) - set(df.columns)
            
            if missing_cols:
                validation_results['valid'] = False
                validation_results['errors'].append(
                    f"В справочнике '{dataset_name}' отсутствуют колонки: {', '.join(missing_cols)}"
                )
        
        return validation_results
    
    def clean_data(self, doctors_df, cabinets_df, appointments_df, revenue_df):
        """Clean and standardize data"""
        
        # Clean doctors data
        doctors_cleaned = self._clean_doctors_data(doctors_df.copy())
        
        # Clean cabinets data
        cabinets_cleaned = self._clean_cabinets_data(cabinets_df.copy())
        
        # Clean appointments data
        appointments_cleaned = self._clean_appointments_data(appointments_df.copy())
        
        # Clean revenue data
        revenue_cleaned = self._clean_revenue_data(revenue_df.copy())
        
        return {
            'doctors': doctors_cleaned,
            'cabinets': cabinets_cleaned,
            'appointments': appointments_cleaned,
            'revenue': revenue_cleaned
        }
    
    def _clean_doctors_data(self, df):
        """Clean doctors DataFrame"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['doctor_id'])
        
        # Standardize names
        df['name'] = df['name'].str.strip().str.title()
        
        # Standardize specialties
        df['specialty'] = df['specialty'].str.strip().str.title()
        
        # Clean experience years
        df['experience_years'] = pd.to_numeric(df['experience_years'], errors='coerce')
        df['experience_years'] = df['experience_years'].fillna(0)
        
        # Standardize shift types
        shift_mapping = {
            'утро': 'morning',
            'вечер': 'evening',
            'день': 'day',
            'ночь': 'night'
        }
        df['shift_type'] = df['shift_type'].str.lower().map(shift_mapping).fillna(df['shift_type'])
        
        # Add additional columns if they exist
        optional_columns = ['is_star', 'dms_enabled', 'house_calls', 'sick_leave_enabled', 'cabinet_binding']
        for col in optional_columns:
            if col not in df.columns:
                if col == 'is_star':
                    df[col] = False
                elif col in ['dms_enabled', 'house_calls', 'sick_leave_enabled']:
                    df[col] = True
                else:
                    df[col] = None
        
        return df
    
    def _clean_cabinets_data(self, df):
        """Clean cabinets DataFrame"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['cabinet_id'])
        
        # Standardize names
        df['name'] = df['name'].str.strip().str.title()
        
        # Parse specialty_allowed (comma-separated)
        df['specialty_allowed'] = df['specialty_allowed'].apply(
            lambda x: [s.strip().title() for s in str(x).split(',') if s.strip()]
        )
        
        # Parse working hours
        df['working_hours'] = df['working_hours'].apply(self._parse_working_hours)
        
        return df
    
    def _clean_appointments_data(self, df):
        """Clean appointments DataFrame"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['appointment_id'])
        
        # Convert appointment_date to datetime
        df['appointment_date'] = pd.to_datetime(df['appointment_date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['appointment_date'])
        
        # Clean cost (remove zero-cost appointments as likely test data)
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df = df[df['cost'] > 0]
        
        # Standardize service names
        df['service_name'] = df['service_name'].str.strip().str.title()
        
        # Clean is_dms flag
        df['is_dms'] = df['is_dms'].astype(bool)
        
        # Add derived columns
        df['year'] = df['appointment_date'].dt.year
        df['month'] = df['appointment_date'].dt.month
        df['day_of_week'] = df['appointment_date'].dt.dayofweek
        df['hour'] = df['appointment_date'].dt.hour
        
        return df
    
    def _clean_revenue_data(self, df):
        """Clean revenue DataFrame"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['doctor_id', 'month'])
        
        # Convert month to datetime if it's string
        if df['month'].dtype == 'object':
            df['month'] = pd.to_datetime(df['month'], format='%Y-%m', errors='coerce')
        
        # Clean numeric columns
        df['total_revenue'] = pd.to_numeric(df['total_revenue'], errors='coerce')
        df['appointments_count'] = pd.to_numeric(df['appointments_count'], errors='coerce')
        
        # Remove rows with invalid data
        df = df.dropna(subset=['month', 'total_revenue', 'appointments_count'])
        
        # Calculate average appointment cost
        df['avg_appointment_cost'] = df['total_revenue'] / df['appointments_count']
        
        return df
    
    def _parse_working_hours(self, hours_str):
        """Parse working hours string into structured format"""
        try:
            # Expected format: "09:00-17:00" or "09:00-13:00,14:00-18:00"
            hours_str = str(hours_str).strip()
            periods = []
            
            for period in hours_str.split(','):
                period = period.strip()
                if '-' in period:
                    start, end = period.split('-')
                    periods.append({
                        'start': start.strip(),
                        'end': end.strip()
                    })
            
            return periods if periods else [{'start': '09:00', 'end': '17:00'}]
            
        except Exception:
            # Default working hours
            return [{'start': '09:00', 'end': '17:00'}]
    
    def get_data_quality_report(self, doctors_df, cabinets_df, appointments_df, revenue_df):
        """Generate data quality report"""
        
        report = {
            'doctors': {
                'total_records': len(doctors_df),
                'unique_doctors': doctors_df['doctor_id'].nunique(),
                'missing_specialties': doctors_df['specialty'].isnull().sum(),
                'specialties_count': doctors_df['specialty'].nunique()
            },
            'cabinets': {
                'total_records': len(cabinets_df),
                'unique_cabinets': cabinets_df['cabinet_id'].nunique(),
                'avg_specialties_per_cabinet': np.mean([len(x) for x in cabinets_df['specialty_allowed']])
            },
            'appointments': {
                'total_records': len(appointments_df),
                'date_range': {
                    'start': appointments_df['appointment_date'].min(),
                    'end': appointments_df['appointment_date'].max()
                },
                'unique_services': appointments_df['service_name'].nunique(),
                'dms_percentage': appointments_df['is_dms'].mean() * 100,
                'avg_cost': appointments_df['cost'].mean()
            },
            'revenue': {
                'total_records': len(revenue_df),
                'date_range': {
                    'start': revenue_df['month'].min(),
                    'end': revenue_df['month'].max()
                },
                'total_revenue': revenue_df['total_revenue'].sum(),
                'avg_monthly_revenue': revenue_df['total_revenue'].mean()
            }
        }
        
        return report
    
    def validate_referential_integrity(self, doctors_df, cabinets_df, appointments_df, revenue_df):
        """Check referential integrity between datasets"""
        
        integrity_issues = []
        
        # Check if all doctors in appointments exist in doctors table
        appointment_doctors = set(appointments_df['doctor_id'].unique())
        reference_doctors = set(doctors_df['doctor_id'].unique())
        missing_doctors = appointment_doctors - reference_doctors
        
        if missing_doctors:
            integrity_issues.append(f"Врачи в записях, но не в справочнике: {missing_doctors}")
        
        # Check if all cabinets in appointments exist in cabinets table
        appointment_cabinets = set(appointments_df['cabinet_id'].unique())
        reference_cabinets = set(cabinets_df['cabinet_id'].unique())
        missing_cabinets = appointment_cabinets - reference_cabinets
        
        if missing_cabinets:
            integrity_issues.append(f"Кабинеты в записях, но не в справочнике: {missing_cabinets}")
        
        # Check revenue data consistency
        revenue_doctors = set(revenue_df['doctor_id'].unique())
        missing_revenue_doctors = revenue_doctors - reference_doctors
        
        if missing_revenue_doctors:
            integrity_issues.append(f"Врачи в доходах, но не в справочнике: {missing_revenue_doctors}")
        
        return integrity_issues
