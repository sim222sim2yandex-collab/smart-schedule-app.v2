import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from datetime import datetime
import numpy as np

class DatabaseManager:
    def __init__(self):
        self.database_url = os.environ.get('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")
        
        self.engine = create_engine(self.database_url)
        self.conn = None
    
    def get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(self.database_url)
        return self.conn
    
    def close_connection(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()
    
    def load_doctors_from_excel(self, df):
        """Load doctors data from DataFrame to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Clean column names
            df.columns = ['doctor_id', 'doctor_fio', 'branch_id', 'role_specialty', 
                         'required_cab_specs', 'available_nomenclature_ids', 'is_star',
                         'start_date', 'is_dms', 'is_mobile', 'issues_sick_leave',
                         'schedule_rule', 'standard_slot_time', 'vacation_dates',
                         'hard_binding_cab_id', 'soft_binding_cab_ids']
            
            # Remove rows with empty doctor_id
            df = df[df['doctor_id'].notna() & (df['doctor_id'] != '')]
            
            # Remove duplicate doctor_ids
            df = df.drop_duplicates(subset=['doctor_id'], keep='first')
            
            # Clean data
            df = df.fillna({
                'branch_id': '',
                'required_cab_specs': '',
                'available_nomenclature_ids': '',
                'schedule_rule': '',
                'vacation_dates': '',
                'hard_binding_cab_id': '',
                'soft_binding_cab_ids': ''
            })
            
            # Convert boolean fields
            df['is_star'] = df['is_star'].apply(lambda x: str(x).lower() in ['да', 'yes', 'true', '1'])
            df['is_dms'] = df['is_dms'].apply(lambda x: str(x).lower() in ['да', 'yes', 'true', '1'])
            df['is_mobile'] = df['is_mobile'].apply(lambda x: str(x).lower() in ['да', 'yes', 'true', '1'])
            df['issues_sick_leave'] = df['issues_sick_leave'].apply(lambda x: str(x).lower() in ['да', 'yes', 'true', '1'])
            
            # Convert date
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
            
            # Replace NaT with None for SQL NULL
            df['start_date'] = df['start_date'].apply(lambda x: None if pd.isna(x) else x)
            
            # Replace NaN with None for numeric fields
            df['standard_slot_time'] = df['standard_slot_time'].apply(lambda x: None if pd.isna(x) else x)
            
            # Insert data
            insert_query = """
                INSERT INTO doctors (
                    doctor_id, doctor_fio, branch_id, role_specialty, required_cab_specs,
                    available_nomenclature_ids, is_star, start_date, is_dms, is_mobile,
                    issues_sick_leave, schedule_rule, standard_slot_time, vacation_dates,
                    hard_binding_cab_id, soft_binding_cab_ids
                ) VALUES %s
                ON CONFLICT (doctor_id) DO UPDATE SET
                    doctor_fio = EXCLUDED.doctor_fio,
                    branch_id = EXCLUDED.branch_id,
                    role_specialty = EXCLUDED.role_specialty,
                    required_cab_specs = EXCLUDED.required_cab_specs,
                    available_nomenclature_ids = EXCLUDED.available_nomenclature_ids,
                    is_star = EXCLUDED.is_star,
                    start_date = EXCLUDED.start_date,
                    is_dms = EXCLUDED.is_dms,
                    is_mobile = EXCLUDED.is_mobile,
                    issues_sick_leave = EXCLUDED.issues_sick_leave,
                    schedule_rule = EXCLUDED.schedule_rule,
                    standard_slot_time = EXCLUDED.standard_slot_time,
                    vacation_dates = EXCLUDED.vacation_dates,
                    hard_binding_cab_id = EXCLUDED.hard_binding_cab_id,
                    soft_binding_cab_ids = EXCLUDED.soft_binding_cab_ids,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            # Convert to list and replace NaN/NaT with None
            values = df.replace({np.nan: None}).values.tolist()
            execute_values(cursor, insert_query, values)
            conn.commit()
            
            return len(df)
        
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error loading doctors: {str(e)}")
        finally:
            cursor.close()
    
    def load_cabinets_from_excel(self, df):
        """Load cabinets data from DataFrame to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Clean column names
            df.columns = ['cabinet_id', 'branch_id', 'cabinet_number', 'cabinet_specs', 'schedule_rule']
            
            # Remove rows with empty cabinet_id
            df = df[df['cabinet_id'].notna() & (df['cabinet_id'] != '')]
            
            # Remove duplicate cabinet_ids
            df = df.drop_duplicates(subset=['cabinet_id'], keep='first')
            
            # Clean data
            df = df.fillna({
                'branch_id': '',
                'cabinet_specs': '',
                'schedule_rule': ''
            })
            
            # Insert data
            insert_query = """
                INSERT INTO cabinets (
                    cabinet_id, branch_id, cabinet_number, cabinet_specs, schedule_rule
                ) VALUES %s
                ON CONFLICT (cabinet_id) DO UPDATE SET
                    branch_id = EXCLUDED.branch_id,
                    cabinet_number = EXCLUDED.cabinet_number,
                    cabinet_specs = EXCLUDED.cabinet_specs,
                    schedule_rule = EXCLUDED.schedule_rule,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            values = df.values.tolist()
            execute_values(cursor, insert_query, values)
            conn.commit()
            
            return len(df)
        
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error loading cabinets: {str(e)}")
        finally:
            cursor.close()
    
    def load_appointments_from_excel(self, df):
        """Load appointments data from DataFrame to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Clean column names
            df.columns = ['appointment_id', 'appointment_date', 'doctor_id', 'cabinet_id',
                         'slot_time', 'nomenclature_id', 'service_price', 'source_channel',
                         'sick_leave_issued']
            
            # Remove rows with empty appointment_id
            df = df[df['appointment_id'].notna() & (df['appointment_id'] != '')]
            
            # Remove duplicate appointment_ids
            df = df.drop_duplicates(subset=['appointment_id'], keep='first')
            
            # Clean data
            df = df.fillna({
                'nomenclature_id': '',
                'source_channel': '',
                'sick_leave_issued': False
            })
            
            # Convert date
            df['appointment_date'] = pd.to_datetime(df['appointment_date'], errors='coerce')
            df['appointment_date'] = df['appointment_date'].apply(lambda x: None if pd.isna(x) else x)
            
            # Convert boolean
            df['sick_leave_issued'] = df['sick_leave_issued'].apply(
                lambda x: str(x).lower() in ['да', 'yes', 'true', '1']
            )
            
            # Insert data
            insert_query = """
                INSERT INTO appointments (
                    appointment_id, appointment_date, doctor_id, cabinet_id, slot_time,
                    nomenclature_id, service_price, source_channel, sick_leave_issued
                ) VALUES %s
                ON CONFLICT (appointment_id) DO NOTHING
            """
            
            # Replace NaN/NaT with None
            values = df.replace({np.nan: None}).values.tolist()
            execute_values(cursor, insert_query, values)
            conn.commit()
            
            return len(df)
        
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error loading appointments: {str(e)}")
        finally:
            cursor.close()
    
    def load_doctor_revenue_from_excel(self, df):
        """Load doctor revenue data from DataFrame to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Clean column names
            df.columns = ['doctor_id', 'month', 'total_direct_income', 'total_referral_income']
            
            # Convert date
            df['month'] = pd.to_datetime(df['month'], errors='coerce')
            df['month'] = df['month'].apply(lambda x: None if pd.isna(x) else x)
            
            # Insert data
            insert_query = """
                INSERT INTO doctor_revenue (
                    doctor_id, month, total_direct_income, total_referral_income
                ) VALUES %s
                ON CONFLICT (doctor_id, month) DO UPDATE SET
                    total_direct_income = EXCLUDED.total_direct_income,
                    total_referral_income = EXCLUDED.total_referral_income
            """
            
            # Replace NaN/NaT with None
            values = df.replace({np.nan: None}).values.tolist()
            execute_values(cursor, insert_query, values)
            conn.commit()
            
            return len(df)
        
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error loading doctor revenue: {str(e)}")
        finally:
            cursor.close()
    
    def load_seasonal_coefficients_from_excel(self, df):
        """Load seasonal coefficients from DataFrame to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Clean column names
            df.columns = ['season_id', 'month_number', 'specialty', 'seasonal_factor', 'comment']
            
            # Remove rows with empty season_id
            df = df[df['season_id'].notna() & (df['season_id'] != '')]
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['season_id'], keep='first')
            
            # Convert seasonal_factor to numeric (handle datetime objects from Excel)
            def convert_seasonal_factor(val):
                if pd.isna(val):
                    return None
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, datetime):
                    # If it's a datetime, extract day/100 as coefficient (e.g. day 15 -> 1.5)
                    return val.day / 10.0
                try:
                    return float(val)
                except:
                    return 1.0  # Default coefficient
            
            df['seasonal_factor'] = df['seasonal_factor'].apply(convert_seasonal_factor)
            
            # Clean data
            df = df.fillna({'comment': ''})
            
            # Insert data
            insert_query = """
                INSERT INTO seasonal_coefficients (
                    season_id, month_number, specialty, seasonal_factor, comment
                ) VALUES %s
                ON CONFLICT (season_id) DO UPDATE SET
                    month_number = EXCLUDED.month_number,
                    specialty = EXCLUDED.specialty,
                    seasonal_factor = EXCLUDED.seasonal_factor,
                    comment = EXCLUDED.comment
            """
            
            values = df.values.tolist()
            execute_values(cursor, insert_query, values)
            conn.commit()
            
            return len(df)
        
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error loading seasonal coefficients: {str(e)}")
        finally:
            cursor.close()
    
    def load_promo_calendar_from_excel(self, df):
        """Load promo calendar from DataFrame to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Clean column names
            df.columns = ['promo_id', 'start_date', 'end_date', 'specialty', 'promo_factor', 'promo_name']
            
            # Remove rows with empty promo_id
            df = df[df['promo_id'].notna() & (df['promo_id'] != '')]
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['promo_id'], keep='first')
            
            # Convert promo_factor to numeric (handle datetime objects from Excel)
            def convert_promo_factor(val):
                if pd.isna(val):
                    return None
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, datetime):
                    # If it's a datetime, extract day/100 as coefficient
                    return val.day / 10.0
                try:
                    return float(val)
                except:
                    return 1.0  # Default coefficient
            
            df['promo_factor'] = df['promo_factor'].apply(convert_promo_factor)
            
            # Convert dates
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
            df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
            df['start_date'] = df['start_date'].apply(lambda x: None if pd.isna(x) else x)
            df['end_date'] = df['end_date'].apply(lambda x: None if pd.isna(x) else x)
            
            # Insert data
            insert_query = """
                INSERT INTO promo_calendar (
                    promo_id, start_date, end_date, specialty, promo_factor, promo_name
                ) VALUES %s
                ON CONFLICT (promo_id) DO UPDATE SET
                    start_date = EXCLUDED.start_date,
                    end_date = EXCLUDED.end_date,
                    specialty = EXCLUDED.specialty,
                    promo_factor = EXCLUDED.promo_factor,
                    promo_name = EXCLUDED.promo_name
            """
            
            # Replace NaN/NaT with None
            values = df.replace({np.nan: None}).values.tolist()
            execute_values(cursor, insert_query, values)
            conn.commit()
            
            return len(df)
        
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error loading promo calendar: {str(e)}")
        finally:
            cursor.close()
    
    def get_doctors(self):
        """Get all doctors from database"""
        query = "SELECT * FROM doctors ORDER BY doctor_id"
        return pd.read_sql(query, self.engine)
    
    def get_cabinets(self):
        """Get all cabinets from database"""
        query = "SELECT * FROM cabinets ORDER BY cabinet_id"
        return pd.read_sql(query, self.engine)
    
    def get_appointments(self):
        """Get all appointments from database"""
        query = "SELECT * FROM appointments ORDER BY appointment_date DESC"
        return pd.read_sql(query, self.engine)
    
    def get_doctor_revenue(self):
        """Get all doctor revenue from database"""
        query = "SELECT * FROM doctor_revenue ORDER BY doctor_id, month"
        return pd.read_sql(query, self.engine)
    
    def get_seasonal_coefficients(self):
        """Get all seasonal coefficients"""
        query = "SELECT * FROM seasonal_coefficients ORDER BY month_number"
        return pd.read_sql(query, self.engine)
    
    def get_promo_calendar(self):
        """Get all promo calendar entries"""
        query = "SELECT * FROM promo_calendar ORDER BY start_date"
        return pd.read_sql(query, self.engine)
    
    def get_data_statistics(self):
        """Get statistics about data in database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # Doctors count
            cursor.execute("SELECT COUNT(*) FROM doctors")
            stats['doctors_count'] = cursor.fetchone()[0]
            
            # Cabinets count
            cursor.execute("SELECT COUNT(*) FROM cabinets")
            stats['cabinets_count'] = cursor.fetchone()[0]
            
            # Appointments count
            cursor.execute("SELECT COUNT(*) FROM appointments")
            stats['appointments_count'] = cursor.fetchone()[0]
            
            # Revenue records count
            cursor.execute("SELECT COUNT(*) FROM doctor_revenue")
            stats['revenue_count'] = cursor.fetchone()[0]
            
            # Seasonal coefficients count
            cursor.execute("SELECT COUNT(*) FROM seasonal_coefficients")
            stats['seasonal_count'] = cursor.fetchone()[0]
            
            # Promo calendar count
            cursor.execute("SELECT COUNT(*) FROM promo_calendar")
            stats['promo_count'] = cursor.fetchone()[0]
            
            # Date ranges
            cursor.execute("SELECT MIN(appointment_date), MAX(appointment_date) FROM appointments")
            result = cursor.fetchone()
            stats['appointments_date_range'] = {'min': result[0], 'max': result[1]} if result[0] else None
            
            return stats
        
        finally:
            cursor.close()
    
    def clear_all_data(self):
        """Clear all data from database (for testing)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("TRUNCATE appointments, doctor_revenue CASCADE")
            cursor.execute("TRUNCATE doctors, cabinets CASCADE")
            cursor.execute("TRUNCATE seasonal_coefficients, promo_calendar CASCADE")
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error clearing data: {str(e)}")
        finally:
            cursor.close()
