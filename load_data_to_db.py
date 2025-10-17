#!/usr/bin/env python3
"""
Script to load data from Excel file into PostgreSQL database
"""
import pandas as pd
from database_manager import DatabaseManager
import sys
from data_processor import DataProcessor

def load_excel_to_database(excel_file_path):
    """Load all data from Excel file to database"""
    
    print(f"Loading data from: {excel_file_path}")
    
    try:
        db = DatabaseManager()
        processor = DataProcessor()
        print("✓ Connected to database")
        
        # Load Doctors
        print("\nLoading doctors...")
        doctors_df = pd.read_excel(excel_file_path, sheet_name='Справочник Врачи')
        doctors_count = db.load_doctors_from_excel(doctors_df)
        print(f"✓ Loaded {doctors_count} doctors")
        
        # Load Cabinets
        print("\nLoading cabinets...")
        cabinets_df = pd.read_excel(excel_file_path, sheet_name='Справочник Кабинеты')
        cabinets_count = db.load_cabinets_from_excel(cabinets_df)
        print(f"✓ Loaded {cabinets_count} cabinets")
        
        # Load Appointments
        print("\nLoading appointments history...")
        appointments_df = pd.read_excel(excel_file_path, sheet_name='История Записей')
        appointments_count = db.load_appointments_from_excel(appointments_df)
        print(f"✓ Loaded {appointments_count} appointments")
        
        # Load Doctor Revenue
        print("\nLoading doctor revenue...")
        revenue_df = pd.read_excel(excel_file_path, sheet_name='Отчет по доходам врачей (помеся')
        revenue_count = db.load_doctor_revenue_from_excel(revenue_df)
        print(f"✓ Loaded {revenue_count} revenue records")
        
        # Load Seasonal Coefficients
        print("\nLoading seasonal coefficients...")
        seasonal_df = pd.read_excel(excel_file_path, sheet_name='справочник сезонных коэффициент')
        seasonal_count = db.load_seasonal_coefficients_from_excel(seasonal_df)
        print(f"✓ Loaded {seasonal_count} seasonal coefficients")
        
        # Load Promo Calendar
        print("\nLoading promo calendar...")
        promo_df = pd.read_excel(excel_file_path, sheet_name=' календарь маркетинговых акций')
        promo_count = db.load_promo_calendar_from_excel(promo_df)
        print(f"✓ Loaded {promo_count} promo calendar entries")
        
        # Show statistics
        print("\n" + "="*50)
        print("DATABASE STATISTICS")
        print("="*50)
        stats = db.get_data_statistics()
        print(f"Doctors: {stats['doctors_count']}")
        print(f"Cabinets: {stats['cabinets_count']}")
        print(f"Appointments: {stats['appointments_count']}")
        print(f"Revenue Records: {stats['revenue_count']}")
        print(f"Seasonal Coefficients: {stats['seasonal_count']}")
        print(f"Promo Calendar Entries: {stats['promo_count']}")
        
        if stats['appointments_date_range']:
            print(f"\nAppointments Date Range:")
            print(f"  From: {stats['appointments_date_range']['min']}")
            print(f"  To: {stats['appointments_date_range']['max']}")
        
        print("\n✅ All data loaded successfully!")
        
        db.close_connection()
        
    except Exception as e:
        print(f"\n❌ Error loading data: {str(e)}")
        sys.exit(1)

def load_all_data_to_db():
    """Load all reference data from CSV files into the database"""
    
    db_manager = DatabaseManager()
    processor = DataProcessor()
    
    # File paths
    files = {
        'doctors': 'test_doctors.csv',
        'cabinets': 'test_cabinets.csv',
        'appointments': 'test_appointments.csv',
        'revenue': 'test_revenue.csv',
    }
    
    # Load and insert data
    for table_name, file_path in files.items():
        try:
            df = processor.load_file(file_path)
            
            # For main tables, insert into DB
            if table_name in ['doctors', 'cabinets', 'appointments', 'revenue']:
                count = db_manager.insert_data(table_name, df)
                print(f"Inserted {count} records into '{table_name}' from '{file_path}'")
            
        except FileNotFoundError:
            print(f"Warning: File not found at '{file_path}'. Skipping.")
        except Exception as e:
            print(f"Error loading data for '{table_name}': {str(e)}")
            
    print("Data loading process complete.")

if __name__ == "__main__":
    excel_file = "attached_assets/Структура данных для составления динамического расписания врачей_1759756400921.xlsx"
    load_excel_to_database(excel_file)
    load_all_data_to_db()
