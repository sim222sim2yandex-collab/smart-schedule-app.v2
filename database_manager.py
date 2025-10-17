import os
import pandas as pd
import sqlite3
from sqlalchemy import create_engine, text
from datetime import datetime
import numpy as np

class DatabaseManager:
    def __init__(self):
        """Initialize database connection and create tables"""
        self.database_path = 'clinic.db'
        self.engine = create_engine(f'sqlite:///{self.database_path}')
        self.conn = None
        self.create_tables()
    
    def get_connection(self):
        """Get database connection"""
        if not self.conn:
            self.conn = sqlite3.connect(self.database_path)
        return self.conn
    
    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create doctors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                doctor_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                specialty TEXT NOT NULL,
                shift_type TEXT,
                experience_years INTEGER,
                is_star BOOLEAN,
                dms_enabled BOOLEAN,
                house_calls BOOLEAN,
                sick_leave_enabled BOOLEAN,
                cabinet_binding TEXT
            )
        """)
        
        # Create cabinets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cabinets (
                cabinet_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                specialty_allowed TEXT,
                working_hours TEXT,
                branch_id TEXT
            )
        """)
        
        # Create appointments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                appointment_id TEXT PRIMARY KEY,
                doctor_id TEXT,
                cabinet_id TEXT,
                service_name TEXT,
                appointment_date TIMESTAMP,
                cost REAL,
                is_dms BOOLEAN,
                FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id),
                FOREIGN KEY (cabinet_id) REFERENCES cabinets(cabinet_id)
            )
        """)
        
        # Create revenue table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS revenue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_id TEXT,
                month DATE,
                total_revenue REAL,
                appointments_count INTEGER,
                FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
            )
        """)
        
        conn.commit()
    
    def insert_data(self, table_name, df):
        """Insert data from DataFrame into specified table"""
        if df is None or df.empty:
            return 0
            
        # Convert boolean columns to integers for SQLite
        bool_columns = df.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            df[col] = df[col].astype(int)
        
        # Use SQLAlchemy to handle the insertion
        df.to_sql(table_name, self.engine, if_exists='replace', index=False)
        return len(df)
    
    def get_data(self, table_name):
        """Retrieve all data from specified table"""
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, self.engine)
        
        # Convert integer boolean columns back to boolean
        if table_name == 'doctors':
            bool_cols = ['is_star', 'dms_enabled', 'house_calls', 'sick_leave_enabled']
        elif table_name == 'appointments':
            bool_cols = ['is_dms']
        else:
            bool_cols = []
            
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        return df
    
    def get_doctors(self):
        """Get all doctors"""
        return self.get_data('doctors')
    
    def get_cabinets(self):
        """Get all cabinets"""
        return self.get_data('cabinets')
    
    def get_appointments(self):
        """Get all appointments"""
        return self.get_data('appointments')
    
    def get_revenue(self):
        """Get all revenue data"""
        return self.get_data('revenue')
    
    def execute_query(self, query, params=None):
        """Execute custom SQL query"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor
        except Exception as e:
            conn.rollback()
            raise Exception(f"Error executing query: {str(e)}")
    
    def clear_table(self, table_name):
        """Clear all data from specified table"""
        self.execute_query(f"DELETE FROM {table_name}")
    
    def drop_table(self, table_name):
        """Drop specified table"""
        self.execute_query(f"DROP TABLE IF EXISTS {table_name}")
    
    def get_data_statistics(self):
        """Get statistics about data in database"""
        stats = {}
        
        tables = ['doctors', 'cabinets', 'appointments', 'revenue']
        for table in tables:
            count = self.execute_query(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[f'{table}_count'] = count
        
        # Get appointment date range
        date_range = self.execute_query(
            "SELECT MIN(appointment_date), MAX(appointment_date) FROM appointments"
        ).fetchone()
        stats['appointments_date_range'] = {
            'min': date_range[0],
            'max': date_range[1]
        } if date_range[0] else None
        
        return stats