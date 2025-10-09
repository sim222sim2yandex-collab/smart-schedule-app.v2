import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
warnings.filterwarnings('ignore')

class DemandForecaster:
    def __init__(self):
        self.models = {}
        self.seasonal_coefficients = {
            1: 1.2,  # January - winter peak
            2: 1.1,  # February
            3: 1.0,  # March
            4: 0.9,  # April
            5: 0.8,  # May
            6: 0.7,  # June - summer low
            7: 0.6,  # July - vacation period
            8: 0.7,  # August
            9: 1.1,  # September - autumn peak begins
            10: 1.3,  # October - ORVI season
            11: 1.5,  # November - peak ORVI
            12: 1.2   # December - holiday season
        }
        
    def forecast_demand(self, appointments_df, forecast_months, seasonal_coef, promo_coef, buffer_coef):
        """Generate demand forecast using Prophet"""
        
        # Prepare data for forecasting
        forecast_data = self._prepare_forecast_data(appointments_df)
        
        # Generate forecasts for each service
        service_forecasts = []
        
        for service in forecast_data['service_name'].unique():
            service_data = forecast_data[forecast_data['service_name'] == service]
            
            if len(service_data) >= 30:  # Minimum data points for reliable forecast
                try:
                    forecast = self._forecast_service_demand(
                        service_data, service, forecast_months, 
                        seasonal_coef, promo_coef, buffer_coef
                    )
                    service_forecasts.append(forecast)
                except Exception as e:
                    print(f"Warning: Could not forecast for service {service}: {str(e)}")
        
        if service_forecasts:
            return pd.concat(service_forecasts, ignore_index=True)
        else:
            # Return empty forecast with proper structure
            return pd.DataFrame(columns=[
                'service', 'date', 'predicted_demand', 'dms_demand', 
                'base_forecast', 'seasonal_factor', 'promo_factor', 'buffer_factor'
            ])
    
    def _prepare_forecast_data(self, appointments_df):
        """Prepare historical data for forecasting"""
        
        # Aggregate appointments by date and service
        daily_demand = appointments_df.groupby(['appointment_date', 'service_name']).agg({
            'appointment_id': 'count',
            'is_dms': 'sum'
        }).reset_index()
        
        daily_demand.rename(columns={
            'appointment_id': 'demand',
            'is_dms': 'dms_demand'
        }, inplace=True)
        
        return daily_demand
    
    def _forecast_service_demand(self, service_data, service_name, forecast_months, 
                                seasonal_coef, promo_coef, buffer_coef):
        """Forecast demand for a specific service using Prophet"""
        
        # Prepare Prophet format
        prophet_data = pd.DataFrame({
            'ds': service_data['appointment_date'],
            'y': service_data['demand']
        })
        
        # Create and fit Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative'
        )
        
        # Add custom seasonality for medical services
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Fit the model
        model.fit(prophet_data)
        
        # Create future dataframe
        future_dates = model.make_future_dataframe(periods=forecast_months * 30)
        
        # Generate forecast
        forecast = model.predict(future_dates)
        
        # Extract future periods only
        future_forecast = forecast.tail(forecast_months * 30).copy()
        
        # Apply business corrections
        future_forecast['base_forecast'] = future_forecast['yhat'].clip(lower=0)
        
        # Apply seasonal coefficient based on month
        future_forecast['month'] = pd.to_datetime(future_forecast['ds']).dt.month
        future_forecast['seasonal_factor'] = future_forecast['month'].map(self.seasonal_coefficients)
        future_forecast['seasonal_factor'] *= seasonal_coef
        
        # Apply promo coefficient (simplified - could be more sophisticated)
        future_forecast['promo_factor'] = promo_coef
        
        # Apply buffer coefficient
        future_forecast['buffer_factor'] = buffer_coef
        
        # Calculate final forecast
        future_forecast['predicted_demand'] = (
            future_forecast['base_forecast'] * 
            future_forecast['seasonal_factor'] * 
            future_forecast['promo_factor'] * 
            future_forecast['buffer_factor']
        ).round().astype(int)
        
        # Estimate DMS demand (based on historical proportion)
        historical_dms_ratio = service_data['dms_demand'].sum() / service_data['demand'].sum() if service_data['demand'].sum() > 0 else 0.2
        future_forecast['dms_demand'] = (future_forecast['predicted_demand'] * historical_dms_ratio).round().astype(int)
        
        # Prepare final output
        result = pd.DataFrame({
            'service': service_name,
            'date': future_forecast['ds'],
            'predicted_demand': future_forecast['predicted_demand'],
            'dms_demand': future_forecast['dms_demand'],
            'base_forecast': future_forecast['base_forecast'],
            'seasonal_factor': future_forecast['seasonal_factor'],
            'promo_factor': future_forecast['promo_factor'],
            'buffer_factor': future_forecast['buffer_factor']
        })
        
        return result
    
    def calculate_financial_metrics(self, revenue_df, appointments_df):
        """Calculate financial performance metrics for doctors"""
        
        # Calculate doctor performance metrics
        doctor_metrics = []
        
        for doctor_id in revenue_df['doctor_id'].unique():
            doctor_revenue = revenue_df[revenue_df['doctor_id'] == doctor_id]
            doctor_appointments = appointments_df[appointments_df['doctor_id'] == doctor_id]
            
            if len(doctor_revenue) > 0 and len(doctor_appointments) > 0:
                metrics = self._calculate_doctor_metrics(doctor_revenue, doctor_appointments)
                metrics['doctor_id'] = doctor_id
                doctor_metrics.append(metrics)
        
        return pd.DataFrame(doctor_metrics) if doctor_metrics else pd.DataFrame()
    
    def _calculate_doctor_metrics(self, doctor_revenue, doctor_appointments):
        """Calculate detailed metrics for a specific doctor"""
        
        # Basic financial metrics
        total_revenue = doctor_revenue['total_revenue'].sum()
        total_appointments = doctor_revenue['appointments_count'].sum()
        avg_appointment_cost = total_revenue / total_appointments if total_appointments > 0 else 0
        
        # Reliability metrics (consistency of performance)
        monthly_revenues = doctor_revenue['total_revenue']
        revenue_std = monthly_revenues.std()
        revenue_mean = monthly_revenues.mean()
        reliability_coefficient = 1 - (revenue_std / revenue_mean) if revenue_mean > 0 else 0
        reliability_coefficient = max(0, min(1, reliability_coefficient))  # Clamp to [0,1]
        
        # Fill rate calculation (theoretical capacity vs actual appointments)
        # Assume 8 hours/day, 20 days/month, 30 min per appointment = 320 theoretical appointments/month
        theoretical_capacity = len(doctor_revenue) * 320  # per month
        actual_appointments = total_appointments
        fill_rate = actual_appointments / theoretical_capacity if theoretical_capacity > 0 else 0
        fill_rate = min(1.0, fill_rate)  # Cap at 100%
        
        # Service diversity (number of different services provided)
        service_diversity = doctor_appointments['service_name'].nunique()
        
        # DMS ratio
        dms_appointments = doctor_appointments['is_dms'].sum()
        dms_ratio = dms_appointments / len(doctor_appointments) if len(doctor_appointments) > 0 else 0
        
        # Temporal patterns
        appointment_hours = doctor_appointments['appointment_date'].dt.hour
        peak_hour_ratio = (appointment_hours.between(16, 19)).sum() / len(appointment_hours) if len(appointment_hours) > 0 else 0
        
        return {
            'total_revenue': total_revenue,
            'total_appointments': total_appointments,
            'avg_appointment_cost': avg_appointment_cost,
            'reliability_coefficient': reliability_coefficient,
            'fill_rate': fill_rate,
            'service_diversity': service_diversity,
            'dms_ratio': dms_ratio,
            'peak_hour_ratio': peak_hour_ratio,
            'months_active': len(doctor_revenue)
        }
    
    def validate_forecast_quality(self, appointments_df, service_name, months_back=3):
        """Validate forecast quality using cross-validation"""
        
        service_data = appointments_df[appointments_df['service_name'] == service_name]
        
        if len(service_data) < 90:  # Need at least 3 months of data
            return {'error': 'Insufficient data for validation'}
        
        # Prepare data
        daily_demand = service_data.groupby('appointment_date').size().reset_index()
        daily_demand.columns = ['ds', 'y']
        
        # Create model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        try:
            model.fit(daily_demand)
            
            # Cross-validation
            df_cv = cross_validation(
                model, 
                horizon=f'{months_back * 30} days',
                period=f'{months_back * 15} days',
                initial=f'{months_back * 60} days'
            )
            
            # Performance metrics
            df_p = performance_metrics(df_cv)
            
            return {
                'mape': df_p['mape'].mean(),
                'rmse': df_p['rmse'].mean(),
                'mae': df_p['mae'].mean(),
                'coverage': df_p['coverage'].mean()
            }
            
        except Exception as e:
            return {'error': f'Validation failed: {str(e)}'}
    
    def get_demand_insights(self, forecast_df):
        """Generate insights from demand forecast"""
        
        insights = {}
        
        # Overall demand trends
        total_demand = forecast_df.groupby('date')['predicted_demand'].sum()
        insights['total_demand_trend'] = {
            'peak_day': total_demand.idxmax(),
            'peak_demand': total_demand.max(),
            'min_day': total_demand.idxmin(),
            'min_demand': total_demand.min(),
            'average_daily': total_demand.mean()
        }
        
        # Service-wise analysis
        service_analysis = forecast_df.groupby('service').agg({
            'predicted_demand': ['sum', 'mean', 'std'],
            'dms_demand': 'sum'
        }).round(2)
        
        service_analysis.columns = ['total_demand', 'avg_daily_demand', 'demand_volatility', 'total_dms']
        service_analysis['dms_ratio'] = service_analysis['total_dms'] / service_analysis['total_demand']
        
        insights['service_analysis'] = service_analysis.to_dict('index')
        
        # Seasonal patterns
        forecast_df['month'] = pd.to_datetime(forecast_df['date']).dt.month
        forecast_df['day_of_week'] = pd.to_datetime(forecast_df['date']).dt.dayofweek
        
        monthly_demand = forecast_df.groupby('month')['predicted_demand'].mean()
        weekly_demand = forecast_df.groupby('day_of_week')['predicted_demand'].mean()
        
        insights['seasonal_patterns'] = {
            'peak_month': monthly_demand.idxmax(),
            'low_month': monthly_demand.idxmin(),
            'peak_weekday': weekly_demand.idxmax(),
            'low_weekday': weekly_demand.idxmin()
        }
        
        return insights
