import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
from datetime import datetime, timedelta

class VisualizationManager:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_calendar_view(self, schedule, target_month):
        """Create calendar visualization of the schedule"""
        
        if not schedule:
            return self._create_empty_calendar()
        
        # Convert schedule to DataFrame
        schedule_df = pd.DataFrame(schedule)
        
        # Filter for target month
        month_schedules = []
        for _, row in schedule_df.iterrows():
            if hasattr(row['day'], 'month') and row['day'].month == target_month.month:
                month_schedules.append(row)
        
        if not month_schedules:
            return self._create_empty_calendar()
        
        schedule_month_df = pd.DataFrame(month_schedules)
        
        # Create calendar grid
        year = target_month.year
        month = target_month.month
        
        # Get calendar data
        cal = calendar.monthcalendar(year, month)
        
        # Count appointments per day
        daily_counts = schedule_month_df.groupby(
            schedule_month_df['day'].dt.day
        ).size().to_dict()
        
        # Create heatmap data
        calendar_data = []
        for week_num, week in enumerate(cal):
            for day_num, day in enumerate(week):
                if day == 0:
                    calendar_data.append({
                        'week': week_num,
                        'day_of_week': day_num,
                        'day': '',
                        'appointments': 0,
                        'text': ''
                    })
                else:
                    appointment_count = daily_counts.get(day, 0)
                    calendar_data.append({
                        'week': week_num,
                        'day_of_week': day_num,
                        'day': day,
                        'appointments': appointment_count,
                        'text': f'{day}<br>{appointment_count} записей'
                    })
        
        cal_df = pd.DataFrame(calendar_data)
        
        # Create calendar heatmap
        fig = go.Figure(data=go.Heatmap(
            x=cal_df['day_of_week'],
            y=cal_df['week'],
            z=cal_df['appointments'],
            text=cal_df['text'],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorscale='Viridis',
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Customize layout
        fig.update_layout(
            title=f'Календарь записей - {calendar.month_name[month]} {year}',
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(7)),
                ticktext=['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(cal))),
                ticktext=[f'Неделя {i+1}' for i in range(len(cal))],
                autorange='reversed'
            ),
            width=800,
            height=400
        )
        
        return fig
    
    def get_daily_schedule(self, schedule, selected_date):
        """Get detailed schedule for a specific day"""
        
        if not schedule:
            return pd.DataFrame()
        
        schedule_df = pd.DataFrame(schedule)
        
        # Filter for selected date
        daily_schedule = schedule_df[
            (schedule_df['day'].dt.date == selected_date) if 'day' in schedule_df.columns
            else pd.Series([False] * len(schedule_df))
        ].copy()
        
        if daily_schedule.empty:
            return pd.DataFrame()
        
        # Format for display
        daily_schedule['Время'] = daily_schedule['start_time'] + ' - ' + daily_schedule['end_time']
        daily_schedule['Врач ID'] = daily_schedule['doctor_id']
        daily_schedule['Кабинет ID'] = daily_schedule['cabinet_id']
        daily_schedule['Смена'] = daily_schedule['shift']
        daily_schedule['Услуга'] = daily_schedule.get('service', 'Не указана')
        daily_schedule['ДМС'] = daily_schedule.get('is_dms', False).map({True: 'Да', False: 'Нет'})
        
        return daily_schedule[['Время', 'Врач ID', 'Кабинет ID', 'Смена', 'Услуга', 'ДМС']].sort_values('Время')
    
    def analyze_doctor_workload(self, schedule, doctors_df):
        """Analyze workload distribution among doctors"""
        
        if not schedule:
            return pd.DataFrame()
        
        schedule_df = pd.DataFrame(schedule)
        
        # Calculate workload per doctor
        doctor_workload = schedule_df.groupby('doctor_id').agg({
            'start_time': 'count',  # Number of appointments
            'day': 'nunique'        # Number of working days
        }).reset_index()
        
        doctor_workload.rename(columns={
            'start_time': 'appointments_count',
            'day': 'working_days'
        }, inplace=True)
        
        # Calculate total hours (assuming 1 hour per appointment)
        doctor_workload['total_hours'] = doctor_workload['appointments_count']
        
        # Merge with doctor information
        doctor_workload = doctor_workload.merge(
            doctors_df[['doctor_id', 'name', 'specialty', 'experience_years']],
            on='doctor_id',
            how='left'
        )
        
        doctor_workload['doctor_name'] = doctor_workload['name'].fillna('Неизвестно')
        doctor_workload['specialty'] = doctor_workload['specialty'].fillna('Не указана')
        
        return doctor_workload.sort_values('total_hours', ascending=False)
    
    def analyze_cabinet_utilization(self, schedule, cabinets_df):
        """Analyze cabinet utilization patterns"""
        
        if not schedule:
            return pd.DataFrame()
        
        schedule_df = pd.DataFrame(schedule)
        
        # Add day of week information
        schedule_df['day_of_week'] = schedule_df['day'].dt.day_name()
        
        # Calculate utilization per cabinet per day of week
        cabinet_utilization = schedule_df.groupby(['cabinet_id', 'day_of_week']).size().reset_index()
        cabinet_utilization.rename(columns={0: 'appointments'}, inplace=True)
        
        # Merge with cabinet information
        cabinet_utilization = cabinet_utilization.merge(
            cabinets_df[['cabinet_id', 'name']],
            on='cabinet_id',
            how='left'
        )
        
        cabinet_utilization['cabinet_name'] = cabinet_utilization['name'].fillna('Неизвестно')
        
        # Calculate utilization rate (assuming 8 hour working day = 8 possible appointments)
        max_appointments_per_day = 8
        cabinet_utilization['utilization_rate'] = (
            cabinet_utilization['appointments'] / max_appointments_per_day * 100
        ).round(1)
        
        return cabinet_utilization
    
    def calculate_financial_impact(self, schedule, financial_metrics, demand_forecast):
        """Calculate financial impact of the schedule"""
        
        if not schedule:
            return {
                'total_revenue': 0,
                'demand_coverage': 0,
                'avg_doctor_utilization': 0
            }
        
        schedule_df = pd.DataFrame(schedule)
        
        # Calculate total potential revenue
        total_revenue = 0
        service_costs = {'default': 1500}  # Default service cost
        
        for _, appointment in schedule_df.iterrows():
            service = appointment.get('service', 'default')
            cost = service_costs.get(service, service_costs['default'])
            
            # Apply DMS multiplier
            if appointment.get('is_dms', False):
                cost *= 1.2
            
            total_revenue += cost
        
        # Calculate demand coverage
        total_predicted_demand = demand_forecast['predicted_demand'].sum() if not demand_forecast.empty else 1
        total_supplied = len(schedule_df)
        demand_coverage = min(1.0, total_supplied / total_predicted_demand) if total_predicted_demand > 0 else 0
        
        # Calculate average doctor utilization
        if not financial_metrics.empty:
            avg_utilization = financial_metrics['fill_rate'].mean()
        else:
            avg_utilization = 0.75  # Default assumption
        
        return {
            'total_revenue': total_revenue,
            'demand_coverage': demand_coverage,
            'avg_doctor_utilization': avg_utilization
        }
    
    def calculate_quality_metrics(self, schedule, doctors_df):
        """Calculate quality metrics for the schedule"""
        
        if not schedule:
            return {
                'reliability_score': 0,
                'strategy_score': 0,
                'personnel_balance': 0
            }
        
        schedule_df = pd.DataFrame(schedule)
        
        # Reliability score (based on doctor experience)
        doctor_experience = schedule_df.merge(
            doctors_df[['doctor_id', 'experience_years']],
            on='doctor_id',
            how='left'
        )
        
        avg_experience = doctor_experience['experience_years'].fillna(3).mean()
        reliability_score = min(1.0, avg_experience / 10)  # Normalize to 0-1
        
        # Strategy score (simplified)
        dms_ratio = schedule_df.get('is_dms', pd.Series([False] * len(schedule_df))).mean()
        strategy_score = min(1.0, dms_ratio * 2)  # Favor DMS appointments
        
        # Personnel balance (workload distribution)
        workload_distribution = schedule_df['doctor_id'].value_counts()
        balance_score = 1 - (workload_distribution.std() / workload_distribution.mean()) if len(workload_distribution) > 0 else 0
        personnel_balance = max(0, min(1, balance_score))
        
        return {
            'reliability_score': reliability_score,
            'strategy_score': strategy_score,
            'personnel_balance': personnel_balance
        }
    
    def analyze_demand_supply_balance(self, schedule, demand_forecast):
        """Analyze balance between demand and supply"""
        
        if demand_forecast.empty:
            return pd.DataFrame()
        
        # Aggregate demand by date
        demand_by_date = demand_forecast.groupby('date')['predicted_demand'].sum().reset_index()
        
        if not schedule:
            demand_by_date['supply'] = 0
        else:
            schedule_df = pd.DataFrame(schedule)
            
            # Aggregate supply by date
            schedule_df['date'] = schedule_df['day']
            supply_by_date = schedule_df.groupby('date').size().reset_index()
            supply_by_date.rename(columns={0: 'supply'}, inplace=True)
            
            # Merge demand and supply
            demand_supply = pd.merge(
                demand_by_date,
                supply_by_date,
                on='date',
                how='outer'
            ).fillna(0)
        
        demand_supply = demand_by_date
        demand_supply['supply'] = demand_supply['predicted_demand'] * 0.8  # Simplified supply calculation
        demand_supply.rename(columns={'predicted_demand': 'demand'}, inplace=True)
        
        return demand_supply
    
    def create_evolution_chart(self, evolution_history):
        """Create evolution progress chart"""
        
        if not evolution_history:
            return go.Figure()
        
        df = pd.DataFrame(evolution_history)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Эволюция приспособленности', 'Разнообразие популяции'),
            vertical_spacing=0.1
        )
        
        # Fitness evolution
        fig.add_trace(
            go.Scatter(
                x=df['generation'],
                y=df['best_fitness'],
                mode='lines+markers',
                name='Лучший результат',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['generation'],
                y=df['avg_fitness'],
                mode='lines',
                name='Средний результат',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        
        # Diversity (standard deviation)
        fig.add_trace(
            go.Scatter(
                x=df['generation'],
                y=df.get('std_fitness', [0] * len(df)),
                mode='lines',
                name='Разнообразие',
                line=dict(color='green', width=1)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            title_text="Прогресс генетического алгоритма",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Поколение", row=2, col=1)
        fig.update_yaxes(title_text="Приспособленность", row=1, col=1)
        fig.update_yaxes(title_text="Стандартное отклонение", row=2, col=1)
        
        return fig
    
    def create_workload_heatmap(self, schedule, doctors_df):
        """Create workload heatmap by doctor and day"""
        
        if not schedule:
            return go.Figure()
        
        schedule_df = pd.DataFrame(schedule)
        
        # Create pivot table
        schedule_df['day_name'] = schedule_df['day'].dt.strftime('%d.%m')
        workload_pivot = schedule_df.groupby(['doctor_id', 'day_name']).size().unstack(fill_value=0)
        
        # Merge with doctor names
        doctor_names = doctors_df.set_index('doctor_id')['name'].to_dict()
        workload_pivot.index = workload_pivot.index.map(lambda x: doctor_names.get(x, f'Doctor {x}'))
        
        # Create heatmap
        fig = px.imshow(
            workload_pivot.values,
            x=workload_pivot.columns,
            y=workload_pivot.index,
            color_continuous_scale='YlOrRd',
            aspect='auto',
            title='Загрузка врачей по дням'
        )
        
        fig.update_layout(
            xaxis_title="День",
            yaxis_title="Врач",
            height=max(400, len(workload_pivot) * 30)
        )
        
        return fig
    
    def _create_empty_calendar(self):
        """Create empty calendar placeholder"""
        
        fig = go.Figure()
        fig.add_annotation(
            text="Нет данных для отображения календаря",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="Календарь расписания",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=800,
            height=400
        )
        
        return fig
