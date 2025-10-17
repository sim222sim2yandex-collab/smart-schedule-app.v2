import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import os

# Application modules
from data_processor import DataProcessor
from forecasting import DemandForecaster
from genetic_algorithm import ScheduleOptimizer
from visualization import VisualizationManager
from utils import ExportManager

# Configure page
st.set_page_config(
    page_title="Умное расписание врачей",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🏥 Умное расписание врачей")
    st.markdown("Система оптимизации расписания с использованием генетического алгоритма")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'forecasts_ready' not in st.session_state:
        st.session_state.forecasts_ready = False
    if 'optimization_complete' not in st.session_state:
        st.session_state.optimization_complete = False
    
    # Sidebar for navigation and parameters
    with st.sidebar:
        st.header("Навигация")
        phase = st.selectbox(
            "Выберите фазу",
            ["Фаза 1: Подготовка данных", "Фаза 2: Генерация расписаний", "Фаза 3: Оптимизация", "Результаты"]
        )
        
        st.header("Параметры алгоритма")
        population_size = st.slider("Размер популяции", 50, 500, 200)
        generations = st.slider("Количество поколений", 10, 100, 50)
        mutation_rate = st.slider("Вероятность мутации", 0.01, 0.3, 0.1)
        crossover_rate = st.slider("Вероятность скрещивания", 0.5, 0.9, 0.8)
        
        st.header("Веса критериев")
        weight_demand = st.slider("Вес спроса", 0.0, 1.0, 0.3)
        weight_revenue = st.slider("Вес выручки", 0.0, 1.0, 0.25)
        weight_reliability = st.slider("Вес надежности", 0.0, 1.0, 0.2)
        weight_strategy = st.slider("Вес стратегии", 0.0, 1.0, 0.15)
        weight_personnel = st.slider("Вес персонала", 0.0, 1.0, 0.1)
    
    # Main content area
    if phase == "Фаза 1: Подготовка данных":
        phase1_data_preparation()
    elif phase == "Фаза 2: Генерация расписаний":
        phase2_schedule_generation(population_size)
    elif phase == "Фаза 3: Оптимизация":
        phase3_optimization(population_size, generations, mutation_rate, crossover_rate,
                          weight_demand, weight_revenue, weight_reliability, weight_strategy, weight_personnel)
    else:
        results_visualization()

def phase1_data_preparation():
    st.header("Фаза 1: Подготовка данных и прогнозирование")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Загрузка справочников")
        
        # File uploaders for reference data
        doctors_file = st.file_uploader("Справочник врачей", type=['csv', 'xlsx'])
        cabinets_file = st.file_uploader("Справочник кабинетов", type=['csv', 'xlsx'])
        appointments_file = st.file_uploader("История записей", type=['csv', 'xlsx'])
        revenue_file = st.file_uploader("Отчет по доходам", type=['csv', 'xlsx'])
        seasonal_file = st.file_uploader("Справочник сезонных коэффициентов", type=['csv', 'xlsx'])
        promo_file = st.file_uploader("Календарь маркетинговых акций", type=['csv', 'xlsx'])
        
        if st.button("Загрузить и обработать данные"):
            if all([doctors_file, cabinets_file, appointments_file, revenue_file, seasonal_file, promo_file]):
                try:
                    processor = DataProcessor()
                    
                    # Process uploaded files
                    with st.spinner("Обработка данных..."):
                        doctors_df = processor.load_file(doctors_file)
                        cabinets_df = processor.load_file(cabinets_file)
                        appointments_df = processor.load_file(appointments_file)
                        revenue_df = processor.load_file(revenue_file)
                        seasonal_df = processor.load_file(seasonal_file)
                        promo_df = processor.load_file(promo_file)
                        
                        # Validate and clean data
                        validation_results = processor.validate_data_structure(
                            doctors_df, cabinets_df, appointments_df, revenue_df,
                            seasonal_df, promo_df
                        )
                        
                        if validation_results['valid']:
                            cleaned_data = processor.clean_data(
                                doctors_df, cabinets_df, appointments_df, revenue_df,
                                seasonal_df, promo_df
                            )
                            
                            # Store in session state
                            st.session_state.doctors_df = cleaned_data['doctors']
                            st.session_state.cabinets_df = cleaned_data['cabinets']
                            st.session_state.appointments_df = cleaned_data['appointments']
                            st.session_state.revenue_df = cleaned_data['revenue']
                            st.session_state.seasonal_df = cleaned_data['seasonal']
                            st.session_state.promo_df = cleaned_data['promo']
                            st.session_state.data_loaded = True
                            
                            st.success("Данные успешно загружены и обработаны!")
                        else:
                            st.error(f"Ошибка валидации данных: {validation_results['errors']}")
                            
                except Exception as e:
                    st.error(f"Ошибка при обработке данных: {str(e)}")
            else:
                st.warning("Пожалуйста, загрузите все четыре файла справочников")
    
    with col2:
        if st.session_state.data_loaded:
            st.subheader("Статистика загруженных данных")
            
            # Display data statistics
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Врачи", len(st.session_state.doctors_df))
                st.metric("Кабинеты", len(st.session_state.cabinets_df))
            with col2b:
                st.metric("Записи", len(st.session_state.appointments_df))
                st.metric("Отчеты доходов", len(st.session_state.revenue_df))
            with col2c:
                st.metric("Сезонные коэффициенты", len(st.session_state.seasonal_df))
                st.metric("Маркетинговые акции", len(st.session_state.promo_df))
            
            # Show sample data
            st.subheader("Предварительный просмотр данных")
            data_type = st.selectbox("Выберите справочник", 
                                   ["Врачи", "Кабинеты", "История записей", "Доходы", 
                                    "Сезонные коэффициенты", "Маркетинговые акции"])
            
            if data_type == "Врачи":
                st.dataframe(st.session_state.doctors_df.head())
            elif data_type == "Кабинеты":
                st.dataframe(st.session_state.cabinets_df.head())
            elif data_type == "История записей":
                st.dataframe(st.session_state.appointments_df.head())
            elif data_type == "Доходы":
                st.dataframe(st.session_state.revenue_df.head())
            elif data_type == "Сезонные коэффициенты":
                st.dataframe(st.session_state.seasonal_df.head())
            else:
                st.dataframe(st.session_state.promo_df.head())
    
    if st.session_state.data_loaded:
        st.subheader("Прогнозирование спроса")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("Параметры прогнозирования:")
            forecast_months = st.slider("Горизонт прогнозирования (месяцы)", 1, 6, 1)
            seasonal_coef = st.number_input("Коэффициент сезонности", 0.5, 2.0, 1.2)
            promo_coef = st.number_input("Коэффициент акций", 0.5, 2.0, 1.1)
            buffer_coef = st.number_input("Коэффициент буфера", 1.0, 1.5, 1.2)
            
            if st.button("Запустить прогнозирование"):
                try:
                    with st.spinner("Выполняется прогнозирование спроса..."):
                        forecaster = DemandForecaster()
                        
                        # Generate forecasts
                        demand_forecast = forecaster.forecast_demand(
                            st.session_state.appointments_df,
                            forecast_months,
                            seasonal_coef,
                            promo_coef,
                            buffer_coef
                        )
                        
                        financial_metrics = forecaster.calculate_financial_metrics(
                            st.session_state.revenue_df,
                            st.session_state.appointments_df
                        )
                        
                        # Store forecasts
                        st.session_state.demand_forecast = demand_forecast
                        st.session_state.financial_metrics = financial_metrics
                        st.session_state.forecasts_ready = True
                        
                        st.success("Прогнозирование завершено!")
                        
                except Exception as e:
                    st.error(f"Ошибка при прогнозировании: {str(e)}")
        
        with col4:
            if st.session_state.forecasts_ready:
                st.write("Результаты прогнозирования:")
                
                # Display forecast summary
                forecast_summary = st.session_state.demand_forecast.groupby('service').agg({
                    'predicted_demand': 'sum',
                    'dms_demand': 'sum'
                }).reset_index()
                
                st.dataframe(forecast_summary)
                
                # Visualization of forecast
                fig = px.line(
                    st.session_state.demand_forecast,
                    x='date',
                    y='predicted_demand',
                    color='service',
                    title='Прогноз спроса по услугам'
                )
                st.plotly_chart(fig, use_container_width=True)

def phase2_schedule_generation(population_size):
    st.header("Фаза 2: Генерация валидных расписаний")
    
    if not st.session_state.forecasts_ready:
        st.warning("Сначала необходимо завершить Фазу 1: Подготовка данных")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Параметры генерации")
        
        target_month = st.date_input("Целевой месяц", datetime.now().replace(day=1))
        
        # Hard constraints configuration
        st.write("Жесткие ограничения:")
        enforce_shifts = st.checkbox("Соблюдение смен", value=True, disabled=True)
        enforce_specializations = st.checkbox("Соответствие специализаций", value=True, disabled=True)
        enforce_star_schedules = st.checkbox("Фиксированные графики 'звезд'", value=True, disabled=True)
        enforce_cabinet_bindings = st.checkbox("Жесткие привязки к кабинетам", value=True, disabled=True)
        
        if st.button("Генерировать популяцию расписаний"):
            try:
                # Создаем контейнер для логов
                log_container = st.empty()
                logs = []
                
                # Создаем прогресс-бар
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                # Функция для обновления логов
                def update_log(message):
                    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
                    log_container.text_area("Логи процесса", "\n".join(logs), height=200)
                
                # Функция для обновления прогресса
                def update_progress(percent, text):
                    progress_bar.progress(int(percent) / 100.0)
                    progress_text.text(f"Прогресс: {text}")
                
                update_log("Начинаем генерацию расписаний...")
                update_progress(0, "Инициализация")
                
                from schedule_generator import ScheduleGenerator
                
                # Обновляем статус
                update_log("Инициализация генератора расписаний...")
                update_progress(10, "Подготовка данных")
                
                with st.spinner(f"Генерация {population_size} валидных расписаний..."):
                    update_log("Настройка генератора расписаний...")
                    update_progress(20, "Настройка генератора")
                    
                    generator = ScheduleGenerator(
                        st.session_state.doctors_df,
                        st.session_state.cabinets_df,
                        st.session_state.demand_forecast
                    )
                    
                    # Generate population
                    update_log("Генерация начальной популяции расписаний...")
                    update_progress(30, "Генерация популяции")
                    
                    population = []
                    
                    # Генерируем популяцию на основе ресурсов
                    generated_population = generator.generate_population(
                        population_size, target_month, 
                        enforce_shifts, enforce_specializations, 
                        enforce_star_schedules, enforce_cabinet_bindings
                    )
                    
                    if generated_population:
                        population.extend(generated_population)
                        update_log(f"Сгенерировано {len(population)} валидных расписаний на основе ресурсов.")
                    else:
                        update_log("Не удалось сгенерировать ни одного расписания на основе ресурсов.")

                    update_log("Проверка и сохранение результатов...")
                    update_progress(95, "Финализация")
                    
                    # Валидация всего расписания
                    if population and generator._is_valid_schedule(population[0]): # Проверяем первое расписание из популяции
                        st.session_state.population = population
                        st.session_state.population_generated = True
                        update_log(f"Успешно сгенерировано {len(population)} валидных расписаний.")
                        st.success(f"Успешно сгенерировано {len(population)} валидных расписаний.")
                    else:
                        st.session_state.population = []
                        st.session_state.population_generated = False
                        update_log("Не удалось сгенерировать валидное расписание. Проверьте ограничения и логи выше.")
                        st.warning("Не удалось сгенерировать валидное расписание.")

                    update_progress(100, "Завершено")
                    
            except Exception as e:
                st.error(f"Ошибка при генерации расписаний: {str(e)}")
    
    with col2:
        if hasattr(st.session_state, 'population_generated') and st.session_state.population_generated:
            st.subheader("Статистика популяции")
            
            # Population statistics
            st.metric("Размер популяции", len(st.session_state.population))
            
            # Sample chromosome visualization
            st.write("Пример расписания (хромосома #1):")
            if len(st.session_state.population) > 0:
                sample_schedule = st.session_state.population[0]
                
                # Convert to DataFrame for display
                schedule_df = pd.DataFrame([
                    {
                        'День': gene['day'],
                        'Врач': gene['doctor_id'],
                        'Кабинет': gene['cabinet_id'],
                        'Смена': gene['shift'],
                        'Часы': f"{gene['start_time']}-{gene['end_time']}"
                    }
                    for gene in sample_schedule[:10]  # Show first 10 genes
                ])
                
                st.dataframe(schedule_df)
                
                # Validation status
                st.write("Статус валидации:")
                validation_info = generator.validate_schedule(st.session_state.population[0])
                
                for constraint, status in validation_info.items():
                    if status:
                        st.success(f"✅ {constraint}")
                    else:
                        st.error(f"❌ {constraint}")

def phase3_optimization(population_size, generations, mutation_rate, crossover_rate,
                       weight_demand, weight_revenue, weight_reliability, weight_strategy, weight_personnel):
    st.header("Фаза 3: Генетическая оптимизация")
    
    if not hasattr(st.session_state, 'population_generated') or not st.session_state.population_generated:
        st.warning("Сначала необходимо завершить Фазу 2: Генерация расписаний")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Параметры оптимизации")
        
        # Display current parameters
        params_df = pd.DataFrame({
            'Параметр': ['Размер популяции', 'Поколения', 'Мутация', 'Скрещивание'],
            'Значение': [population_size, generations, mutation_rate, crossover_rate]
        })
        st.dataframe(params_df)
        
        # Display weights
        weights_df = pd.DataFrame({
            'Критерий': ['Спрос', 'Выручка', 'Надежность', 'Стратегия', 'Персонал'],
            'Вес': [weight_demand, weight_revenue, weight_reliability, weight_strategy, weight_personnel]
        })
        st.dataframe(weights_df)
        
        if st.button("Запустить оптимизацию"):
            optimize_schedules(population_size, generations, mutation_rate, crossover_rate,
                             weight_demand, weight_revenue, weight_reliability, weight_strategy, weight_personnel)
    
    with col2:
        if hasattr(st.session_state, 'evolution_history'):
            st.subheader("Прогресс оптимизации")
            
            # Evolution progress chart
            if len(st.session_state.evolution_history) > 0:
                evolution_df = pd.DataFrame(st.session_state.evolution_history)
                
                fig = px.line(
                    evolution_df,
                    x='generation',
                    y=['best_fitness', 'avg_fitness'],
                    title='Эволюция приспособленности'
                )
                fig.update_layout(
                    xaxis_title="Поколение",
                    yaxis_title="Приспособленность"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Current best fitness
                current_best = max(st.session_state.evolution_history, key=lambda x: x['best_fitness'])
                st.metric("Лучшая приспособленность", f"{current_best['best_fitness']:.4f}")
                
                # Convergence analysis
                if len(st.session_state.evolution_history) >= 10:
                    recent_improvements = [
                        st.session_state.evolution_history[i]['best_fitness'] - 
                        st.session_state.evolution_history[i-1]['best_fitness']
                        for i in range(-5, 0)
                    ]
                    avg_improvement = np.mean(recent_improvements)
                    
                    if avg_improvement < 0.001:
                        st.info("🔄 Алгоритм приближается к оптимуму")
                    else:
                        st.info("🚀 Продолжается активная оптимизация")

def optimize_schedules(population_size, generations, mutation_rate, crossover_rate,
                      weight_demand, weight_revenue, weight_reliability, weight_strategy, weight_personnel):
    """Run the genetic algorithm optimization"""
    
    try:
        optimizer = ScheduleOptimizer(
            st.session_state.doctors_df,
            st.session_state.cabinets_df,
            st.session_state.appointments_df,
            st.session_state.revenue_df,
            st.session_state.demand_forecast,
            st.session_state.financial_metrics
        )
        
        # Set fitness weights
        weights = {
            'demand': weight_demand,
            'revenue': weight_revenue,
            'reliability': weight_reliability,
            'strategy': weight_strategy,
            'personnel': weight_personnel
        }
        
        # Initialize progress tracking
        st.session_state.evolution_history = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run optimization
        with st.spinner("Выполняется генетическая оптимизация..."):
            best_schedule, evolution_stats = optimizer.optimize(
                st.session_state.population,
                generations,
                mutation_rate,
                crossover_rate,
                weights,
                callback=lambda gen, stats: update_progress(gen, generations, stats, progress_bar, status_text)
            )
            
            # Store results
            st.session_state.best_schedule = best_schedule
            st.session_state.evolution_stats = evolution_stats
            st.session_state.optimization_complete = True
            
            st.success("Оптимизация завершена успешно!")
            
            # Display final results
            final_fitness = evolution_stats[-1]['best_fitness']
            st.metric("Финальная приспособленность", f"{final_fitness:.4f}")
            
    except Exception as e:
        st.error(f"Ошибка при оптимизации: {str(e)}")

def update_progress(generation, total_generations, stats, progress_bar, status_text):
    """Update progress during optimization"""
    progress = generation / total_generations
    progress_bar.progress(progress)
    
    # Correctly access stats
    best_fitness = stats.get('max', 0.0)
    avg_fitness = stats.get('avg', 0.0)
    population_size = stats.get('population_size', 'N/A')
    invalid_individuals_count = stats.get('invalid_individuals_count', 'N/A')
    crossover_applied_count = stats.get('crossover_applied_count', 'N/A')
    mutation_applied_count = stats.get('mutation_applied_count', 'N/A')
    
    status_text.text(f"Поколение {generation}/{total_generations} - Лучший результат: {best_fitness:.4f} | Популяция: {population_size} | Недействительные: {invalid_individuals_count} | Скрещивания: {crossover_applied_count} | Мутации: {mutation_applied_count}")
    
    # Store evolution history
    if 'evolution_history' not in st.session_state:
        st.session_state.evolution_history = []
    
    st.session_state.evolution_history.append({
        'generation': generation,
        'best_fitness': best_fitness,
        'avg_fitness': avg_fitness,
        'population_size': population_size,
        'invalid_individuals_count': invalid_individuals_count,
        'crossover_applied_count': crossover_applied_count,
        'mutation_applied_count': mutation_applied_count
    })

def results_visualization():
    st.header("Результаты оптимизации")
    
    if not st.session_state.optimization_complete:
        st.warning("Сначала необходимо завершить оптимизацию в Фазе 3")
        return
    
    # Initialize visualization manager
    viz_manager = VisualizationManager()
    export_manager = ExportManager()
    
    # Tab structure for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Календарное расписание", 
        "👨‍⚕️ По врачам", 
        "🏢 По кабинетам", 
        "📊 Аналитика"
    ])
    
    with tab1:
        st.subheader("Календарное представление расписания")
        
        # Month selection
        target_month = st.session_state.target_month
        
        # Calendar visualization
        calendar_fig = viz_manager.create_calendar_view(
            st.session_state.best_schedule,
            target_month
        )
        st.plotly_chart(calendar_fig, use_container_width=True)
        
        # Daily schedule details
        selected_date = st.date_input("Выберите дату для детального просмотра", target_month)
        
        daily_schedule = viz_manager.get_daily_schedule(
            st.session_state.best_schedule,
            selected_date
        )
        
        if not daily_schedule.empty:
            st.dataframe(daily_schedule, use_container_width=True)
        else:
            st.info("На выбранную дату записей не найдено")
    
    with tab2:
        st.subheader("Распределение нагрузки по врачам")
        
        # Doctor workload analysis
        doctor_workload = viz_manager.analyze_doctor_workload(
            st.session_state.best_schedule,
            st.session_state.doctors_df
        )
        
        # Workload distribution chart
        fig_workload = px.bar(
            doctor_workload,
            x='doctor_name',
            y='total_hours',
            color='specialty',
            title='Загрузка врачей (часы в месяц)'
        )
        fig_workload.update_xaxes(tickangle=45)
        st.plotly_chart(fig_workload, use_container_width=True)
        
        # Doctor details table
        st.dataframe(doctor_workload, use_container_width=True)
    
    with tab3:
        st.subheader("Использование кабинетов")
        
        # Cabinet utilization analysis
        cabinet_utilization = viz_manager.analyze_cabinet_utilization(
            st.session_state.best_schedule,
            st.session_state.cabinets_df
        )
        
        # Utilization heatmap
        fig_util = px.imshow(
            cabinet_utilization.pivot(index='cabinet_name', columns='day_of_week', values='utilization_rate'),
            title='Загрузка кабинетов по дням недели (%)',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_util, use_container_width=True)
        
        # Cabinet details table
        st.dataframe(cabinet_utilization, use_container_width=True)
    
    with tab4:
        st.subheader("Детальная аналитика и KPI")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Финансовые показатели**")
            
            # Calculate financial metrics
            financial_analysis = viz_manager.calculate_financial_impact(
                st.session_state.best_schedule,
                st.session_state.financial_metrics,
                st.session_state.demand_forecast
            )
            
            st.metric("Прогнозируемая выручка", f"₽{financial_analysis['total_revenue']:,.0f}")
            st.metric("Покрытие спроса", f"{financial_analysis['demand_coverage']:.1%}")
            st.metric("Средняя загрузка врачей", f"{financial_analysis['avg_doctor_utilization']:.1%}")
            
        with col2:
            st.write("**Качественные показатели**")
            
            quality_metrics = viz_manager.calculate_quality_metrics(
                st.session_state.best_schedule,
                st.session_state.doctors_df
            )
            
            st.metric("Коэффициент надежности", f"{quality_metrics['reliability_score']:.3f}")
            st.metric("Стратегический балл", f"{quality_metrics['strategy_score']:.3f}")
            st.metric("Баланс персонала", f"{quality_metrics['personnel_balance']:.3f}")
        
        # Demand vs Supply analysis
        st.subheader("Анализ спроса и предложения")
        
        demand_supply = viz_manager.analyze_demand_supply_balance(
            st.session_state.best_schedule,
            st.session_state.demand_forecast
        )
        
        fig_demand = px.line(
            demand_supply,
            x='date',
            y=['demand', 'supply'],
            title='Баланс спроса и предложения'
        )
        st.plotly_chart(fig_demand, use_container_width=True)
        
        # Export options
        st.subheader("Экспорт результатов")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            if st.button("Экспорт в Excel"):
                excel_buffer = export_manager.export_to_excel(
                    st.session_state.best_schedule,
                    financial_analysis,
                    quality_metrics,
                    doctor_workload,
                    cabinet_utilization
                )
                
                st.download_button(
                    label="Скачать Excel отчет",
                    data=excel_buffer,
                    file_name=f"schedule_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col4:
            if st.button("Экспорт в CSV"):
                csv_buffer = export_manager.export_to_csv(st.session_state.best_schedule)
                
                st.download_button(
                    label="Скачать CSV файл",
                    data=csv_buffer,
                    file_name=f"schedule_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col5:
            if st.button("Экспорт аналитики"):
                analytics_buffer = export_manager.export_analytics(
                    financial_analysis,
                    quality_metrics,
                    st.session_state.evolution_history
                )
                
                st.download_button(
                    label="Скачать аналитику",
                    data=analytics_buffer,
                    file_name=f"analytics_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
