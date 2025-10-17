from data_processor import DataProcessor
from io import StringIO

def test_data_processor():
    dp = DataProcessor()
    
    # Создаем виртуальный файл с тестовыми данными
    class MockFile:
        def __init__(self, name, content):
            self.name = name
            self._content = content
            self._pos = 0
        
        def read(self):
            return self._content.encode('utf-8')
        
        def seek(self, pos):
            self._pos = pos

    # Тестируем загрузку файла doctors с расширенным набором колонок
    with open('test_doctors_full.csv', 'r') as f:
        test_content = f.read()
    
    mock_file = MockFile('test_doctors.csv', test_content)
    try:
        print("\nТестируем загрузку doctors:")
        df = dp.load_file(mock_file)
        print(df)
        print("\nТипы данных колонок:")
        print(df.dtypes)
    except Exception as e:
        print(f"Ошибка при тестировании doctors: {e}")
    
    # Тестируем загрузку пользовательского файла appointments
    with open('appointments_user.csv', 'r') as f:
        test_content = f.read()
    
    mock_file = MockFile('appointments.csv', test_content)
    try:
        print("\nТестируем загрузку пользовательского файла appointments:")
        appointments_df = dp.load_file(mock_file)
        print(appointments_df)
        print("\nТипы данных колонок:")
        print(appointments_df.dtypes)
        
        print("\nТестируем валидацию данных:")
        validation_results = dp.validate_data_structure(
            doctors_df=None,
            cabinets_df=None,
            appointments_df=appointments_df,
            revenue_df=None
        )
        print("Результаты валидации:", validation_results)
        
        if validation_results['valid']:
            print("✅ Валидация успешна")
        else:
            print("❌ Ошибки валидации:", validation_results['errors'])
    except Exception as e:
        print(f"Ошибка при тестировании appointments: {e}")

if __name__ == '__main__':
    test_data_processor()