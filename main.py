filename = r'D:\machine learning\main.py'  # استخدام Raw String
try:
    with open(filename, 'rb') as file:
        data = file.read()
except FileNotFoundError:
    print(f"الملف {filename} غير موجود!")