from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('mushroom_classifier.h5')

def preprocess_image(image_path, img_size=256):
    img = cv2.imread(image_path)  # Загружаем изображение
    if img is None:
        raise ValueError("Ошибка загрузки изображения! Проверьте путь.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Перевод BGR → RGB
    img = cv2.resize(img, (img_size, img_size))  # Изменение размера
    img = img / 255.0  # Нормализация пикселей (если использовалось при обучении)
    img = np.expand_dims(img, axis=0)  # Добавляем размерность (1, 224, 224, 3)
    return img

for i in range(1,306):   
    image_path = f'C:/test_mush/{i}.jpg'  # Укажите ваш файл
    processed_img = preprocess_image(image_path)

    predictions = model.predict(processed_img)  # Получаем предсказание
    predicted_class = np.argmax(predictions)  # Индекс класса с максимальной вероятностью
    confidence = np.max(predictions)  # Уверенность модели в ответе

    print(f"Предсказанный класс: {predicted_class}, уверенность: {confidence:.2f}")