from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('mushroom_classifier.h5')

def preprocess_image(image_path, img_size=256):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Ошибка загрузки изображения! Проверьте путь.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

image_path = 'test.jpg'
processed_img = preprocess_image(image_path)

predictions = model.predict(processed_img)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)

print(f"Предсказанный класс: {predicted_class}, уверенность: {confidence:.2f}")