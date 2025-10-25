import cv2
import os
import argparse

# Извлечение кадров
def extract_all_frames(video_path, output_dir):
    """
    Извлекает все кадры из видеофайла.

    Args:
        video_path (str): Путь к исходному видеофайлу
        output_dir (str): Директория для сохранения кадров
    """
    # Проверяем существование видеофайла
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Видеофайл не найден: {video_path}")

    # Создаем директорию для кадров, если она не существует
    os.makedirs(output_dir, exist_ok=True)

    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Ошибка при открытии видеофайла")

    # Получаем информацию о видео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Информация о видео:")
    print(f"  Файл: {video_path}")
    print(f"  Разрешение: {width}x{height}")
    print(f"  Всего кадров: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Кадры сохраняются в: {output_dir}")
    print("Обработка...")

    saved_count = 0
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Сохраняем каждый кадр
        frame_filename = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
        success = cv2.imwrite(frame_filename, frame)

        if success:
            saved_count += 1
        else:
            print(f"Ошибка при сохранении кадра {saved_count}")

        # Показываем прогресс каждые 100 кадров
        if saved_count % 100 == 0:
            print(f"Обработано кадров: {saved_count}")

        frame_number += 1

    cap.release()

    print(f"\nГотово!")
    print(f"Успешно извлечено кадров: {saved_count}")

    if saved_count == 0:
        print("Предупреждение: не было извлечено ни одного кадра!")
        print("Возможные причины:")
        print("  - Неподдерживаемый формат видео")
        print("  - Поврежденный видеофайл")
        print("  - Проблемы с кодеками")


def vid_to_frames(video_path, output_dir):
    try:
        extract_all_frames(
            video_path=video_path,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"Ошибка: {e}")
        return False
    else:
        return True
