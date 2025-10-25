# Создание видео из кадров
import os
import cv2
import subprocess
from pathlib import Path
import argparse


class VideoAssembler:
    def __init__(self):
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp'}

    def get_video_info(self, video_path):
        """Получает информацию о видео"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps

        cap.release()

        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration
        }

    def extract_audio(self, video_path, audio_output):
        """Извлекает аудио из видео"""
        try:
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-q:a', '0', '-map', 'a',
                str(audio_output), '-y'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка извлечения аудио: {e}")
            return False
        except FileNotFoundError:
            print("❌ FFmpeg не найден. Установите FFmpeg и добавьте в PATH")
            return False

    def find_frames(self, frames_dir):
        """Находит все кадры в директории"""
        frames_dir = Path(frames_dir)
        frames = []

        for ext in self.supported_image_formats:
            frames.extend(frames_dir.glob(f"*{ext}"))
            # frames.extend(frames_dir.glob(f"*{ext.upper()}"))

        # Сортируем по имени для правильной последовательности
        frames = sorted(frames, key=lambda x: x.name)
        return frames

    def create_video_from_frames(self, frames, output_path, fps, width, height):
        """Создает видео из кадров"""
        if not frames:
            print("❌ Не найдены кадры для создания видео")
            return False

        # Определяем кодек (H.264 для лучшей совместимости)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print(f"🎬 Создание видео: {len(frames)} кадров, {fps} FPS, {width}x{height}")

        for i, frame_path in enumerate(frames, 1):
            if i % 100 == 0:
                print(f"📹 Обработано кадров: {i}/{len(frames)}")

            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"⚠️  Не удалось загрузить кадр: {frame_path}")
                continue

            # Изменяем размер если необходимо
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            out.write(frame)

        out.release()
        return True

    def combine_video_audio(self, video_path, audio_path, output_path):
        """Объединяет видео и аудио"""
        try:
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-i', str(audio_path),
                '-c:v', 'copy',  # Копируем видео без перекодирования
                '-c:a', 'aac',  # Кодируем аудио в AAC
                '-shortest',  # Обрезаем до самой короткой дорожки
                str(output_path), '-y'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка объединения видео и аудио: {e}")
            return False

    def assemble_video(self, original_video, frames_dir, output_video):
        """Основная функция сборки видео"""
        print("🎬 Запуск сборки видео...")

        # Проверяем исходное видео
        if not Path(original_video).exists():
            print(f"❌ Исходное видео не найдено: {original_video}")
            return False

        # Получаем информацию о видео
        video_info = self.get_video_info(original_video)
        if not video_info:
            print(f"❌ Не удалось получить информацию о видео: {original_video}")
            return False

        print(f"📊 Информация о исходном видео:")
        print(f"   FPS: {video_info['fps']}")
        print(f"   Кадров: {video_info['frame_count']}")
        print(f"   Разрешение: {video_info['width']}x{video_info['height']}")
        print(f"   Длительность: {video_info['duration']:.2f} сек")

        # Находим кадры
        frames = self.find_frames(frames_dir)
        if not frames:
            print(f"❌ В директории {frames_dir} не найдены кадры")
            return False

        print(f"📁 Найдено кадров: {len(frames)}")

        # Создаем временные файлы
        temp_dir = Path("temp_video_files")
        temp_dir.mkdir(exist_ok=True)

        temp_video = temp_dir / "temp_video_no_audio.mp4"
        temp_audio = temp_dir / "temp_audio.wav"

        # Извлекаем аудио
        print("🔊 Извлечение аудио из исходного видео...")
        if not self.extract_audio(original_video, temp_audio):
            return False

        # Создаем видео из кадров
        print("🎞️ Создание видео из кадров...")
        if not self.create_video_from_frames(frames, temp_video, video_info['fps'],
                                             video_info['width'], video_info['height']):
            return False

        # Объединяем видео и аудио
        print("🔊 Объединение видео и аудио...")
        if not self.combine_video_audio(temp_video, temp_audio, output_video):
            return False

        # Проверяем результат
        if Path(output_video).exists():
            result_info = self.get_video_info(output_video)
            if result_info:
                print(f"✅ Видео успешно создано: {output_video}")
                print(f"📊 Итоговая информация:")
                print(f"   Длительность: {result_info['duration']:.2f} сек")
                print(f"   FPS: {result_info['fps']}")
                print(f"   Разрешение: {result_info['width']}x{result_info['height']}")

                # Очищаем временные файлы
                try:
                    temp_video.unlink()
                    temp_audio.unlink()
                    temp_dir.rmdir()
                except:
                    pass

                return True

        print("❌ Не удалось создать итоговое видео")
        return False


# Упрощенная версия без аргументов командной строки
def assemble(original_video, frames_dir, output_video):
    """Упрощенная версия с настройками по умолчанию"""
    assembler = VideoAssembler()
    print("🎬 Сборщик видео из кадров")
    print("=" * 50)

    # Проверяем существование файлов
    if not Path(original_video).exists():
        print(f"❌ Исходное видео не найдено: {original_video}")
        print("📝 Укажите правильный путь к видео")
        return False

    if not Path(frames_dir).exists():
        print(f"❌ Директория с кадрами не найдена: {frames_dir}")
        print("📝 Убедитесь что кадры обработаны и находятся в указанной папке")
        return False

    success = assembler.assemble_video(
        original_video=original_video,
        frames_dir=frames_dir,
        output_video=output_video
    )

    if success:
        print(f"🎉 Видео сохранено как: {output_video}")
        return True
    else:
        print("💥 Ошибка при создании видео")
        return False

