# Обработка кадров
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys
from pathlib import Path


def check_environment():
    """Проверка окружения"""
    print("=" * 50)
    print("Проверка системы...")

    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA версия: {torch.version.cuda}")
    else:
        print("⚠️  GPU не обнаружена! Будет использоваться CPU (медленно)")

    print("=" * 50)


class FixedBatchAnimeGAN:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Поддерживаемые форматы изображений (в нижнем регистре)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def load_model(self):
        """Загрузка модели AnimeGANv2"""
        try:
            print("🔄 Загрузка модели AnimeGANv2...")
            model = torch.hub.load(
                'bryandlee/animegan2-pytorch:main',
                'generator',
                pretrained=True,
                device=self.device
            )
            self.model = model
            print("✅ Модель загружена успешно!")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

    def tensor_to_image(self, tensor):
        """Конвертирует тензор обратно в изображение"""
        tensor = tensor.squeeze(0)
        tensor = tensor * 0.5 + 0.5
        tensor = tensor.clamp(0, 1)
        image = tensor.cpu().detach().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)

    def convert_single_image(self, input_path, output_path):
        """Конвертация одного изображения в аниме-стиль"""
        if self.model is None:
            return False

        try:
            pil_image = Image.open(input_path).convert('RGB')
            original_size = pil_image.size
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output_tensor = self.model(input_tensor)

            result_image = self.tensor_to_image(output_tensor)
            result_image = result_image.resize(original_size, Image.LANCZOS)
            result_image.save(output_path, quality=95)
            return True
        except Exception as e:
            print(f"❌ Ошибка конвертации {input_path}: {e}")
            return False

    def find_images(self, directory):
        """Находит все изображения в директории (исправленная версия)"""
        directory = Path(directory)

        if not directory.exists():
            print(f"❌ Директория {directory} не существует!")
            return []

        # Используем более эффективный метод без дублирования
        image_paths = []
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_paths.append(file_path)

        return sorted(image_paths)

    def process_directory(self, input_dir, output_dir, start_from=0):
        """Обрабатывает все изображения в директории с возможностью продолжения"""
        if self.model is None:
            print("❌ Модель не загружена!")
            return False

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_paths = self.find_images(input_dir)

        if not image_paths:
            print(f"❌ В директории {input_dir} не найдено изображений!")
            return False

        print(f"📁 Найдено {len(image_paths)} изображений для обработки")
        print("=" * 50)

        success_count = 0
        failed_count = 0
        skipped_count = 0

        for i, image_path in enumerate(image_paths, 1):
            # Пропускаем уже обработанные, если указано начало
            if i < start_from:
                skipped_count += 1
                continue

            print(f"🔄 Обрабатывается {i}/{len(image_paths)}: {image_path.name}")

            output_file = output_path / f"{image_path.stem}_anime{image_path.suffix}"

            # Пропускаем если файл уже существует
            if output_file.exists():
                print(f"⏭️  Пропущено (уже существует): {output_file.name}")
                skipped_count += 1
                continue

            if self.convert_single_image(str(image_path), str(output_file)):
                print(f"✅ Успешно: {output_file.name}")
                success_count += 1
            else:
                print(f"❌ Ошибка: {image_path.name}")
                failed_count += 1

        # Выводим итоги
        print("=" * 50)
        print("🎉 Обработка завершена!")
        print(f"✅ Успешно: {success_count}")
        print(f"❌ Ошибок: {failed_count}")
        print(f"⏭️  Пропущено: {skipped_count}")
        print(f"📁 Результаты сохранены в: {output_dir}")

        return success_count > 0


def frame_to_anime(input_dir, output_dir):
    check_environment()
    gan = FixedBatchAnimeGAN()

    if gan.model is None:
        print("❌ Не удалось загрузить модель!")
        return
    Path(input_dir).mkdir(exist_ok=True)

    image_paths = gan.find_images(input_dir)
    if not image_paths:
        print(f"📁 Поместите изображения в папку '{input_dir}' и запустите скрипт снова")
        return

    # Проверяем, сколько файлов реально найдено
    print(f"🔍 Проверка файлов в {input_dir}:")
    for ext in gan.supported_formats:
        count = len(list(Path(input_dir).glob(f"*{ext}")))
        count_upper = len(list(Path(input_dir).glob(f"*{ext.upper()}")))
        if count > 0 or count_upper > 0:
            print(f"  {ext}: {count} файлов (в нижнем регистре), {count_upper} файлов (в верхнем регистре)")

    print(f"📊 Всего уникальных файлов: {len(image_paths)}")


    # Спрашиваем, нужно ли продолжить с определенного места

    print(f"🎨 Начинаем пакетную обработку...")
    success = gan.process_directory(input_dir, output_dir, start_from=0)

    if success:
        print("🎉 Все операции завершены успешно!")
        return True
    else:
        print("💥 Произошли ошибки при обработке")
        return False
