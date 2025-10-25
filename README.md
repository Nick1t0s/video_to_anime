<div align="center">

# 🎬 Video to Anime Converter

**Преобразуйте ваши видео в аниме-стиль с помощью нейросетей!**  
*Этот проект использует модель AnimeGANv2 для обработки видео кадр за кадром*

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Возможности

🎥 **Извлечение кадров** из видео любого формата  
🎨 **Стилизация в аниме** с помощью AnimeGANv2  
🔊 **Сохранение аудиодорожки** из оригинального видео  
⚡ **Поддержка GPU** для ускорения обработки  
🔄 **Продолжение обработки** с места остановки  
📁 **Пакетная обработка** кадров

---

## 🚀 Быстрый старт

### 📋 Предварительные требования
- **Python 3.8 или выше**
- **FFmpeg** (для работы с видео и аудио)
- **NVIDIA GPU** (рекомендуется) или CPU

### ⚙️ Установка
```bash
git clone https://github.com/your-username/video-to-anime-converter.git
cd video-to-anime-converter
pip install -r requirements.txt
```

### 🎯 Использование
```bash
# Графический интерфейс
python main.py

# Командная строка
python main.py input_video.mp4 output_video.mp4
```

---

## 📁 Структура проекта
```
video-to-anime-converter/
├── main.py                 # Основной скрипт запуска
├── parse_video.py          # Извлечение кадров из видео
├── frame_to_anime.py       # Обработка кадров в аниме-стиль
├── create_video.py         # Сборка видео из кадров
├── requirements.txt        # Зависимости Python
└── README.md              # Документация
```

---

## 🔧 Настройка

**📹 Поддерживаемые форматы видео:** MP4, AVI, MOV, MKV, WMV, FLV, WebM  
**🖼️ Поддерживаемые форматы изображений:** JPG, JPEG, PNG, BMP, TIFF, WebP  
**⚙️ Параметры обработки:** Сохраняется оригинальное разрешение, FPS и аудио

---

## ⚡ Производительность

| Платформа | Скорость | Память |
|-----------|----------|---------|
| 🚀 GPU | ~0.1-0.5 сек/кадр | 2-4 GB VRAM |
| 🐢 CPU | ~1-3 сек/кадр | 1-2 GB RAM |

---

## 🛠️ Для разработчиков

```python
# Основной поток обработки
1. vid_to_frames()      # Разбивка видео на кадры
2. frame_to_anime()     # Стилизация кадров  
3. assemble()           # Сборка видео + аудио
```

---

## 🤝 Вклад в проект
1. 🍴 Форкните репозиторий
2. 🌿 Создайте feature branch
3. 💾 Закоммитьте изменения
4. 📤 Запушьте branch
5. 🔔 Откройте Pull Request

---

## ⚠️ Известные проблемы
- 💾 Обработка длинных видео требует много места на диске
- ⏳ На CPU обработка может занимать значительное время
- 🔧 Некоторые форматы видео могут требовать дополнительных кодеков

---

## 📝 Лицензия
**MIT License** - подробнее в файле `LICENSE`

---

## 🙏 Благодарности
- [AnimeGANv2](https://github.com/bryandlee/animegan2-pytorch) - модель для стилизации
- [FFmpeg](https://ffmpeg.org/) - работа с видео и аудио  
- [OpenCV](https://opencv.org/) - компьютерное зрение

---

### ⭐ Если вам понравился проект, поставьте звезду на GitHub!

*Сделано с ❤️ для сообщества аниме-энтузиастов*

📧 **Контакты:** your-email@example.com  
🐛 **Баги и предложения:** создавайте Issues на GitHub

</div>
