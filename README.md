# MEPHI-DustFinder
## Об алгоритме 🌐

Алгоритм был создан в качестве решения одной из задач хакатона "[Nuclear IT Hack](http://nuclearhack.mephi.ru/)". Алгоритм предназначен для анализа изображения в термоядерном реакторе. Задачей было распознать частицы радиоактивной пыли на картинке, посчитать их количество, определить размеры, составить таблицу и диаграмму.

## Использованные библиотеки 📚

- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [PIL](https://pypi.org/project/Pillow/)
- [csv](https://docs.python.org/3/library/csv.html)
- [time](https://docs.python.org/3/library/time.html)
- [cv2](https://pypi.org/project/opencv-python/)
- [pytesseract](https://pypi.org/project/pytesseract/)

## Перед запуском нужно ✨

Пользователь загружает фото пыли в папку `dust_photos` и указывает название конкретного фото, которое хочет обработать в строке 20:
```filename = "your_filename_here"```

⚠ В папку можно загружать только изображения определенного формата. Если хотите протестировать, то [отсюда](https://disk.yandex.ru/d/349goG_LDHexdw) можно скачать изображения.

Далее пользователь создаёт еще 2 папки: `dust_cropped`, в которую будут сохраняться обрезанные фото, и `dust_csv`, в которую будут сохраняться полученные таблицы.

Также в строке 52 пользователь должен указать минимальный размер частицы в микрометрах:
```min_area = 0```

⚠ *Размер* - диагональ квадрата, описанного вокруг окружности с такой же площадью, что и у пылинки

Установите все нужные библиотеки.

## Принцип работы алгоритма ⚒

1.  Загрузка изображения
2.  Обрезка изображения, считывание текста с фото
3.  Повышение контрастности (для 2 контуров), уменьшение шума
4.  Бинаризация изображения
5.  Поиск контуров с помощью OpenCV
6.  Нахождение размеров, координат и т.п. 
7.  Создание CSV файла (таблицы)
8.  Создание гистограммы
9.  Показ финального окна

## Результат 🧾

Откроется окно, на котором будет видно черно-белое фото, фото с обведенными контурами, гистограмма и немного текста.

![photo](https://lh5.googleusercontent.com/30etDSfeKcrrVa5snM8JWglIRIt8mlVXin0msiqnZMBKjKJOmDHbDdSZzZB7mbYZQRzQRmYi9_ssDWOhYZUM3EG55r8jAQZG5d16HCtTrhAi3erhpB5lFELUhpoqJ4695F3G_yVwDHXjFTMVY8oghAHbfQ=s2048)

Зеленым на фото показаны куски пыли, в которых алгоритм **уверен**. Красным - в которых **не уверен**.

Также в правом верхнем углу есть текст, где написан *размер* экрана в микрометрах, *коэффициент* (сколько микрометров в пикселе), *время работы* программы, *количество* частичек пыли, в которых алгоритм уверен и не уверен.

В папку `dust_csv` сохраняется таблица с координатами и размерами пылинок:

![csv](https://lh6.googleusercontent.com/Wzdg9wpD_NWPxQsJkOdcxzKwCX4-PIo1sPhnlJXbU0ExyUy_Z-JceJn8RgUel1urA2kxiLVIkqAq5XiOUPWC1bkVhHw0WkJa_yxzqcr4t8PMlxmKajmI59tLFc_C1iCVkX5YQoZXvDN5xGqGKUNLgqgc7Q=s2048)

⚠ Сохраняются только пылинки, в которых алгоритм **уверен**.

## Создатели ❤

Алгоритм создали [FoxFil](https://github.com/FoxFil) и [UltraGeoPro](https://github.com/Ultrageopro1966) :)
