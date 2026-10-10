# RadOnc Segmentation (nnUnet INT8 openVino)

GUI-пайплайн автоматической сегментации органов риска головы и шеи по КТ:
импорт DICOM → конвертация в NIfTI → инференс нейросети (OpenVINO INT8, CPU) → экспорт RTSTRUCT для планирующей системы.
Плюс инструменты постпроцессинга масок, квантизации и сравнения метрик (Dice, HD95).

Если вы используете репозиторий, не забывайте ссылаться на:
```text
Залялов И.Ф. Сегментация органов риска головы и шеи при помощи свёрточных нейросетей.
Онкологический журнал: лучевая диагностика, лучевая терапия. 2026;9(3):19-26.
https://doi.org/10.37174/2587-7593-2026-9-3-19-26
```
```text
Zalyalov I.F. Head and Neck Organs at Risk Segmentation Using Convolutional Neural Networks.
Journal of oncology: diagnostic radiology and radiotherapy. 2026;9(3):19-26. (In Russ.)
https://doi.org/10.37174/2587-7593-2026-9-3-19-26
```

Проект навайбкожен при помощи GLM5.3 (Z.ai), Nemotron (Nvidia) и Алиса AI (Яндекс) - ошибки возможны, проект развивается (тизер - ожидается дистилляция модели).
В основе работы свёрточная нейросеть nnUnet (https://github.com/mic-dkfz/nnunet), обученная на своём датасете и квантизованная в INT8 openVINO (https://github.com/openvinotoolkit/openvino) для запуска на офисных компьютерах без GPU.

## Сегментация
Сегментация выполняется **ансамблем из 5 фолдов** nnUNet-модели, конвертированной в OpenVINO INT8, — целиком на **CPU, без GPU**: полный том ~18-20 минут на 4-8 логических ядрах. Сегментируется 31 структура (полный список — `radonc_segmentation/config.py`, словарь `LABELS`).

| Вкладка | Что делает |
|---|---|
| **Работа** | импорт DICOM пациента, запуск инференса, экспорт RTSTRUCT, watch-папка |
| **Постпроцессинг** | правка масок per-структурными методами (closing, dilate/erode, keep-largest…), Dice/HD95 против GT, шаблоны методов |
| **Квантизация** | список моделей, бенчмарк точности/скорости двух моделей |
| **Настройки** | устройство, потоки, пути, модель, значения по умолчанию |

Скриншоты — в папке [`_screenshots/`](_screenshots/).

## Быстрый старт (Windows)

Нужен установленный Python (3.10-3.12) — всё остальное
скрипт сделает сам:

```bat
git clone https://github.com/zalyalovi-arch/radonc_segmentation.git
cd radonc_segmentation
copy settings.example.ini settings.ini
python setup_env.py
```

`setup_env.py`:
1. Найдёт conda (Miniconda/Anaconda) в типовых местах; если нет — предложит скачать и установить Miniconda автоматически (спросит подтверждение).
2. Создаст окружение `nnunet_env` (Python 3.10).
3. Поставит зависимости из `requirements.txt` и сам пакет (`pip install -e .`, обычный pip, ~10-20 минут).
4. Проверит окружение и подстроит настройки под железо (потоки, накопитель).

Повторный запуск безопасен — готовые шаги пропускаются.

> **Проверка целостности (для клинических ПК):** `setup_env.py` скачивает Miniconda
> **зафиксированной версии** и сверяет SHA256 установщика до запуска установки —
> при расхождении настройка прерывается с понятным сообщением. Хеш запечатан в
> `setup_env.py`; при обновлении пина он пересчитывается из официального
> `repo.anaconda.com`.

> **Автоматизировать не удалось:** создать окружение без conda (venv) нельзя — `setup_env.py` опирается на conda и при её отсутствии предлагает автоматическую установку Miniconda (требует подтверждения в консоли — это единственный интерактивный шаг).

Если python не найден — GUI откроется, но при запуске инференса сообщит, чего не хватает.

Скачайте `nnunet_gui.exe` со страницы [Releases](https://github.com/zalyalovi-arch/radonc_segmentation/releases) и положите его в папку проекта (рядом с `settings.ini`).

### Запуск

```bat
run_gui.bat                     :: GUI из окружения (все вкладки)
nnunet_gui.exe                  :: exe (если собрали/скачали)
designer.bat                    :: Qt Designer для правки раскладок
python -m pytest tests          :: offscreen-проверка GUI + скриншоты
```

## Сегментация пациента

1. Вкладка **Работа** → укажите папку с экспортированной КТ-серией (DICOM) → **Импорт** (создаст `patients\<модель>\<ID>\` с NIfTI-томом).
2. **Запустить инференс** — результат: `patients\<модель>\<ID>\seg_nnunet\<ID>.nii.gz`.
3. **Экспорт RTSTRUCT** — готовый файл для планирующей системы рядом с томом.

Скриптом целиком:

```bat
python -m radonc_segmentation.pipeline P110   :: import → infer → RTSTRUCT
```

Тот же пайплайн доступен по watch-папке: кладёте подпапку с DICOM в `inbox\` — обработается автоматически (кнопки watch на вкладке «Работа»).

## Модель

Веса **не публикуются** в репозитории — папка `models\` пустая (см. `models/example.txt`). Ожидаемая структура:

```
models\
└── headandneck\              ← MODEL_NAME (settings.ini)
    └── int8\                 ← MODEL (settings.ini)
        ├── fold_0\quantized_model.xml
        ├── …
        └── fold_4\quantized_model.xml
```

После того как положили модель — GUI подхватит её сама (вкладка «Настройки»), или проверьте `MODEL`/`MODEL_NAME` в `settings.ini`.

### Обучение своей модели

Обучение nnUNet и подготовка весов — ручной этап, автоматизации нет. Требуется GPU и знания nnUNet v2. Состав шагов:

1. Обучить nnUNet v2 (5 фолдов) на своём датасете — стандартный процесс [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet).
2. Экспорт PyTorch → ONNX → OpenVINO IR:
   `python -m radonc_segmentation.quantization.openvino_converter`.
3. INT8-квантизация (NNCF): `python -m radonc_segmentation.quantization.quantize_ov`.
4. Разложить веса по фолдам в `models\<MODEL_NAME>\<MODEL>\fold_*\`.

Пакеты для этого: экстры квантизации
`python -m pip install -e .[quantization] --extra-index-url https://download.pytorch.org/whl/cpu`.

## Структура проекта

```
├── run_gui.bat                ← запуск GUI из окружения
├── build.bat                  ← пересборка nnunet_gui.exe (после правок кода)
├── designer.bat               ← Qt Designer на gui\ui
├── nnunet_gui.exe             ← готовый exe (Releases)
├── nnunet_gui_pyside.spec     ← рецепт сборки exe (PyInstaller)
├── settings.example.ini       ← шаблон настроек (копируйте в settings.ini)
├── requirements.txt           ← зависимости пайплайна и GUI
├── pyproject.toml             ← метаданные пакета (pip install -e ., экстры)
├── models\                    ← СЮДА кладутся веса (пустая, см. example.txt)
├── patients\                  ← данные пациентов (создаётся при импорте)
├── gui\                       ← код GUI (PySide6)
│   ├── main.py                ← точка входа (+ _gui_error.log при краше exe)
│   ├── main_window.py         ← логика главного окна
│   ├── postprocess_tab.py     ← логика вкладки «Постпроцессинг»
│   ├── workers.py             ← фоновые задачи (QThread + сигналы)
│   ├── mpl_canvas.py          ← matplotlib-канвас (колесо = срезы)
│   └── ui\*.ui                ← раскладки окон (правятся в Qt Designer)
├── radonc_segmentation\       ← пакет: пайплайн и инструменты
│   ├── pipeline.py            ← весь пайплайн одной командой
│   ├── import_dicom.py        ← DICOM → NIfTI (patients\<ID>\nnunet_input\)
│   ├── infer_ov.py            ← инференс OpenVINO (ансамбль 5 фолдов)
│   ├── export_rtstruct.py     ← копирование RTSTRUCT к исходному DICOM
│   ├── make_rtstruct_hn.py    ← сборка RTSTRUCT из маски (rt-utils)
│   ├── postprocess_core.py    ← методы постпроцессинга (использует GUI)
│   ├── compare_dice.py        ← метрики Dice/HD95 двух масок
│   ├── watch_folder.py        ← автообработка inbox-папки
│   ├── dicom_to_nifti.py      ← конвертация DICOM→NIfTI (отдельный вызов)
│   ├── validation_to_nifti.py ← подготовка валидационных кейсов
│   ├── config.py              ← настройки по умолчанию + settings.ini
│   ├── env_check.py           ← проверка окружения (используется при старте)
│   ├── check_devices.py       ← список устройств OpenVINO
│   ├── aligned\               ← вариант инференса, выровненный по обучению
│   ├── quantization\          ← экспорт/квантизация своей модели
│   ├── external_baselines\    ← сравнение с TotalSegmentator (справочно)
│   ├── postprocess_experiment.py ← исследование постпроцессинга (CLI)
│   ├── postprocess_viewer.py  ← старый tkinter-вьюер (справочник)
│   └── postprocess_templates\ ← JSON-шаблоны постпроцессинга
├── setup_env.py               ← настройка нового компьютера одной командой
└── tests\                     ← conftest.py + offscreen-проверка GUI
```

## Настройка

Все настройки — в `settings.ini` (создаётся из `settings.example.ini`)
или на вкладке «Настройки» GUI. Ключевое:

| Секция | Ключ | Смысл |
|---|---|---|
| `[device]` | `DEVICE` | `cpu` (основной вариант) или `gpu` |
| `[device]` | `NUM_THREADS` | логические ядра; подбирается автоматически при первом запуске |
| `[inference]` | `VARIANT` | `legacy` (дефолт) или `aligned` — варианты преобработки |
| `[inference]` | `CROP_AIR` | кроп воздуха — ускоряет инференс в 1.5-2 раза |
| `[inference]` | `ACCUMULATOR` | `disk` (надёжно) или `ram` (быстрее, нужно ~7 ГБ свободной ОЗУ) |
| `[model]` | `MODEL_NAME`, `MODEL` | подпапка модели: `models\<MODEL_NAME>\<MODEL>` |
| `[paths]` | `PATIENTS_ROOT` и др. | пусто = подпапки проекта (переносимо) |

Относительные/пустые пути считаются от папки проекта — проект можно копировать между компьютерами. При первом запуске на новом железе настройки (потоки, накопитель, python) подстраиваются автоматически, отпечаток машины запоминается в `settings.ini`.

## Сборка exe (не обязательна)

Готовый exe — в [Releases](https://github.com/zalyalovi-arch/radonc_segmentation/releases).
Собрать самому (после правок python-кода):

```bat
build.bat    :: ~4-5 минут; dist\nnunet_gui.exe → копия в корень
```

В exe упакованы PySide6, numpy/scipy, SimpleITK и исходники пакета `radonc_segmentation`. Тяжёлые ML-пакеты (torch, nnunetv2, matplotlib) исключены: инференс выполняет python окружения через subprocess. Из-за этого в exe недоступна вкладка «Постпроцессинг» (matplotlib) — используйте `run_gui.bat`.

Правки раскладок `gui\ui\*.ui` подхватываются перезапуском GUI без пересборки (`.ui` читаются при старте, не компилируются).

## Тесты

```bat
python -m pytest tests                      :: GUI собирается, вкладки работают
python -m radonc_segmentation.env_check     :: зависимости против requirements.txt
python -m radonc_segmentation.check_devices :: устройства OpenVINO
```

## Лицензия

[MIT](LICENSE). Веса моделей и данные пациентов не публикуются (медицинская приватность и лицензии исходного датасета).
