# RadOnc Segmentation (nnUnet INT8 openVino)

GUI-пайплайн автоматической сегментации органов риска головы и шеи по КТ:
импорт DICOM → конвертация в NIfTI → инференс нейросети (по умолчанию OpenVINO INT8 на CPU; опционально оригинальный nnUNet на NVIDIA GPU) → экспорт RTSTRUCT для планирующей системы.
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
| **Постпроцессинг** | правка масок per-структурными методами (closing, dilate/erode, keep-largest…), Dice/HD95 против GT, шаблоны методов, переключение сегментаций разных моделей на одних срезах |
| **Квантизация** | импорт обученного чекпоинта (.pth → OpenVINO IR), тест модели на валидации с гейтом качества, INT8/INT4-квантизация, список моделей, бенчмарк точности/скорости |
| **Настройки** | устройство, потоки, пути, модель, значения по умолчанию |

Скриншоты — в папке [`_screenshots/`](_screenshots/).

### Время инференса по вариантам модели

Все протестированные варианты модели (дистилляция, прунинг, квантизация, движки) и
время инференса; прочерк — комбинация не тестировалась. с/патч = среднее по 10
валидационным кейсам (патч 56×192×192, stride 0.5, LATENCY), мин/кейс — среднее того
же прогона. Железо — ноутбук i5-11300H, 8 потоков (детали профилей — в
[BENCHMARKS.md](BENCHMARKS.md)); исключения помечены.

| Вариант модели | Движок | Фолды | с/патч | мин/кейс |
|---|---|---|---|---|
| Учителя FP32 (нативный nnUNet, ансамбль) | PyTorch (Kaggle GPU) | 5 | — ⁵ | — ⁵ |
| Оригинал FP32 | OpenVINO IR | 5 | 4.7 ¹ | 24.1 ¹ |
| Production INT8 (ансамбль) | OpenVINO INT8 | 5 | 2.2 | 15.1 |
| Дистиллят FP32 | OpenVINO IR | 1 | 6.4 | 8.9 |
| **Дистиллят INT8** (прод-конфигурация) | OpenVINO INT8 | 1 | **3.4** | **6.0** |
| Pruned FP32 (one-shot + recovery) | OpenVINO IR | 1 | — ² | — |
| Pruned INT8 (RecoveryKD, эпоха 60) | OpenVINO INT8 | 1 | 4.8 | 8.2 |
| Prox FP32 (эксперимент прунинга) | ONNX Runtime | 1 | ≈15-22 ³ | 37-45 ³ |
| Дистиллят FP32 (нативный nnUNet) | PyTorch | 1 | — ⁴ | 404 |

¹ 1 кейс, legacy-вариант преобработки (10 кейсов не прогонялись).
² Локальный CPU не гонялся; на Kaggle CPU (4 vCPU) — 8.0 с/патч против 4.5 у INT8 того же прогона (×1.78).
³ Kaggle CPU 4 vCPU; плотная архитектура (prox зануляет каналы, но физически их не режет), 125-150 патчей/кейс.
⁴ Нативный движок с/патч не логирует; CPU-прогон — проверка эквивалентности движков (voxel-совпадение с OpenVINO 0.9856), не прод-путь (для GPU — раздел ниже).
⁵ Предсказания учителей получены на Kaggle GPU (T4), время инференса не замерялось.

Лучшее сочетание качества и скорости — **дистиллят INT8**: Dice 0.8394 при 6.0 мин/кейс
против 0.8398 / 15.1 мин у 5-фолдового Production INT8 (разница −0.0004). Полные таблицы
качества (Dice по 31 органу и по кейсам), методология и профили железа — в
[BENCHMARKS.md](BENCHMARKS.md).

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
3. Поставит зависимости из `requirements.txt` и сам пакет (`pip install -e .`, обычный pip, ~10-20 минут, ~1 ГБ на диске).
4. Поставит опциональные пакеты квантизации из `requirements-quantization.txt` (onnx, nncf и др., ~0.3 ГБ) и torch CPU-сборкой (`requirements-torch-cpu.txt`, ~0.8 ГБ) — нужны только для импорта/квантизации своей модели; сбой этого шага не фатален.
5. При наличии NVIDIA GPU спросит, поставить ли пакеты GPU-инференса (`requirements-gpu.txt`: torch CUDA + nnunetv2, ~3.5 ГБ) — только по явному согласию: torch переустанавливается с CPU- на CUDA-сборку, квантизация продолжает работать. На машинах без NVIDIA шаг 4.6 по согласию ставит стек нативного движка для CPU-проверок (`requirements-native.txt`). Подробнее — раздел «Движки инференса и профили установки».
6. Проверит окружение и подстроит настройки под железо (потоки, накопитель).

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

### Добавление своей модели (обучение/дистилляция) через GUI

Единственный ручной шаг — обучить модель и скачать чекпоинт (например, после
дистилляции на Kaggle: Output ядра → `checkpoint_best.pth`, рядом полезен
`plans.json` из того же Output). Дальше всё в GUI, вкладка **Квантизация**:

| Шаг | Действие в программе |
|---|---|
| 0-1. Импорт → OpenVINO IR (FP32) | Группа «Импорт обученной модели»: укажите `checkpoint_best.pth` (или `.zip`/папку из Output), имя модели (напр. `distilled`) → **Импортировать и экспортировать**. Результат: `models\<анатомия>\<имя>\fold_*\model.xml`. |
| 2-3. Тест FP32 + гейт качества | Группа «Тест модели на валидации»: модель `<анатомия>/<имя>`, эталон — рабочая модель, валидационный кейс → **Запустить тест**. Гейт: Dice(mean) ≥ 0.8485, малые структуры (Lens/Cochlea/TMJ) не хуже −0.03 относительно эталона. |
| 4. INT8-квантизация | Группа «Параметры квантизации»: источник — папка импортированной модели, имя `headandneck/distilled_int8` → **Квантизовать**. |
| 5. Тест INT8 | Ещё раз «Тест модели на валидации» с моделью `headandneck/distilled_int8`. |
| 6. Визуальная проверка | Вкладка **Постпроцессинг** → кейс `[валидация]` → в списке «Сегментация» переключайте модели (distilled / int8) и сравнивайте контуры с GT на срезах — особенно мелкие структуры. |
| 7. Внедрение | Бенчмарк скорости (внизу вкладки «Квантизация»), затем вкладка **Настройки** → `MODEL` = `headandneck/distilled_int8` → **Сохранить**. |

Тот же поток скриптами (без GUI):

```bat
python -m radonc_segmentation.quantization.import_checkpoint --src checkpoint_best.pth --name distilled --folds 0
python -m radonc_segmentation.quantization.validate_model --model headandneck/distilled --reference headandneck/original --case validation_DICOM_0001 --folds 0
python -m radonc_segmentation.quantization.quantize_ov --source models\headandneck\distilled --name headandneck/distilled_int8 --mode int8
```

Пакеты импорта/квантизации ставит `setup_env.py` (шаг 4); вручную:
`python -m pip install -r requirements-quantization.txt`. Torch в этот
набор не входит — он ставится отдельным шагом 4.1 (CPU-сборка, файл
`requirements-torch-cpu.txt`) и нужен в любой сборке: CPU на машинах
без NVIDIA, CUDA на машинах с NVIDIA (см. ниже).

## Движки инференса и профили установки

Инференс идёт одним из двух движков — ключ `[inference] ENGINE` в
`settings.ini` (или комбо «Движок» на вкладке «Работа»):

| Движок | Что делает | Устройство |
|---|---|---|
| `openvino` (по умолчанию) | INT8-IR, своя преобработка | CPU (или Intel GPU: `[device] DEVICE = gpu`) |
| `nnunet_native` | оригинальный nnUNet (PyTorch) | NVIDIA GPU; на CPU — только проверка эквивалентности (`--allow-cpu`, 65+ мин/кейс) |

Torch нужен обоим опциональным сценариям (квантизация и `nnunet_native`),
но ставится **один раз в одной сборке под железо** — переключение движков
не требует переустановок:

| Профиль машины | Что ставить | Что получается |
|---|---|---|
| Без NVIDIA (типовой офисный ПК) | `setup_env.py` ставит сам: квантизация + `requirements-torch-cpu.txt` | OpenVINO CPU + квантизация/импорт .pth; нативный движок — по желанию (шаг 4.6 спросит) |
| С NVIDIA GPU | `setup_env.py` + на шаге 4.5 согласиться (или вручную `pip install -r requirements-gpu.txt`) | Всё то же + `nnunet_native` на GPU; CUDA-сборка torch покрывает и квантизацию |

## GPU-инференс (нативный nnUNet, NVIDIA)

Помимо основного OpenVINO-движка (CPU, INT8) программа умеет считать
**оригинальным nnUNet** (`nnUNetv2Predictor`, PyTorch) — движок
`nnunet_native`. Это тот же путь, которым модель обучалась: nnUNet сам
делает преобработку, sliding window, ансамбль фолдов и TTA, результат —
эталонное качество, на GPU заметно быстрее CPU-инференса.

**Кому это нужно.** Пользователям с NVIDIA GPU (у офисных машин без
дискретной видеокарты остаётся дефолтный openvino-путь — он не менялся).
Сборка torch в окружении одна: `requirements-gpu.txt` (CUDA) заменяет
собой `requirements-torch-cpu.txt`; пакеты квантизации с обеими сборками
совместимы, переустановка ничего не ломает.

### Установка на машину с NVIDIA GPU

```bat
copy settings.example.ini settings.ini
python setup_env.py                  :: базовые пакеты; на шаге про GPU согласитесь
python -m pip install -r requirements-gpu.txt   :: если шаг 4.5 пропустили: torch CUDA + nnunetv2 (~3.5 ГБ)
```

### Установка нативного движка на машину БЕЗ NVIDIA (только проверки)

По умолчанию не ставится. Если нужно сравнить движки или провалидировать
.pth-модель на CPU-машине (65+ минут на кейс), доустановите стек поверх
torch-CPU: `python -m pip install -r requirements-native.txt`
(setup_env.py шаг 4.6 предложит это сам).

### Веса

Чекпоинты в репозиторий не входят. Модель нужна в виде **.pth** (то, что
скачивается из Kaggle Output после обучения/дистилляции). Импортируйте её
как обычно — GUI: «Квантизация» → «Импорт модели», или

```bat
python -m radonc_segmentation.quantization.import_checkpoint --src checkpoint_final.pth --name distilled
```

Импорт кладёт чекпоинты в `models\_src\<имя>\` и экспортирует OpenVINO IR.
Нативному движку IR не нужен — он работает прямо по стейджингу `_src`
(весы не копируются).

### Включение

- GUI, вкладка «Работа»: комбо **Движок** → `nnunet_native` (опции
  кропа/stride/async при этом не применимы — их делает сам nnUNet);
- или `settings.ini` → `[inference] ENGINE = nnunet_native`;
- или разово: `python -m radonc_segmentation.pipeline P110 --engine nnunet_native`.

Без CUDA нативный запуск блокируется с объяснением (нативный nnUNet на
CPU — 65+ минут на кейс против ~6 минут у INT8). Для проверки
эквивалентности движков добавьте `--allow-cpu`; GUI при выборе
`nnunet_native` без CUDA сам спросит «Запустить на CPU?» и подставит
флаг. GUI при старте пишет в лог, доступна ли CUDA.

### Проверка качества

Тот же «Тест модели на валидации» (GUI) или:

```bat
python -m radonc_segmentation.quantization.validate_model --model headandneck/distilled --engine nnunet_native --case validation_DICOM_0001
```

Сравните Dice с OpenVINO-результатом той же модели — расхождение в
пределах ~0.001 ожидаемо (другая реализация свёрток).

## Структура проекта

```
├── run_gui.bat                ← запуск GUI из окружения
├── build.bat                  ← пересборка nnunet_gui.exe (после правок кода)
├── designer.bat               ← Qt Designer на gui\ui
├── nnunet_gui.exe             ← готовый exe (Releases)
├── nnunet_gui_pyside.spec     ← рецепт сборки exe (PyInstaller)
├── settings.example.ini       ← шаблон настроек (копируйте в settings.ini)
├── requirements.txt           ← зависимости пайплайна и GUI
├── requirements-torch-cpu.txt ← опц. PyTorch CPU-сборка (ставит setup_env.py)
├── requirements-quantization.txt ← опц. пакеты импорта/квантизации (onnx, nncf)
├── requirements-native.txt    ← опц. стек нативного движка без torch (CPU-машины)
├── requirements-gpu.txt       ← опц. GPU-инференс: torch CUDA + весь нативный стек
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
│   ├── native\                ← нативный nnUNet-движок (GPU): стейджинг .pth + nnUNetv2Predictor
│   ├── quantization\          ← импорт .pth → IR, тест модели (Dice-гейт), INT8-квантизация, бенчмарк
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
| `[inference]` | `VARIANT` | `legacy` (дефолт) или `aligned` — варианты преобработки (только движок openvino) |
| `[inference]` | `ENGINE` | `openvino` (дефолт) или `nnunet_native` — см. «GPU-инференс» |
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

В exe упакованы PySide6, numpy/scipy, SimpleITK и исходники пакета `radonc_segmentation`. Тяжёлые ML-пакеты (torch, nnunetv2, matplotlib) исключены: вся тяжёлая работа (пайплайн, импорт модели, квантизация) выполняется python окружения `nnunet_env` через незаметный subprocess — поэтому в exe доступны и вкладка «Работа», и весь жизненный цикл модели на вкладке «Квантизация» (импорт .pth → IR, тест с гейтом, INT8). Недоступна только вкладка «Постпроцессинг» (matplotlib) — для неё используйте `run_gui.bat`. Пакеты квантизации доустанавливаются кнопкой «Установить пакеты квантизации» на вкладке «Квантизация» (терминал не нужен).

Правки раскладок `gui\ui\*.ui` подхватываются перезапуском GUI без пересборки (`.ui` читаются при старте, не компилируются).

## Тесты

```bat
python -m pytest tests                      :: GUI собирается, вкладки работают
python -m radonc_segmentation.env_check     :: зависимости против requirements.txt
python -m radonc_segmentation.check_devices :: устройства OpenVINO
```

## Лицензия

[MIT](LICENSE). Веса моделей и данные пациентов не публикуются (медицинская приватность и лицензии исходного датасета).
