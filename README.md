# Прогноз автомобильного трафика

Лабораторная работа №1 по дисциплине «Облачные вычислительные системы».
Сервис FastAPI прогнозирует количество автомобилей за час на участке I-94.
Структура основана на [CloudCS-Lab1](https://github.com/kpdvstu/CloudCS-Lab1).

## Данные и модель

Используется [Metro Interstate Traffic Volume](https://doi.org/10.24432/C5X60B)
(UCI, John Hogue, CC BY 4.0). Из исходного набора выбраны 10 000 уникальных
часовых наблюдений: первые 8000 по времени — обучение, последние 2000 — проверка.
Перед выборкой удалены ошибочные измерения температуры и осадков и повторные
записи одного часа. В подготовленных данных нет пропусков и дубликатов.

Семь признаков: час, день недели, месяц, тип погоды, температура в °C,
осадки за час и облачность. Целевая переменная — `traffic_volume`.

Pipeline объединяет ColumnTransformer, OneHotEncoder для трёх категориальных
признаков и RandomForestRegressor на 200 деревьях (`random_state=42`, `n_jobs=4`).
Числовые признаки не масштабируются. Неизвестные категории обрабатываются
через `handle_unknown="ignore"`. В `models/pipeline.pkl` сохраняется весь Pipeline.

| MAE, автомобилей/ч | RMSE, автомобилей/ч | R² |
|---:|---:|---:|
| 298,07 | 538,95 | 0,9264 |

Метрики получены на более поздних данных с наблюдавшейся погодой.
Проверочная выборка использовалась при выборе числа деревьев; это не независимая
финальная оценка. Модель относится к одному посту учёта.

## Запуск в Windows PowerShell

Из корня проекта, при первой установке (Python 3.10 или 3.12):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Откройте `notebooks/auto-mpg.ipynb` в VS Code, выберите ядро
`.venv\Scripts\python.exe` и выполните все ячейки. Либо запустите обучение:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/auto-mpg.ipynb --ExecutePreprocessor.timeout=600
```

Исходный датасет находится в `data/`, интернет для обучения не требуется.
Файл модели исключён из Git и создаётся при выполнении notebook.

Запуск API в активированном окружении:

```powershell
$env:MODEL_PATH="./models/pipeline.pkl"
uvicorn main:app --app-dir src
```

Swagger: http://127.0.0.1:8000/docs. В Authorize введите `00000`.
GET `/healthcheck` работает без токена. POST `/predictions` принимает:

```json
{
  "hour": 18,
  "day_of_week": "Monday",
  "month": "July",
  "weather_main": "Clouds",
  "temperature_c": 20,
  "rain_1h": 0,
  "clouds_all": 75
}
```

Ответ сохранённой модели: `{"predicted_traffic_volume": 4265.89}`.
Сервис загружает Pipeline при старте и возвращает прогноз в ответ на запрос.
После переобучения сервис нужно перезапустить.

```powershell
curl.exe http://127.0.0.1:8000/healthcheck
curl.exe -X POST http://127.0.0.1:8000/predictions -H "Authorization: Bearer 00000" -H "Content-Type: application/json" --data-binary "@data/example_request.json"
```

Неверный или отсутствующий токен даёт HTTP 401, некорректные данные — HTTP 422.

## Проверки и файлы

```powershell
python -m pytest test -v
```

Все 8 тестов проходят. GitHub Actions выполняет notebook и тесты при push
и pull_request на Python 3.10 и 3.12.

- `data/` — исходные данные, выборка и пример запроса.
- `notebooks/auto-mpg.ipynb` — подготовка данных и обучение; имя сохранено из исходного проекта.
- `src/main.py` — API, валидация и авторизация.
- `src/model_utils.py` — загрузка Pipeline и прогноз.
- `models/` — обученный Pipeline и метрики.
- `test/` — тесты сервиса и инференса.
