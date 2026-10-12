### Запуск проекта

```bash
# синхронизация проекта и сопоставление зависимостей
uv sync

# выгрузка документов из скаченных zip архивов
uv run python extract_upd.py

# получение статистики и подготовка датасета для выполения задания 2 Лабораторной работы 1
uv run python prepare_dataset.py --input upd --out out

# создаем синтетическую вариацию названий колонок дял проверки больших языковых моделей
uv run make_hard_set.py

# запуск zero-shot прогона
uv run python run_llm_benchmark.py --stage zero

# запуск few-shot прогона с 3 примерами
uv run python run_llm_benchmark.py --stage few --shots 3

# запуск для создания сводной статистики после 2 прогонов и определения лучшей модели
uv run python analyze_results.py --dir llm_out_hard
```

### Ссылка на гугл диск с документами и результатами Лабораторной работы 1

https://drive.google.com/drive/folders/1I47B5rXethOOZST8-Pcs0Rs4_sEg01N3?usp=drive_link
