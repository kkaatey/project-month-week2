# Поиск товаров

Финальный проект проектного месяца.

В блокноте выполнены подготовка каталога, лексический поиск TF IDF, семантический поиск MiniLM, два эксперимента по улучшению, финальная оценка и логирование в MLflow.

Эксперименты сравниваются на фиксированном golden-set и полном каталоге товаров. Проверены короткий шаблон текста карточки и гибридный поиск RRF. Лучший NDCG@10 показал TF-IDF по названию и категории — 0.1006. На финальном test он получил NDCG@10 0.0808 и Recall@10 0.0928.

## Запуск

1. Поместить четыре файла Parquet в папку `data`.
2. Установить зависимости из `requirements.txt`.
3. Создать `.env` с `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` и `MLFLOW_TRACKING_PASSWORD`.
4. Запустить `semantic_search_project.ipynb` командой Restart and Run All.

## Streamlit

Приложение принимает поисковый запрос и выводит топ товаров. Для каждого товара показаны название, категория, бренд, описание и оценка схожести.

Используется лучший пайплайн по итогам валидации — TF-IDF по названию и категории на полном каталоге. Индекс кешируется Streamlit.

Запуск одной командой:

```bash
streamlit run app.py
```

Данные должны лежать в папке `data`. Для приложения нужны:

- `wb_products_raw_sample.parquet`
- `queries_synthetic_train.parquet`
- `queries_synthetic_test.parquet`

Для проверки MLflow выполнить:

```bash
python check_mlflow.py project-month-week3-kkaatey --env-file .env
```

## Результаты

- `golden_set_week3.csv` — зафиксированная выборка для сравнения.
- `week3_metrics_summary.csv` — сводные метрики всех конфигураций.
- `week3_golden_detailed.csv` — метрики по запросам.
- `week3_final_test_metrics.csv` — финальная оценка выбранной конфигурации.
- `week3_error_analysis.csv` — запросы с наиболее низким NDCG@10.
- `week3_metrics_comparison.png` — сравнение NDCG@10.

Файлы данных и рассчитанные эмбеддинги не добавляются в Git.
