# MovieLens 25M recommendation system and local agents

Учебный проект для второго блока: современная рекомендательная модель Two-Tower и единая мультиагентная система на LangGraph. Запрос проходит планирование, поиск, выбор фильмов и независимую проверку. Модели запускаются локально через llama.cpp; внешний платный LLM API не требуется.

[Отчёт по второму блоку (PDF)](report/MovieLens25M_Block2_Report.pdf) · [Проверки GitHub Actions](https://github.com/Novice2311/movielens-25m-agent-recsys/actions/runs/42916062978)

## Что реализовано

- Полный официальный MovieLens 25M, проверка MD5, схемы, ссылочной целостности и дубликатов.
- Глобальное разбиение по времени: train до 2018-01-01 UTC, validation 2018, test 2019.
- Пять методов на одном каталоге: Popularity, Recent90d, ContentGenres, UserKNN40, TwoTower.
- Две независимые нелинейные башни PyTorch с ID и жанровыми признаками, BPR loss, негативы без уже оценённых фильмов.
- Ранжирование полного каталога; Recall@10, NDCG@10, bootstrap CI, coverage, p50 и p95. Test используется после выбора эпохи по validation.
- Три LLM: Qwen3 0.6B, Qwen3.5 0.8B и Gemma3 270M. Gemma проверяет результат другого семейства.
- Локальная Jev-подобная модель ограниченных решений: символьные признаки, обучаемый softmax, temperature scaling, confidence и abstention. Это собственный узкий классификатор, а не модель Jev от TypeSafe.
- Типизированные tools, проверка существования ID и жанров, один повтор выбора и детерминированный fallback.
- Память запроса в StateGraph, эпизоды в SQLite на 30 дней и явные предпочтения на 90 дней с версиями и удалением.
- Собственные `prompts/`, `skills/` и `agent/SOUL.md`, загружаемые в системный контекст.
- Docker Compose, непривилегированные процессы, read-only rootfs, ограничения памяти и CPU, внутренняя сеть. На Windows контейнеры работают внутри WSL2 VM.
- Prometheus, Grafana, OpenTelemetry, Tempo, Alertmanager и локальный webhook; JSONL-логи и траектории.

## Воспроизведение

Нужны Python 3.12 и Docker Compose. Память и процессорный бюджет зависят от профиля загрузки; запуск использует CPU и три отдельных resident model server.

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv/Scripts/Activate.ps1
pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m scripts.download_dataset
python -m recsys.train --zip data/ml-25m.zip --epochs 3 --eval-users 1000
python -m recsys.evaluate_cold --zip data/ml-25m.zip
python -m scripts.train_gate
python -m scripts.download_models
python -m pytest -q
docker compose up --build -d
python -m scripts.evaluate_llms
python -m scripts.evaluate_agents
python -m scripts.check_isolation
python -m scripts.benchmark_api
python -m scripts.verify_deployment
```

После новых evals пересоберите образ приложения, чтобы вложенные JSON-результаты обновились. Артефакты обучения и веса остаются локально в `artifacts/` и `models/` и монтируются только для чтения. После первого запуска сеть сервисов закрыта для внешнего интернета. Данные и веса скачиваются отдельным подготовительным шагом.

`verify_deployment` временно останавливает только контейнер `llm-gateway` этого проекта, проверяет fallback, алерт и трассу, затем запускает его снова. На время проверки не отправляйте параллельные пользовательские запросы.

- Приложение: <http://localhost:18080>
- OpenAPI: <http://localhost:18080/docs>
- Grafana: <http://localhost:18300/d/movie-agent>
- Prometheus: <http://localhost:18090>
- Alertmanager: <http://localhost:19093>

Все порты привязаны к loopback через `ingress`, который пересылает TCP только в фиксированные сервисы. App и LLM находятся только во внутренней сети. Для публичного многопользовательского размещения потребуются аутентификация, авторизация памяти и API, TLS, ограничение частоты запросов и отдельная проверка изоляции. Нынешний запуск предназначен для локальной демонстрации.

## Проверяемые результаты

На 1 000 warm-start пользователях тестового периода:

| Метод | Recall@10 | NDCG@10 | p95 поиска внутри процесса |
| --- | ---: | ---: | ---: |
| Popularity | 0.0354 | 0.0595 | 1.1 мс |
| Recent90d | 0.0478 | 0.0817 | 1.5 мс |
| ContentGenres | 0.0020 | 0.0029 | 1.5 мс |
| UserKNN40 | 0.0648 | 0.1014 | 165.0 мс |
| TwoTower | 0.0530 | 0.0862 | 1.6 мс |

UserKNN40 дал лучший NDCG@10. TwoTower быстрее и превосходит общую популярность; парный 95% интервал разницы NDCG составляет [0.0180, 0.0356]. Для разницы с Recent90d интервал включает ноль. Слабый профиль оценён отдельно в `cold_eval.json`.

Три LLM дали корректный JSON на 8/8 жанровых запросах каждая. Точность жанра: Qwen3 0.6B 6/8, Qwen3.5 0.8B 8/8, Gemma3 270M 8/8. Агентный flow выполнил 12/12 проверяемых сценариев, p95 7.82 с. Отдельный сбой gateway вызвал безопасный fallback, алерт и трассу в Tempo. Прошло 17 тестов. Прямой retrieval API на localhost показал p95 81.2 мс на 100 запросах с восемью параллельными клиентами.

`results/recsys_metrics.json` — обучающие эпохи и offline-метрики; `split.json` — размеры разбиений; `llm_eval.json` — одинаковые задания для трёх LLM; `agent_eval.json` — сквозные сценарии; `decision_eval.json` — метрики локального дискриминатора. JSON сохраняются самим кодом эксперимента.

Warm-start метрики рассчитаны для пользователей с достаточной историей. Профили с менее чем пятью положительными обучающими оценками проверены отдельно; фильмы, которых не было в обучении, отсутствуют в каталоге эксперимента. MovieLens не содержит показов, кликов, длительности просмотра или выручки, поэтому CTR, удержание и прирост бизнеса в этой работе не измерялись.

## Архитектура и исходники

`docs/architecture/` содержит семь представлений C4 и sequence diagram; `recsys/train.py` — обучение и единый протокол оценки; `agent/graph.py` — связанный граф; `llm_proxy.py` — gateway с фиксированным списком моделей; `infra/` — наблюдаемость. Gateway использует Ollama-совместимый формат запросов, но выполняет inference в llama.cpp.

## Источники и условия использования

MovieLens: [GroupLens dataset](https://grouplens.org/datasets/movielens/25m/) и [официальный README](https://files.grouplens.org/datasets/movielens/ml-25m-README.html). Использован полный архив с MD5 `6b51fb2759a8657d3bfcbfc42b592ada`. Исходный датасет не публикуется в репозитории. Условия GroupLens требуют отдельного разрешения для распространения данных и коммерческого использования.

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. DOI: [10.1145/2827872](https://doi.org/10.1145/2827872).

Архитектурные источники: [two-tower retrieval](https://docs.cloud.google.com/architecture/implement-two-tower-retrieval-large-scale-candidate-generation), [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server), [C4](https://c4model.com/diagrams), [TypeSafe Jev](https://docs.typesafe.ai/introduction).

LLM-веса загружаются по отдельным лицензиям их владельцев; это не меняет лицензионные условия моделей. Названия и metadata распределения: [Qwen3](https://ollama.com/library/qwen3), [Qwen3.5](https://ollama.com/library/qwen3.5), [Gemma3](https://ollama.com/library/gemma3).
