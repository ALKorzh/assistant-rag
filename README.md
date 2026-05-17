# Agentic RAG Assistant

Личный ИИ-ассистент на базе **FastAPI**, **LangGraph** и **Gemini**: маршрутизация запроса к нужному инструменту, **RAG** по локальным документам (Qdrant + Ollama-эмбеддинги) и вызовы внешних сервисов (погода, Wikipedia, YouTube, веб-поиск, калькулятор).

---

## Содержание

1. [Возможности](#возможности)
2. [Стек технологий](#стек-технологий)
3. [Архитектура](#архитектура)
4. [Агент LangGraph](#агент-langgraph)
5. [RAG и индексация](#rag-и-индексация)
6. [Инструменты](#инструменты)
7. [HTTP API и безопасность](#http-api-и-безопасность)
8. [Наблюдаемость и готовность](#наблюдаемость-и-готовность)
9. [Переменные окружения](#переменные-окружения)
10. [Docker и порты](#docker-и-порты)
11. [Локальная разработка](#локальная-разработка)
12. [Тесты и CI](#тесты-и-ci)
13. [Структура репозитория](#структура-репозитория)
14. [Дополнительные скрипты](#дополнительные-скрипты)

---

## Возможности

- **Умная маршрутизация**: один LLM-роутер (Gemini, низкая температура) выбирает сценарий — RAG, погода, Wikipedia, YouTube, калькулятор, веб-поиск или прямой ответ без инструментов.
- **RAG с фильтром релевантности**: после извлечения чанков из Qdrant выполняется **lexical grader** (лемматизация `pymorphy3`, числа, учебная лексика); при «нет» контекст переписывается в запрос к DuckDuckGo.
- **Мультимодальные ответы Gemini**: извлечение текста из ответа модели приведено к безопасной строке (`_stringify_model_content`), чтобы запросы в YouTube/DDG не превращались в `repr` списков блоков.
- **YouTube**: приоритет **YouTube Data API v3** (если задан ключ), с валидацией URL и деградацией на `youtube-search-python` / DuckDuckGo; эвристики для «развлекательных» vs технических запросов.
- **Загрузка документов**: PDF, текст, Markdown, изображения с **OCR** (Tesseract `rus+eng` в образе бэкенда).
- **Опциональная защита API**: заголовок `X-API-Key`, если в окружении задан `API_KEY`.
- **Идемпотентность трассировки**: middleware назначает и возвращает `X-Request-ID`.
- **Проверка готовности**: `GET /ready` проверяет Qdrant и наличие модели эмбеддингов в Ollama.

---

## Стек технологий

| Слой | Технологии |
| --- | --- |
| Backend | Python **3.12**, **FastAPI**, **Uvicorn**, **Pydantic** |
| Оркестрация агента | **LangGraph**, **LangChain** (community, qdrant, ollama, google-genai) |
| LLM | **Google Gemini** через `langchain_google_genai` (`ChatGoogleGenerativeAI`) |
| Векторное хранилище | **Qdrant** |
| Эмбеддинги | **Ollama** + модель по умолчанию `nomic-embed-text` (768-мерные векторы) |
| Поиск в сети | **DuckDuckGo** (`DuckDuckGoSearchRun`, `duckduckgo_search`) |
| OCR | **Tesseract** + **Pillow** + **pytesseract** |
| Морфология (RAG grader) | **pymorphy3** |
| Frontend | **React 18**, **TypeScript**, **Vite 5**, **Tailwind CSS**, **react-markdown**, **lucide-react** |
| Контейнеризация | **Docker**; фронт в **nginx** (прокси `/api/*` → бэкенд) |

---

## Архитектура

```
┌─────────────────┐     POST /api/chat, /api/upload      ┌────────────────────────────┐
│  Браузер / curl │ ─────────────────────────────────▶ │  FastAPI (app.main)         │
│  React UI        │ ◀───────────────────────────────── │  CORS + RequestIDMiddleware │
└─────────────────┘                                      │  Роуты: /chat, /upload       │
        │                                                └──────────────┬─────────────┘
        │ docker:5173 (nginx)                                         │
        │   /api/* → backend:8000                                     ▼
        │                                        ┌────────────────────────────────┐
        └──────────────────────────────────────│  LangGraph: router → … → generator│
                                                 └──────────────┬─────────────────┘
                                                                │
                    ┌───────────────────────────────────────────┼───────────────────────────┐
                    ▼                                           ▼                           ▼
            ┌───────────────┐                           ┌───────────────┐           ┌──────────────┐
            │   Qdrant      │                           │   Ollama      │           │  Внешние API  │
            │  (векторы)    │                           │  эмбеддинги   │           │ Weather, Wiki, │
            └───────────────┘                           └───────────────┘           │ YouTube, DDG   │
                                                                                 └──────────────┘
```

- В **docker-compose** фронтенд слушает **5173** (хост) → контейнер **nginx:80**; путь `/api/` проксируется на сервис `backend:8000` с корневым маппингом (`/api/chat` → бэкенд `/chat`).
- Бэкенд монтирует `./data` в `/app/data`: сырые загрузки и общее хранилище данных на хосте.

---

## Агент LangGraph

Граф собирается в `app/agent/graph.py`, тип состояния — `AgentState` (`app/agent/state.py`).

### Узлы графа

| Узел | Назначение |
| --- | --- |
| `router` | Вызов Gemini с системным промптом маршрутизации; поле `next_step`: `rag`, `weather`, `wikipedia`, `youtube`, `calculator`, `web_search`, `direct`. |
| `rag` | Семантический поиск по Qdrant, формирование контекста для ответа. |
| `grader` | Быстрая проверка релевантности контекста запросу (`app/agent/grader.py`). |
| `rewrite` | Переформулирование запроса под веб-поиск (при ветке `web_search` после нерелевантного RAG). |
| `search` | DuckDuckGo, результат подмешивается в генератор. |
| `weather` / `wikipedia` / `youtube` / `calculator` | Специализированные узлы с вызовами инструментов. |
| `generator` | Итоговый ответ Gemini с учётом префиксов контекста (RAG, погода, wiki, YouTube, калькулятор, веб). |
| `reflection` | Пост-обработка/оценка качества; при необходимости одна итерация повторной генерации (`retry` → снова `generator`). |

### Поток (упрощённо)

1. **router** → ветка по `next_step`.
2. Для **RAG**: `rag` → **grader** → при релевантности `generator`, иначе `rewrite` → `search` → `generator`.
3. Для инструментов: прямой переход в `generator`.
4. После ответа: **reflection** → конец или повтор генерации.

Технические детали узлов см. в `app/agent/nodes.py`: например, **детерминированные** ответы для калькулятора и части сценариев YouTube (без «галлюцинаций» ссылок), фильтрация строк с чужими YouTube URL в ответе, склейка пропущенных ссылок из тулза.

Два клиента Gemini (`app/agent/llm.py`): **router** — низкая температура; **answer** — повышенная + смягчённые safety thresholds (`BLOCK_ONLY_HIGH`), чтобы реже получать пустые ответы на энциклопедический контент.

---

## RAG и индексация

- **Чанкинг**: `RecursiveCharacterTextSplitter` (`RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP` в настройках).
- **Коллекция Qdrant**: создаётся при первом обращении, косинусное расстояние, размер вектора из `QDRANT_VECTOR_SIZE` (по умолчанию 768 под `nomic-embed-text`).
- **Подсказки из вопроса** (`hints_from_question` в `rag_service.py`): из текста запроса извлекаются имена файлов (`*.pdf`, `*.txt`, `*.md`) и фрагменты после слов «файл», «документ» и т.д. Чанки, попавшие под hints, поднимаются выше в порядке выдачи; при наличии hints увеличивается глубина выборки (`fetch_k`).
- **Grader**: эвристики по леммам, числам и наборам «учебных» терминов — отсев заведомо неподходящего контекста до дорогой генерации.

Поддерживаемые загрузки (см. `ALLOWED_UPLOAD_EXTENSIONS`): PDF, TXT, MD, PNG, JPG, JPEG, WEBP (и др. по конфигу). Максимальный размер — `UPLOAD_MAX_BYTES` (по умолчанию 15 MiB).

---

## Инструменты

| Инструмент | Файл | Особенности |
| --- | --- | --- |
| Погода | `app/tools/weather.py` | OpenWeather, ключ `OPENWEATHER_API_KEY`, таймауты и URL из конфига. |
| Wikipedia | `app/tools/wiki_tool.py` | Язык и лимиты, корректный `User-Agent` для MediaWiki. |
| YouTube | `app/tools/youtube_tool.py`, `youtube_data_api.py` | Цепочка провайдеров, ретраи HTTP (`app/core/retry_utils.py`), очистка «битых» запросов от multimodal-bлоков LLM. |
| Веб-поиск | `DuckDuckGoSearchRun` в `nodes.py` | После rewrite промпта. |
| Калькулятор | `app/tools/calculator.py` | Безопасное вычисление выражений; результат отдаётся пользователю детерминированно. |

---

## HTTP API и безопасность

| Метод | Путь | Описание |
| --- | --- | --- |
| `POST` | `/chat` | Тело JSON: `{ "text": "..." }`. Ответ: `{ "answer": "..." }`. |
| `POST` | `/upload` | `multipart/form-data`, поле файла; индексация в Qdrant. |
| `GET` | `/health` | `{ "status": "ok" }` — процесс жив. |
| `GET` | `/ready` | JSON с детализацией по Qdrant и Ollama; код **503**, если зависимости не готовы. |

- Если задан **`API_KEY`**, ко всем эндпоинтам роутеров требуется заголовок **`X-API-Key`** (`app/api/security.py`, сравнение через `secrets.compare_digest`). Если ключ не задан — проверка отключена (удобно для локальной разработки).
- **Frontend**: при сборке можно передать `VITE_API_KEY` (и при необходимости `VITE_API_BASE_URL`), см. `frontend/src/lib/api.ts`.

Swagger UI доступен на `/docs` (стандарт FastAPI).

---

## Наблюдаемость и готовность

- **`RequestIDMiddleware`**: для каждого запроса генерируется или пробрасывается `X-Request-ID`, заголовок возвращается в ответе; в логах чата/загрузки фигурирует `request_id`.
- **`GET /ready`**: проверка Qdrant (`get_collections`) и Ollama (`/api/tags` + наличие модели эмбеддингов). Используется оркестраторами и в Kubernetes-подобных сценариях (liveness vs readiness).

---

## Переменные окружения

Ниже — основные переменные (полный парсинг — `app/core/config.py`, шаблон — `.env.example`).

### LLM и режимы генерации

| Переменная | Назначение | По умолчанию |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Ключ Google AI (Gemini) | — |
| `GEMINI_MODEL` | Идентификатор модели | `gemini-3.1-flash-lite` |
| `GEMINI_ROUTER_TEMPERATURE` | Температура роутера | `0.0` |
| `GEMINI_ANSWER_TEMPERATURE` | Температура ответа | `0.7` |

### API сервера

| Переменная | Назначение | По умолчанию |
| --- | --- | --- |
| `API_TITLE` / `API_DESCRIPTION` | Метаданные OpenAPI | см. код |
| `API_HOST` / `API_PORT` | Bind uvicorn | `0.0.0.0` / `8000` |
| `CORS_ALLOW_ORIGINS` | Список origin через запятую | `*` |
| `API_KEY` | При непустом значении включается проверка `X-API-Key` | — |

### Qdrant и Ollama

| Переменная | Назначение | По умолчанию |
| --- | --- | --- |
| `QDRANT_URL` | URL HTTP API Qdrant | `http://localhost:6333` (в Docker: `http://qdrant:6333`) |
| `QDRANT_COLLECTION` | Имя коллекции | `my_documents` |
| `QDRANT_VECTOR_SIZE` | Размер вектора | `768` |
| `OLLAMA_BASE_URL` | Базовый URL Ollama | `http://localhost:11434` |
| `OLLAMA_EMBEDDING_MODEL` | Модель эмбеддингов | `nomic-embed-text` |

### RAG и загрузка

| Переменная | По умолчанию |
| --- | --- |
| `RAG_CHUNK_SIZE` | `600` |
| `RAG_CHUNK_OVERLAP` | `100` |
| `RAG_SEARCH_LIMIT` | `3` |
| `UPLOAD_DIR` | `data/raw` (в Docker: `/app/data/raw`) |
| `UPLOAD_MAX_BYTES` | `15728640` |
| `ALLOWED_UPLOAD_EXTENSIONS` | `.pdf`, `.txt`, `.md`, изображения и др. |

### Внешние инструменты и HTTP

| Переменная | Назначение |
| --- | --- |
| `OPENWEATHER_API_KEY`, `OPENWEATHER_URL`, `OPENWEATHER_FORECAST_URL`, `OPENWEATHER_TIMEOUT` | Погода |
| `WIKIPEDIA_LANGUAGE`, `WIKIPEDIA_*`, `WIKIPEDIA_USER_AGENT` | Wikipedia |
| `YOUTUBE_DATA_API_KEY` или `YOUTUBE_API_KEY`, `YOUTUBE_REGION_CODE`, `YOUTUBE_RELEVANCE_LANGUAGE`, `YOUTUBE_DATA_API_TIMEOUT`, `YOUTUBE_PRIMARY_PROVIDER`, `YOUTUBE_RESULTS_LIMIT` | YouTube |
| `HTTP_RETRY_ATTEMPTS`, `HTTP_RETRY_BASE_DELAY` | Повторы HTTP-вызовов |
| `DUCKDUCKGO_TIMEOUT` | Таймаут DDG |

---

## Docker и порты

Сервисы (`docker-compose.yml`):

| Сервис | Назначение | Порты (хост) |
| --- | --- | --- |
| `qdrant` | Векторная БД | 6333, 6334 |
| `ollama` | Эмбеддинги | 11434 |
| `ollama_init` | Однократный `ollama pull` модели эмбеддингов | — |
| `backend` | FastAPI | 8000 |
| `frontend` | nginx + статика Vite | 5173 → 80 |

Образ бэкенда (`Dockerfile`): Python 3.12-slim, системные пакеты для сборки и **Tesseract** (rus/eng). Данные Qdrant и Ollama — в именованных volumes.

---

## Локальная разработка

### Backend

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Поднимите Qdrant и Ollama (например `docker compose up -d qdrant ollama ollama_init`), задайте `.env` по образцу `.env.example`, затем:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite dev-server проксирует запросы: префикс **`/api`** (см. `vite` конфиг) на бэкенд; цель задаётся **`VITE_API_PROXY_TARGET`** (по умолчанию `http://localhost:8000`). Для прод-сборки в Docker API доступен с браузера как `/api/...` на том же хосте, что и UI.

---

## Тесты и CI

- Запуск: `python -m pytest` (или `pytest -q`).
- **GitHub Actions** (`.github/workflows/ci.yml`): на push/PR в `main`/`master` — Python **3.12**, установка `requirements.txt`, прогон `pytest`.

---

## Структура репозитория

```
.
├── app/
│   ├── main.py                 # фабрика FastAPI, middleware, /health, /ready
│   ├── agent/                  # LangGraph: graph, nodes, state, prompts, llm, grader
│   ├── api/
│   │   ├── routes/             # chat, upload
│   │   ├── deps.py             # внедрение графа и RAGService
│   │   └── security.py         # API key
│   ├── core/                   # config, retry_utils
│   ├── middleware/             # RequestID
│   ├── schemas/                # Pydantic-схемы API
│   ├── services/               # rag_service, readiness
│   └── tools/                  # weather, wiki, youtube, calculator, youtube_data_api
├── frontend/                   # Vite + React + Tailwind
├── data/                       # загрузки и примеры (часть может быть в .gitignore)
├── docker-compose.yml
├── Dockerfile                  # backend
├── requirements.txt
├── .env.example
└── scripts/                    # в т.ч. отчёты API (PowerShell)
```

---

## Дополнительные скрипты

- **`scripts/run_detailed_api_report.ps1`** (Windows): при работающем стеке снимает подробный журнал HTTP; по умолчанию результат в `SUPERVISOR_DETAILED_API_REPORT.md` — путь указан в `.gitignore`, файл не предназначен для коммита.

---

## Быстрый старт (Docker)

1. Установите **Docker** и **Docker Compose v2**.
2. `cp .env.example .env` — задайте минимум **`GOOGLE_API_KEY`**, при необходимости погоду, YouTube Data API и др.
3. `docker compose up --build`
4. UI: **http://localhost:5173**  
   API: **http://localhost:8000**  
   Документация: **http://localhost:8000/docs**

При первом запуске `ollama_init` подтянет модель эмбеддингов (`OLLAMA_EMBEDDING_MODEL`, по умолчанию `nomic-embed-text`).
