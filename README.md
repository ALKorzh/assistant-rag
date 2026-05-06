# Agentic RAG Assistant

Личный ИИ-ассистент с агентным маршрутизатором, RAG-поиском по локальным
документам и набором внешних инструментов (погода, Wikipedia, YouTube,
веб-поиск). Все языковые вызовы идут через Gemini 2.5 Flash.

## Архитектура

```
┌────────────┐     HTTP /api/*     ┌──────────────────┐
│  Frontend  │  ─────────────────▶ │  FastAPI backend │
│  React/TS  │  ◀───────────────── │   LangGraph      │
└────────────┘                     └──────┬───────────┘
                                          │
                                ┌─────────┴─────────┐
                                ▼                   ▼
                        ┌──────────────┐    ┌─────────────┐
                        │   Qdrant     │    │   Ollama    │
                        │ (vector DB)  │    │ (embeddings)│
                        └──────────────┘    └─────────────┘
```

- **Frontend** — Vite + React + TypeScript + Tailwind, раздаётся через nginx,
  который проксирует `/api/*` на бэкенд.
- **Backend** — FastAPI с пакетами `app/api`, `app/agent`, `app/services`,
  `app/tools`, `app/core/config.py` и `app/schemas`.
- **Qdrant** — векторное хранилище для RAG.
- **Ollama** — локальные эмбеддинги (`nomic-embed-text`).
- **Gemini 2.5 Flash** — генерация и маршрутизация (через `langchain_google_genai`).

## Структура репозитория

```
.
├── app/                     # FastAPI backend
│   ├── agent/               # LangGraph: state, prompts, llm, grader, nodes, graph
│   ├── api/                 # routers + dependencies
│   ├── core/                # Settings (config.py)
│   ├── schemas/             # Pydantic-модели запросов/ответов
│   ├── services/rag_service.py
│   └── tools/               # weather, wiki, youtube
├── frontend/                # React UI (Vite + Tailwind)
├── docker-compose.yml       # qdrant + ollama + backend + frontend
├── Dockerfile               # backend image
├── requirements.txt         # python deps
└── .env.example             # шаблон переменных окружения
```

## Быстрый запуск через Docker

1. Убедитесь, что установлены **Docker** и **Docker Compose v2**.
2. Создайте файл `.env` на основе шаблона и пропишите ключи:

   ```bash
   cp .env.example .env
   # GOOGLE_API_KEY=...
   # OPENWEATHER_API_KEY=...
   ```

3. Поднимите весь стек одной командой:

   ```bash
   docker compose up --build
   ```

4. После сборки откройте интерфейс: <http://localhost:5173>.
   API доступен на <http://localhost:8000>, документация Swagger — <http://localhost:8000/docs>.

При первом запуске сервис `ollama_init` автоматически загружает модель эмбеддингов
`nomic-embed-text` в контейнер Ollama.

## Локальная разработка

### Backend

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Перед запуском поднимите Qdrant и Ollama (`docker compose up qdrant ollama ollama_init`)
и выставьте переменные окружения (см. `.env.example`).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite поднимается на <http://localhost:5173> и проксирует `/api/*` на
`http://localhost:8000`. Цель прокси можно переопределить через
`VITE_API_PROXY_TARGET`.

## Переменные окружения

| Переменная | Назначение | По умолчанию |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Ключ Google AI Studio для Gemini | — |
| `OPENWEATHER_API_KEY` | Ключ OpenWeather (инструмент погоды) | — |
| `GEMINI_MODEL` | Имя модели Gemini | `gemini-2.5-flash` |
| `QDRANT_URL` | URL Qdrant | `http://qdrant:6333` |
| `OLLAMA_BASE_URL` | URL Ollama | `http://ollama:11434` |
| `OLLAMA_EMBEDDING_MODEL` | Модель эмбеддингов | `nomic-embed-text` |
| `CORS_ALLOW_ORIGINS` | Origins через запятую | `*` |
| `UPLOAD_DIR` | Папка для загруженных документов | `data/raw` |

Полный список — в `app/core/config.py`.

## API

| Метод | Путь | Описание |
| --- | --- | --- |
| `POST` | `/chat` | Отправить сообщение агенту, получить ответ |
| `POST` | `/upload` | Загрузить PDF/TXT и проиндексировать в Qdrant |
| `GET`  | `/health` | Проверка работоспособности |

Пример:

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"text":"Привет!"}'
```
