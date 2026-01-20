# AlloyGraph Docker Setup

## Overview

AlloyGraph is an AI-powered platform for nickel superalloy research and design. It uses a multi-container Docker architecture with six services:

- **Frontend** (Vue.js + Nginx): Web interface with Research Chat, Evaluate, and Design modes
- **Backend** (Flask + Gunicorn): AI-powered alloy API with CrewAI agents
- **Pipeline** (Python): One-time data initialization service
- **Weaviate**: Vector database for semantic search
- **GraphDB**: RDF triplestore for the alloy knowledge graph
- **Transformers**: Embedding service for Weaviate

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Nginx)                         │
│                          Port 3000                               │
│         ┌──────────────┬──────────────┬──────────────┐          │
│         │ Research Chat│   Evaluate   │    Design    │          │
│         └──────────────┴──────────────┴──────────────┘          │
└─────────────────────────────┬───────────────────────────────────┘
                              │ /api proxy
┌─────────────────────────────▼───────────────────────────────────┐
│                      Backend (Flask/Gunicorn)                    │
│                         Internal Port 5001                       │
│              ┌─────────────────┬─────────────────┐              │
│              │   Chat Service  │  CrewAI Agents  │              │
│              └─────────────────┴─────────────────┘              │
└──────────────────┬─────────────────────────┬────────────────────┘
                   │                         │
    ┌──────────────▼──────────┐   ┌─────────▼─────────┐
    │      Weaviate           │   │     GraphDB       │
    │   (Vector Search)       │   │  (Knowledge Graph)│
    │     Port 8081           │   │    Port 7200      │
    └──────────────┬──────────┘   └───────────────────┘
                   │
    ┌──────────────▼──────────┐
    │    T2V Transformers     │
    │   (Embedding Service)   │
    └─────────────────────────┘
```

## Quick Start

### Prerequisites
- Docker & Docker Compose v2
- At least 8GB RAM available for Docker
- API key for Groq (required) - [Get one here](https://console.groq.com/keys)

### 1. Configure Environment

Create a `.env` file in the project root with your API keys:

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (for additional LLM providers)
OPENAI_API_KEY=your_openai_api_key_here
TOGETHER_API_KEY=your_together_api_key_here
```

### 2. Start All Services

```bash
docker compose up -d
```

This will:
1. Start GraphDB and Weaviate databases
2. Run the pipeline to load data (first time only)
3. Start the backend API
4. Start the frontend web server

### 3. Access the Application

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:3000 | Main web interface |
| **GraphDB Workbench** | http://localhost:7200 | Browse knowledge graph |
| **Weaviate** | http://localhost:8081 | Vector database API |

## Services

| Service | Image | Purpose |
|---------|-------|---------|
| `frontend` | alloygraph-frontend | Vue.js app served by Nginx |
| `backend` | alloygraph-backend | Flask API with CrewAI agents |
| `pipeline` | alloygraph-backend | Data initialization (runs once) |
| `weaviate` | weaviate:1.33.2 | Vector database |
| `graphdb` | graphdb:10.8.0 | RDF triplestore |
| `t2v-transformers` | transformers-inference | Text embeddings |

## Common Commands

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f backend
docker compose logs -f frontend

# Rebuild after code changes
docker compose build backend frontend
docker compose up -d

# Stop all services
docker compose down

# Full reset (removes data volumes)
docker compose down -v
```

## Troubleshooting

### Backend won't start
Check if the pipeline completed successfully:
```bash
docker compose logs pipeline
```


### Data not loading
The pipeline skips if data already exists. To force reload:
```bash
docker compose down -v  # Removes volumes
docker compose up -d
```

### Check service health
```bash
docker compose ps
curl http://localhost:3000/health
curl http://localhost:7200/rest/repositories
```

## Key Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Service orchestration |
| `.env` | API keys |
| `backend/Dockerfile` | Backend container |
| `frontend/Dockerfile` | Frontend multi-stage build |
| `frontend/nginx.conf` | Nginx proxy configuration |
