# Enterprise Resume Parser (ATS)

A high-performance, AI-powered resume extraction and semantic search engine designed for enterprise scale.

## 🚀 Features
- **Hybrid Extraction**: Supports PDF, DOCX, and Scanned Images (OCR).
- **AI-Powered Parsing**: Uses spaCy for entity recognition and Gemini 3 Flash for high-fidelity data refinement.
- **Semantic Search**: Vector-based search to find candidates based on job descriptions, not just keywords.
- **Distributed Processing**: Scalable background processing using Celery and Redis.
- **Skill Normalization**: Dictionary-augmented normalization to ensure data consistency across millions of records.

## 📁 Project Structure
- `app/`: Core logic including database, embeddings, and parsing services.
- `scripts/`: Maintenance and service management scripts.
- `main.py`: FastAPI entry point and Search Dashboard.
- `requirements.txt`: Project dependencies.

## 🛠️ Getting Started

### 1. Setup Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure .env
Create a `.env` file in the root with:
```env
GOOGLE_API_KEY=your_gemini_key
CELERY_BROKER_URL=redis://localhost:6379/0
```

### 3. Initialize Database
```powershell
python scripts/migrate_db.py
python scripts/seed_skills.py
```

### 4. Start Services
```powershell
python scripts/start_services.py
```

## 📊 API Endpoints
- `POST /api/v1/extract`: Upload and process a resume.
- `GET /api/v1/status/{task_id}`: Check processing status.
- `GET /api/v1/search`: Hybrid semantic/boolean search.
- `GET /search`: Access the visual search dashboard.
