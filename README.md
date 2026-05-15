# 🚀 AI-Powered Enterprise ATS & Resume Parser

A high-fidelity, scalable resume extraction and semantic discovery engine. This system transforms unstructured PDF/DOCX/Image resumes into structured JSON data and enables intelligent talent discovery using a hybrid Semantic-Boolean search pipeline.

---

## 🌟 Key Features

- **Multi-Format Extraction**: Intelligent text extraction from PDF, DOCX, and Scanned Images (OCR).
- **Dual-Layer AI Parsing**: 
  - **Local Layer**: Pattern-based entity recognition using `spaCy`.
  - **AI Layer**: High-precision refinement using `Gemini 3 Flash`.
- **Hybrid Discovery Engine**:
  - **Semantic Search**: Vector embeddings (`Sentence-Transformers`) for matching Job Descriptions.
  - **Boolean Filter**: Strict keyword-based filtering with dictionary-augmented normalization.
- **Enterprise Scale**: Distributed background processing using **Celery** & **Redis** to handle high-volume ingestions.
- **Skill Normalization**: Automatic mapping of 500+ skill aliases (e.g., "JS", "ES6" -> "JavaScript") using a SQL-backed master dictionary.

---

## 🏗️ Architecture

```mermaid
graph TD
    User((User/Recruiter)) -->|Upload Resume| API[FastAPI]
    API -->|Queue Task| Redis[(Redis Broker)]
    Redis -->|Process| Worker[Celery Worker]
    Worker -->|OCR/Extract| Extractor[Resume Extractor]
    Worker -->|NLP/Gemini| Parser[AI Parser]
    Parser -->|Embeddings| ST[Sentence Transformers]
    Parser -->|Save| DB[(Supabase/PostgreSQL)]
    
    User -->|Search Query| API
    API -->|Vector Search| DB
    API -->|Keyword Filter| Normalizer[Skill Normalizer]
    API -->|Results| User
```

---

## 📁 Project Structure

```text
ATS/
├── app/                  # Core Application Logic
│   ├── database.py       # Supabase/SQLAlchemy models & DB operations
│   ├── embeddings.py     # Sentence-Transformer service
│   ├── extractor.py      # PDF/DOCX/OCR extraction logic
│   ├── normalizer.py     # Skill normalization & dictionary scan
│   ├── parser.py         # spaCy pipeline & Gemini refinement
│   ├── tasks.py          # Celery background task definitions
│   └── celery_app.py     # Celery configuration
├── scripts/              # Maintenance & Automation
│   ├── migrate_db.py     # Database schema & Vector Engine setup
│   ├── seed_skills.py    # Seed 200+ skill aliases into DB
│   └── start_services.py # One-click service manager (API, Worker, Redis)
├── main.py               # API Entry point & Search Dashboard UI
├── .env                  # Configuration & API Keys
└── README.md             # This guide
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- Redis (installed and running)
- PostgreSQL (Supabase recommended with `pgvector` enabled)

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
# Database URL (PostgreSQL)
DATABASE_URL=postgresql://user:pass@host:5432/postgres
# Celery/Redis
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

### 3. Install Dependencies
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

---

## 🚀 Running the System

### 1. Initialize the Database (First time only)
This sets up the `pgvector` engine and seeds the skill dictionary.
```powershell
python scripts/migrate_db.py
python scripts/seed_skills.py
```

### 2. Start All Services
Use the automated manager to start the API and Worker in separate windows:
```powershell
python scripts/start_services.py
```

---

## 🔍 Usage

### Visual Dashboard
Open your browser and navigate to:
- **Search UI**: `http://localhost:8000/search`
- **Upload UI**: `http://localhost:8000/`

### API Documentation
Interactive Swagger docs are available at `http://localhost:8000/docs`.

#### Example Search Request:
```bash
GET /api/v1/search?query=Fullstack+React+Developer&keywords=Python,AWS
```

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.
