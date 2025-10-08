## ViLawAI Chatbot Deployment

### 1) Copy the Chroma DB

Place the existing `law1_chroma_db` directory inside this `deploy/` folder:

```
deploy/
  law1_chroma_db/
    chroma.sqlite3
    ...
```

Alternatively, set an absolute path via environment variable `CHROMA_DB_DIR`.

### 2) Configure API Key

Create a `.env` file next to `app.py`:

```
GOOGLE_API_KEY=your_api_key_here
# Optional: override DB path
# CHROMA_DB_DIR=C:\\absolute\\path\\to\\law1_chroma_db
```

### 3) Install dependencies

```
pip install -r requirements.txt
```

### 4) Run the API

```
uvicorn deploy.app:app --host 0.0.0.0 --port 8000
```

### 5) Example request

```
POST http://localhost:8000/chat
{
  "question": "Thủ tục đăng ký kết hôn?",
  "session_id": "user-1"
}
```
