## ViLawAI Chatbot Deployment

### Setup

1. Copy `law1_chroma_db` folder into `deploy/` directory
2. Set `GOOGLE_API_KEY` in environment variables
3. Run: `python deploy/app.py`

### Deploy to Render

1. Push to GitHub
2. Connect repo to Render
3. Set `GOOGLE_API_KEY` in Render environment variables
4. Deploy

### API Usage

POST `/chat` with:

```json
{
  "question": "Thủ tục đăng ký kết hôn?",
  "session_id": "user-1"
}
```
