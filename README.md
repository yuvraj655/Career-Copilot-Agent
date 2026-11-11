# Career Copilot Agent

An AI agent for engineering students (placements, interviews, resumes, study plans),
built with Python, LangChain, Gemini and LangServe. It has three tools: a safe
calculator, the current date/time, and Wikipedia lookups.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then put your GOOGLE_API_KEY in .env
uvicorn server:app --reload
```
Open http://localhost:8000 (it redirects to the playground at `/agent/playground/`).

## API
- `POST /agent/invoke` with body `{"input": {"input": "your question"}}`
- `GET /agent/playground/` interactive test page
- `GET /docs` API documentation

## Deploy on Render
1. Push this folder to a GitHub repo.
2. On Render: New > Web Service > connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
5. Add environment variables `GOOGLE_API_KEY` (your key) and `PYTHON_VERSION` = `3.11.9`.
6. Deploy, then open `https://<your-service>.onrender.com/agent/playground/`.
