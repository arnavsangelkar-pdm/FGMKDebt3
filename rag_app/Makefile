setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

serve:
	uvicorn rag_app.app:app --host 0.0.0.0 --port 8000 --reload
