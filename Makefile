.PHONY: chat dashboard demo ingest

chat:
	unset VIRTUAL_ENV; uv run streamlit run app/app.py

dashboard:
	unset VIRTUAL_ENV; uv run streamlit run monitoring/dashboard.py

demo:
	unset VIRTUAL_ENV; uv run streamlit run app/app_dashboard.py

ingest:
	uv run python -m ingest.ingest

evaluate-retrieval:
	uv run python -m evaluation.evaluate_retrieval

evaluate-answers:
	uv run python -m evaluation.evaluate_answers
