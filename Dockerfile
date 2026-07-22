FROM python:3.13-slim

RUN pip install uv --quiet

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8501

ENV TOKENIZERS_PARALLELISM=false
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["uv", "run", "streamlit", "run", "app/app.py"]
