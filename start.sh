#!/usr/bin/env bash
set -e

export STREAMLIT_HOST="${STREAMLIT_HOST:-127.0.0.1}"
export STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

# Start Streamlit in the background on an internal port
streamlit run app.py --server.address "${STREAMLIT_HOST}" --server.port "${STREAMLIT_PORT}" &

# Start FastAPI (proxy) in the foreground on Render's $PORT
exec gunicorn -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:"${PORT}" --timeout 300
