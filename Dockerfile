# Single-stage runtime image for the job-hunt knowledge bank chatbot.
#
# Design decisions:
# - No build-time ingest and no build secrets: ingest runs at container start
#   only when the vector store is missing (first boot), so the image never
#   contains the API key and Azure's managed build (az containerapp up
#   --source) works without passing secrets.
# - GOOGLE_API_KEY is expected from the container environment at runtime
#   (set via ACA env vars / secrets), matching how the app already loads it
#   with python-dotenv.
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Ingest only when chroma_db/ is missing (first boot / fresh volume); reuse
# the existing vector store on subsequent restarts to avoid re-embedding.
CMD ["sh", "-c", "[ -d chroma_db ] || python ingest.py; exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]
