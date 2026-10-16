# Optional container deployment

The included Dockerfile packages Streamlit and Tesseract for a cloud API backend. It excludes private `.env`, local data, and model downloads. No deployment has been performed.

Local container build/run (requires Docker):

```bash
docker build -t documind .
docker run --rm -p 8501:8501 --env-file .env -e AI_PROVIDER=groq documind
```

For Cloud Run, configure a project, billing, Artifact Registry, and a Secret Manager secret containing the Groq key. Keep authentication enabled. An example after replacing placeholders:

```bash
gcloud run deploy documind --source . --region REGION --no-allow-unauthenticated --memory 2Gi --cpu 1 --timeout 600 --set-env-vars AI_PROVIDER=groq --set-secrets GROQ_API_KEY=GROQ_SECRET_NAME:latest
```

The runtime service account needs access to that secret. Grant invoker access to intended users. Do not place API keys in images, source, URLs, or command-line literals.

Cloud Run filesystems are ephemeral. Custom categories/templates in SQLite will not reliably persist across instances or deployments; use a persistent database before relying on that functionality in a multi-instance service. Uploaded documents and images are session memory, not permanent storage. No Google Cloud Storage integration is configured. The downloaded DocVQA corpus is intentionally excluded from the image.

The cloud container defaults to Groq. To use Ollama, provide a reachable Ollama server and set AI_PROVIDER=ollama and OLLAMA_BASE_URL appropriately; localhost inside Cloud Run does not refer to your laptop. The supplied container does not bundle or run a local Gemma model. Selecting Ollama in that cloud UI will fail unless such a server is configured.
