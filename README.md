# DocuMind AI — Intelligent Document Question Answering and Form Analyzer

A Streamlit student project combining local PDF parsing / Tesseract OCR with Ollama + Gemma 3 4B multimodal inference (and an optional Groq backend), structured extraction, and deterministic validation.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Local model setup: Ollama + Gemma 3 4B

Ollama is installed on this Mac. Start its server in a separate terminal:

```bash
ollama serve
```

Download the model once (the model includes text and image support):

```bash
ollama pull gemma3:4b
```

Set `.env` beside `app.py`:

```env
AI_PROVIDER=ollama
OLLAMA_MODEL=gemma3:4b
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_NUM_CTX=16384
OLLAMA_TIMEOUT=600
```

Both text-only and vision analysis now use the same local model. No API key is required and there is no automatic cloud fallback. Keep the Ollama server running while using Streamlit. Initial model loading and image inference can be slower than cloud inference. On low-memory machines, try one page at a time and a smaller context setting; long documents may then need shorter inputs.

Official references: https://ollama.com/library/gemma3:4b and https://docs.ollama.com/api/chat . Only the 4B and larger Gemma 3 variants support images; do not substitute a text-only variant.

To select the optional cloud backend instead:

```env
AI_PROVIDER=groq
GROQ_API_KEY=your_private_groq_key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_VISION_MODEL=qwen/qwen3.8-27b
```

Groq credentials already in your `.env` are preserved but are unused when `AI_PROVIDER=ollama`. The local adapter uses Ollama's native `/api/chat`, base64 page images, and JSON output mode. Current Groq vision limits: https://console.groq.com/docs/vision .

Install Tesseract for scanned pages/images. On macOS with Homebrew: `brew install tesseract`. On Windows install Tesseract and optionally set `TESSERACT_CMD` to the executable path in `.env`; otherwise it is discovered on PATH. Vision still works if OCR is unavailable, provided the selected model is running.

## Workflow

1. Choose **Ollama · Gemma 3 4B (local)** or **Groq (cloud)** in the sidebar, then upload a PDF or PNG/JPEG. The choice applies only to your session and does not edit `.env`. Ollama needs its server running; Groq needs `GROQ_API_KEY`. Switching backends clears previous insights and chat while retaining the document.
2. Choose **Vision + OCR** or **Text only**, and select one to three pages. For longer PDFs, enable **Analyze entire PDF in batches** (up to 30 pages). Each model request uses at most three images. Answers are scoped to the selected/retrieved pages.
3. Process the document to render pages, extract local text, and classify it.
If automatic classification returns unknown, choose an existing type or enter a new type. Custom category names are persisted in `data/categories.sqlite3` and included in future classification prompts. This does not train the model. Save custom required fields to create a reusable completeness template for a new category.

4. In **Overview**, extract a summary, fields, confidence estimates, evidence, and page/region references. Download the JSON.
5. In **Document checks**, review expected fields or enter custom required field names, then select **Save required fields for this category** to persist the template. Review arithmetic/date rules or request an AI semantic review; semantic flags require human review.
6. In **Source text**, inspect the actual page images and extracted text.
7. Ask questions in **Ask a question**. Answers request source page numbers and evidence; question history is retained for the current selection.

Local pipeline: upload → selected-page rendering and normalization → PDF text or Tesseract OCR → Ollama/Groq (images + text in vision mode) → Pydantic document validation → missing-field and consistency rules → evidence and natural-language answers.

Page images are normalized to RGB, oriented from EXIF, resized to at most 1800px on the longest side, and JPEG encoded. OCR uses grayscale/autocontrast. No automatic margin crop is applied because it can remove evidence.

## Demonstration

A synthetic invoice and two labeled example questions are included. Recreate them with:

```bash
python -m scripts.create_demo
```

Upload `examples/demo-invoice.pdf`. Ask “What is the total?” (220), “What is the invoice number?” (INV-1045), or “Who is the client?” (Example College). These files are synthetic demonstration data, not DocVQA benchmark results.

## DocVQA subset and evaluation

Download a local 2,000-question subset directly from the mirror:

```bash
python -m scripts.download_docvqa --train 1600 --validation 400 --seed 13
python -m scripts.download_docvqa --validate-only
```

The downloader streams selected TAR shards, stores only the chosen examples, and reuses saved records on reruns. It pins the dataset revision in `data/docvqa/provenance.json`. Sampling uses a seeded shard order and source order within each shard; this is a practical subset, not a globally uniform sample. It preserves the original train/validation assignments and excludes training source documents from validation. Each split has its own `questions.jsonl`, content-hashed images, and original annotations including OCR metadata. `page: 1` identifies the standalone image for the app; `source_page` preserves its page number in the original UCSF document.

Validate checks image hashes, question IDs, labels, annotation alignment, and source-document separation. Large source images require disk space and network access; no Groq calls are made by downloading or validating.

To evaluate later with the selected backend (local for Ollama):

```bash
python -m scripts.evaluate data/docvqa/validation/questions.jsonl --limit 50 --output reports/docvqa-validation.json
```

Dataset source: https://huggingface.co/datasets/pixparse/docvqa-wds . Official challenge: https://rrc.cvc.uab.es/?ch=17 . The mirror uses TAR shards; the preparation helper below accepts the original question JSON annotations and corresponding downloaded images, not TAR shards directly. Obtain those from the official challenge. Keep training and validation sources separate; start with 50 examples before evaluating 2,000–5,000.

```bash
python -m scripts.prepare_subset /path/to/val_v1.0.json /path/to/images --limit 2000 --seed 13
python -m scripts.evaluate data/docvqa-subset/questions.jsonl --limit 50
```

Use labeled training/validation examples; unlabeled test questions cannot produce accuracy scores. The dataset mirror lists MIT licensing; review original image access conditions before redistribution. Downloaded images stay in the ignored local `data/` folder and should not be committed or redistributed without reviewing the original terms.

The evaluation manifest is JSONL with local paths relative to the manifest:

```json
{"id": 1, "document_id": "invoice-1", "document": "images/invoice.png", "page": 1, "question": "What is the total?", "answers": ["220"]}
```

Run the synthetic smoke benchmark with:

```bash
python -m scripts.evaluate examples/demo.jsonl --limit 2 --output reports/demo.json
```

This runs real inference using the selected backend. Ollama runs locally; Groq uses account quota. Reports contain answers, references, exact-match accuracy, normalized edit similarity (ANLS-style, threshold 0.5), per-request latency, extraction method, and error counts for **local text + LLM** versus **images + local text + vision LLM**. Failed requests are reported separately; aggregate accuracy is over successful requests. Preprocessing times are cached per document/page and excluded from request latency. Optional `category` and `fields` labels (field names mapped to reference-answer lists) enable structured field scores. Native DocVQA question labels do not automatically supply those field annotations.

The runner compares an OCR-only heuristic, text + LLM, and vision + text + LLM. QA scores do not measure pure OCR transcript accuracy. The Ollama backend uses the same model for both modes by default. For a controlled Groq comparison, set `GROQ_MODEL` and `GROQ_VISION_MODEL` to the same vision-capable model. Character/word OCR error rates need full ground-truth transcripts; missing-field and consistency precision/recall need manually labeled examples. Those metrics are not inferred from question-answer labels. No benchmark scores are claimed until evaluation has been run.

## OCR evaluation and non-LLM answer baseline

Run local Tesseract evaluation without Groq requests:

```bash
python -m scripts.evaluate_ocr data/docvqa/validation/questions.jsonl --limit 20 --output reports/ocr-validation-20.json
```

The report includes CER/WER transcript comparisons, exact match and ANLS-style QA scores, answer presence in the OCR text, errors, and unique-page OCR time. Transcript metrics use lowercase text with whitespace collapsed and punctuation retained. CER/WER are micro-averaged over unique pages, so repeated questions do not reweight a transcript. Rates can exceed 100% when insertions dominate. Input images use the app's existing orientation/resize/grayscale/autocontrast pipeline.

The OCR-only QA baseline selects a line by meaningful question-word overlap and returns the value after a matching label where available. It uses neither an LLM nor the accepted answers. It is intentionally weak on tables and relationships. Answer presence is a separate oracle diagnostic that uses accepted answers; it is not QA accuracy.

**DocVQA's supplied OCR is a machine reference. Comparisons against it measure agreement, not human-ground-truth accuracy.** For true OCR accuracy, manually transcribe and verify complete page text, then add these fields to each corresponding manifest row:

```json
{"transcript": "transcripts/page-1.txt", "transcript_kind": "human_verified"}
```

The transcript path is relative to the manifest. Do not label copied machine OCR as human-verified. Without verified transcripts, the human-reference CER/WER fields remain null.

The first local smoke run on 20 validation questions completed with no OCR errors. The heuristic QA baseline scored 0% exact match and 0 ANLS; accepted answer strings appeared in 60% of OCR outputs. This small result does not estimate full-benchmark accuracy and illustrates why text recognition alone is insufficient for question answering. The machine-reference agreement rates are reported separately in `reports/ocr-validation-20.json`.

The general `scripts.evaluate` runner now also includes this OCR-only baseline alongside text + LLM and vision + text + LLM. The general runner uses the configured model backend; the OCR-only runner makes no model requests.

## Structure

- `app.py`: Streamlit workflow, page selection, and session state.
- `documind/config.py`: environment configuration.
- `documind/documents.py`: page images, normalization, local OCR and parsing.
- `documind/extraction.py`: standalone text-extraction helper.
- `documind/ai.py`: provider routing, text/vision prompts, retries and JSON parsing.
- `documind/ollama_backend.py`: native local Ollama image and JSON adapter.
- `documind/prompts.py`: classification, extraction and grounding instructions.
- `documind/models.py`: Pydantic document schema.
- `documind/validation.py`: category templates and deterministic validation.
- `documind/ui.py`: styling and field/evidence rendering.
- `documind/evaluation.py`: answer scoring.
- `scripts/`: dataset conversion, synthetic demo and benchmark runner.
- `tests/`: offline regression checks.

```bash
python -m unittest discover -s tests -v
```

## Practical limits

Model-provided page/region descriptions remain unverified until checked. **Locate quote on source** searches parsed/OCR word tokens and highlights a source-derived bounding box when the exact quote matches. An invented quote or unavailable page stays unverified. Located evidence confirms the quote location, not correctness of the extracted value. Confidence labels are estimates, not calibrated probabilities. “Not found” means not extracted, not proof that a field is absent. Inconsistency warnings require human review. Small text, handwriting, poor scans and complex tables can fail. Whole-document analysis is bounded to 30 pages and batches of three. Text-based Q&A retrieval chooses up to three pages; explicitly mentioning a page prioritizes it. Image-only pages without OCR may not rank correctly. Batch validation does not combine unrelated totals, and semantic review does not check contradictions across batches. Each Q&A request is grounded in the current document and question; displayed history is not passed as conversation context.

OCR runs locally. With Ollama, text and images go to the configured local server. With Groq, vision sends selected images and text to Groq; text-only sends extracted text. Keep `.env` private. Copy the complete project, including `documind`, rather than `app.py` alone. Local deployment is ready; Cloud Run hosting is optional and is not configured here.

## Historical Groq comparison status (20-question smoke test)

The live text pipeline completed 20/20 questions: exact match 40%, ANLS-style score 0.561. The vision rerun completed only 5/20 because Groq's daily token quota was nearly exhausted; its 60% exact match / 0.600 ANLS are over just those five responses and are not a comparable full-sample accuracy. See `reports/groq-comparison-20.json` for the paired subset, errors, and settings. These models differ, so this is a pipeline comparison, not an isolated image ablation. Vision's concise output budget is now 512 tokens with Qwen reasoning disabled; the text values were reused from the earlier 4096-token-budget run.

To resume the historical Groq run, select `AI_PROVIDER=groq` first and wait for quota. Do not combine it with Ollama answers:

```bash
python -m scripts.evaluate data/docvqa/validation/questions.jsonl --limit 20 --modes vision --reuse reports/groq-comparison-20.json --retry-failed --output reports/groq-comparison-resumed-20.json
```

`--retry-failed` preserves successful answers. A rate limit stops further calls for that mode; remaining examples are marked not run. OCR runs locally. Quota availability is an account constraint, not evidence that an unanswered document question is wrong.

## Evaluate Gemma locally

Use a fresh report to keep Gemma results separate from historical Groq results:

```bash
python -m scripts.evaluate examples/demo.jsonl --limit 2 --output reports/ollama-demo.json
python -m scripts.evaluate data/docvqa/validation/questions.jsonl --limit 20 --output reports/ollama-validation-20.json
```

These commands run both text-only and vision questions on `gemma3:4b` when `AI_PROVIDER=ollama`. Reusing reports from a different recorded model is rejected.

## Verification and quality suite

```bash
python -m scripts.verify_local
python -m scripts.create_validation_cases
python -m scripts.evaluate_ocr examples/validation/cases.jsonl --limit 3 --output reports/synthetic-ocr.json
python -m scripts.evaluate_quality examples/validation/cases.jsonl --output reports/ollama-quality.json
```

The quality suite measures scalar field exact match, missing-field precision/recall, warning-rule precision/recall, exact quote-location rate, and an automated unsupported-question refusal heuristic. Three generated cases provide known transcripts and labels; their results demonstrate the evaluation machinery, not real-world accuracy. Refusal heuristics can misclassify responses and need review. Real DocVQA question labels do not supply human-verified full transcripts or all field/contradiction annotations.

Field metadata now uses typed Pydantic models (HIGH/MEDIUM/LOW confidence, evidence text, optional positive integer source page, and region). Raw field values are normalized into that schema. Values can still be nested structures to support tables and different document categories.

See `docs/USABILITY.md` for participant tasks; no human study results have been fabricated. See `docs/DEPLOYMENT.md` for the optional container/Cloud Run setup. Deployment files are provided but no cloud service has been published. Cloud Run persistence requires an external database; the local SQLite registry is not persistent multi-instance cloud storage.

## Latest measured results

See [docs/EVALUATION.md](docs/EVALUATION.md). The local 20-question Gemma sample completed with no failed requests: text exact match 30%, vision exact match 65%. These are preliminary sample results. All three synthetic quality cases completed; their perfect fixture scores do not establish real-document accuracy.

Regenerate the readable report from saved JSON files:

```bash
python -m scripts.report_results
```

The local verification and semantic smoke reports are saved in `reports/ollama-smoke.json` and `reports/semantic-smoke.json`. Required templates persist locally in SQLite. Evidence matching derives highlight coordinates from PDF/OCR words, rather than trusting generated coordinates.

## Automatic category discovery

Classification first checks the current persisted list and custom descriptions. If no category fits, a second model call proposes a new reusable document type. It is saved only for a HIGH-confidence proposal with a supporting quote located in the supplied text or page OCR. Otherwise the UI asks the user to choose/add a type. Saved descriptions are included in future classification prompts, and the sidebar list updates after processing. This is persistent prompt context, not model retraining or a guarantee of perfect classification.
