# Evaluation status

These are measured smoke-test results, not full-corpus or production accuracy. No human usability study or cloud deployment is claimed.

## Local model verification

Model: `gemma3:4b`

| Check | Passed | Seconds |
|---|---|---:|
| text_qa | True | 15.21 |
| image_only_qa | True | 11.34 |
| classification | True | 10.89 |
| structured_extraction | True | 49.69 |

## DocVQA validation sample

Questions: 20; same `gemma3:4b` model for text and vision. Dataset source revision is recorded in `data/docvqa/provenance.json`.

| Pipeline | Succeeded | Failed | Exact match | ANLS-style score | Mean request seconds |
|---|---:|---:|---:|---:|---:|
| ocr_only | 20 | 0 | 0.0% | 0.000 | 0.00 |
| text | 20 | 0 | 30.0% | 0.554 | 12.69 |
| vision | 20 | 0 | 65.0% | 0.771 | 25.27 |

Accuracy is over successful requests; failed requests are reported separately. Timing excludes preprocessing and includes backend/model loading or queueing when present. A small seeded-shard sample is not a statistically representative full DocVQA benchmark.

## Synthetic OCR ground truth

Pages: 3. CER: 0.0%; WER: 0.0%. References are the exact generated text used to render these clean pages. This does not replace human transcription of real documents.

## Synthetic field and quality suite

Completed: 3/3; failures: 0. Fixtures: clean invoice, inconsistent invoice total, and form with blank address/signature.

| Metric | Result |
|---|---:|
| Field exact match | 100.0% |
| Missing-field precision | 100.0% |
| Missing-field recall | 100.0% |
| Consistency warning precision | 100.0% |
| Consistency warning recall | 100.0% |
| Evidence quote-location rate | 100.0% |
| Unsupported-question refusal heuristic | 100.0% |

Quote location confirms literal source text, not correctness of an extracted value. Refusal scoring is an automated phrase heuristic and requires review. This three-case suite is deliberately small and synthetic.

## Remaining external validation

- Real-document CER/WER requires verified transcripts; machine-reference agreement is reported separately.
- Real field, missing-field, and semantic-contradiction accuracy needs a larger independently labeled dataset.
- Use `docs/USABILITY.md` to record actual participant results; automated interface tests are not a human study.
- `Dockerfile` and `docs/DEPLOYMENT.md` prepare optional deployment. No cloud project, service, or persistent cloud database has been provisioned.
