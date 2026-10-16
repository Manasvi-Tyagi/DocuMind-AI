# Usability acceptance checks

Automated UI checks exercise provider switching, document-state rendering, required fields, and question submission with mocked model replies. They do not establish real user usability or visual accessibility.

Use this checklist with at least three classmates. Give each person the synthetic examples without coaching, record elapsed time and errors, and do not include personal information in shared reports.

| Task | Expected result | Success? | Seconds | Notes |
|---|---|---|---|---|
| Select Ollama or Groq | Correct backend caption and data-flow explanation | | | |
| Upload demo invoice and process | Invoice detected; page preview available | | | |
| Extract insights | INV-1045 and total 220 visible | | | |
| Locate a quote | Matching source region highlighted, or explicit unverified notice | | | |
| Ask the total | Answer 220 with source context | | | |
| Ask an unsupported question | Explicit uncertainty; no invented account number | | | |
| Use an unknown category | Add type and reuse it after restart | | | |
| Save required fields | Template is available for a later upload | | | |
| Review inconsistent invoice | Subtotal + tax warning appears | | | |
| Analyze a PDF with four pages | Batch results and retrieval scope are understandable | | | |
| Switch backend | Old insights/chat clear; document remains | | | |

Also check keyboard access, labels, readable text and buttons, narrow-window layout, empty uploads, OCR failures, stopped Ollama, and missing Groq credentials. Record perceived ease from 1–5 and the most confusing action. Report actual task success rates and median time; leave results blank until participants complete the tasks.
