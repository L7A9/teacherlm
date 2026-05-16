# User Course RAG Evaluation Summary

Conversation: `120ec0f7-4a63-4fd7-8d40-e7f4aacb9e2f`

## Ingestion

- Lecture_05.pdf: ready, 16 chunks
- Lecture_04_V2.pdf: ready, 61 chunks
- Lecture_03_V2_organized.pdf: ready, 39 chunks
- Lecture_02.pdf: ready, 39 chunks
- Lecture_01_organized.pdf: ready, 25 chunks
- Guide_for_Students.pdf: ready, 73 chunks

## Prepared Eval Set

- File: `platform/backend/evals/user_course_retrieval_eval.json`
- Cases: 33
- Mix: general overview, lecture-specific, multi-lecture, and out-of-scope questions

## Chat Smoke Results

- What is this course about? Give me a structured overview. -> answered; sources: course_outline, Lecture_01_organized.pdf, Lecture_02.pdf, Lecture_03_V2_organized.pdf, Lecture_04_V2.pdf, Lecture_05.pdf, Guide_for_Students.pdf
- Explique ce cours comme si j etais un etudiant qui commence. -> answered; sources: course_outline, Lecture_01_organized.pdf, Lecture_02.pdf, Lecture_03_V2_organized.pdf, Lecture_04_V2.pdf, Lecture_05.pdf, Guide_for_Students.pdf
- What are data sparsity and cold start in recommender systems? -> answered; sources: Lecture_03_V2_organized.pdf, Lecture_01_organized.pdf
- How is Pearson correlation used in collaborative filtering? -> stream failed/no assistant answer; sources: none

## Findings

- All six PDFs ingested successfully and produced 253 chunks total.
- General overview questions retrieve broad course context from all lectures and the guide.
- The generated overview is too short and partly noisy for a student; it includes outline artifacts such as PEOPLE WHO BOUGHT.
- The French overview prompt answered in English because the test forced language=en; should be retested with language auto/French.
- The data sparsity/cold start answer retrieved Lecture 01 but ranked Lecture 03 too heavily and claimed sparsity was only implied, although Lecture 01 explicitly covers it.
- The backend stream dropped during the Pearson-correlation question after persisting the user message; this matches the earlier Failed to fetch symptom and points to runtime/model stability under long local-model calls.
- The standalone retrieval eval process is too heavy in this current Docker setup: one run was killed, the reduced run hung; keep Ragas/retrieval eval isolated or make a lighter API-backed eval runner.
