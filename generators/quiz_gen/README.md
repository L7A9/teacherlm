# quiz_gen

`quiz_gen` creates grounded quizzes from retrieved course context. It is designed for broad coverage across the selected ready source files.

## Service

Default port: `8002`

Endpoints:

- `GET /health`
- `GET /info`
- `POST /run`

`/run` streams server-sent events.

Common events:

- `progress`: generation stage updates.
- `token` or `chunk`: streamed response text when emitted.
- `artifact`: generated artifact metadata.
- `done`: final generator output.
- `error`: failure details.

## Generator Info

| Field | Value |
| --- | --- |
| Generator id | `quiz_gen` |
| Output type | `quiz` |
| Retrieval mode | `coverage_broad` |

The backend retrieves broad course coverage before calling this generator. If the student selected specific source files in the frontend, the backend filters retrieval to those files first.

## Quiz Options

Supported request options include:

| Option | Notes |
| --- | --- |
| `question_count` | Default `8`, minimum `3`, maximum `30` |
| `n_questions` | Alias for `question_count` |
| `count` | Alias for `question_count` |
| `question_types` | Values can include `mcq`, `true_false`, `fill_blank` |

Frontend aliases are normalized:

- `multiple_choice` becomes `mcq`.
- `short_answer` becomes `fill_blank`.

The generator uses course concepts, learner state, and Bloom-level planning to shape the quiz. It validates questions, removes weak duplicates, and can top up with grounded true/false questions if needed.

## Output

The final response contains:

- A markdown quiz summary.
- `metadata.quiz_data` with the generated quiz structure.
- Planning metadata such as Bloom distribution and dropped/top-up question counts.
- A JSON artifact uploaded to MinIO when artifact storage is available.
- Learner updates for concepts covered by the quiz.

Questions are grounded in uploaded course chunks. Each multiple-choice question has one correct answer and plausible distractors.

## Environment

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `QUIZ_GEN_OLLAMA_URL` | Ollama base URL override |
| `QUIZ_GEN_MODEL` | quiz-generation model |
| `QUIZ_GEN_DEFAULT_QUESTION_COUNT` | default number of questions |
| `QUIZ_GEN_MIN_QUESTIONS` | lower bound |
| `QUIZ_GEN_MAX_QUESTIONS` | upper bound |
| `QUIZ_GEN_ENHANCE_DISTRACTORS` | enable extra distractor pass |
| `MINIO_ENDPOINT` | artifact storage endpoint |
| `MINIO_ACCESS_KEY` | artifact storage access key |
| `MINIO_SECRET_KEY` | artifact storage secret key |
| `MINIO_BUCKET` | artifact bucket |

## Local Run

From this directory:

```bash
pip install -e ../../packages/teacherlm_core
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8002
```

Through Docker Compose:

```bash
cd ../../platform
docker compose up -d quiz_gen
```

## Tests

From the repository root:

```bash
pytest generators/quiz_gen/tests
```
