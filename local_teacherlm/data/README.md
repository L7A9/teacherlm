# Local Data Layout

Runtime data is not stored in this repository. The desktop app uses the OS
app-data directory by default.

Windows default:

```text
%APPDATA%/TeacherLM/
```

Local development can override this path with:

```text
TEACHERLM_APP_DATA_DIR=C:\path\to\TeacherLM-dev-data
```

Runtime layout:

```text
TeacherLM/
  teacherlm.db
  objects/
    uploads/
    parsed/
    cleaned/
  artifacts/
    quizzes/
    mindmaps/
    podcasts/
    presentations/
    reports/
    charts/
  indexes/
    vector/
    bm25/
    graph/
  models/
    embeddings/
    rerankers/
    tts/
  logs/
  traces/
  plugins/
    generators/
    mcp/
```

Secrets are encrypted before being stored in SQLite. Production packaging should
move the encryption root into Tauri Stronghold or the OS keychain.

