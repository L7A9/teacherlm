# Mind Map Generator — CLAUDE.md

## Port: 8008 | Python 3.14+
## Purpose: Generate hierarchical mind maps (cartes mentales) from 
## uploaded course materials, rendered as interactive Mermaid mindmaps.

## Product Role
Students use mind maps to get a BIRD'S-EYE VIEW of an entire subject.
Unlike the Chart Generator (which makes flowcharts/diagrams/timelines 
for specific relationships), this generator produces:
- A central topic node
- 3-7 main branches (high-level themes)
- Each branch has 2-5 sub-branches (concepts within the theme)
- Each sub-branch has 1-4 leaf nodes (specific facts, terms, examples)
- Depth: 3-4 levels total (including root)

## When Student Uses This
- "Give me an overview of everything in these files"
- Studying for exams — sees the whole subject at a glance
- Before diving into details — gets mental structure first
- Different from Chart Gen which needs a specific relationship to show

## Stack
- FastAPI, ollama, teacherlm_core
- pydantic >=2.12 (for structured hierarchical output)
- jinja2 (interactive HTML with Mermaid rendering)
- No LLM-based Mermaid generation — we build the Mermaid string 
  from validated Pydantic hierarchy. This GUARANTEES valid output.

## Key Architectural Decision
Unlike Chart Generator (which asks LLM for raw Mermaid code), this 
generator asks the LLM for a STRUCTURED hierarchy via Pydantic, then 
programmatically converts that hierarchy to Mermaid syntax.

Why: Mind map quality depends on balanced, well-organized hierarchies.
Getting the LLM to output valid nested Mermaid is error-prone. Getting 
it to output nested JSON (validated by Pydantic) is reliable.

## Module Map (files < 250 lines)
mindmap_gen/
├── CLAUDE.md
├── app.py                         # FastAPI /run /health /info
├── config.py
├── schemas.py                     # MindMapNode (recursive), MindMap
├── pipeline.py
├── services/
│   ├── theme_extractor.py         # extracts high-level themes from chunks
│   ├── hierarchy_builder.py       # recursive LLM-driven tree building
│   ├── balancer.py                # ensures balanced, readable structure
│   ├── mermaid_compiler.py        # Pydantic tree → Mermaid string
│   ├── html_renderer.py           # Jinja2 interactive HTML
│   └── llm_service.py
├── prompts/
│   ├── theme_extraction.txt       # get main branches from content
│   ├── subtopic_expansion.txt     # expand each branch into sub-branches
│   └── leaf_details.txt           # add specific facts/terms to leaves
├── templates/
│   └── mindmap.html.jinja         # Mermaid.js mindmap renderer
├── artifacts/                     # generated .html and .mmd files
├── requirements.txt
├── Dockerfile
└── README.md

## Output Contract
Follows teacherlm_core GeneratorInput/Output. Specifically:
- output_type: "mindmap"
- artifacts: 
    [{type: "html", url: ".../mindmap_{id}.html", filename: "mindmap_{id}.html"},
     {type: "mermaid", url: ".../mindmap_{id}.mmd", filename: "mindmap_{id}.mmd"}]
- metadata:
    {
      "mermaid_code": str,
      "node_count": int,
      "depth": int,
      "central_topic": str,
      "main_branches": list[str]
    }
- learner_updates:
    concepts_covered = all node texts

## Retrieval Mode
This generator expects platform to use "topic_clusters" retrieval mode.
It needs broad coverage of ALL document topics, not narrow query matching.

## Language Support
Follow the language of the source content (French content → French 
mind map). The LLM handles this automatically when prompts instruct it to.

## Mind Map Size Options (from input.options)
- options.size = "concise" → 3-5 main branches, depth 3
- options.size = "standard" → 5-7 main branches, depth 3-4 (default)
- options.size = "comprehensive" → 7-10 main branches, depth 4
- options.max_nodes = hard cap (default 60 to keep readable)

## Mermaid Mindmap Syntax Reminder
mindmap
  root((Central Topic))
    Branch 1
      Subtopic 1.1
        Leaf detail
        Another leaf
      Subtopic 1.2
    Branch 2
      Subtopic 2.1

Indentation matters — 2 spaces per level. No arrows, no shapes on 
leaves (Mermaid mindmaps use hierarchy via indentation alone).