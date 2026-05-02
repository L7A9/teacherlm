from pydantic import BaseModel, Field


class MindMapNode(BaseModel):
    """Recursive node. Children list can contain more nodes (up to
    ~4 levels deep including root)."""

    text: str = Field(
        ...,
        max_length=80,
        description="Short label, max 8 words. Keep it concise.",
    )
    children: list["MindMapNode"] = Field(
        default_factory=list,
        description="Sub-nodes. Empty for leaf nodes.",
    )


MindMapNode.model_rebuild()


class MindMap(BaseModel):
    central_topic: str = Field(
        ...,
        max_length=60,
        description="The central topic (root of the mind map)",
    )
    branches: list[MindMapNode] = Field(
        ...,
        min_length=3,
        max_length=10,
        description="Main branches radiating from center. 3-10 of them.",
    )


class ThemeList(BaseModel):
    """Intermediate: LLM returns list of main themes first."""

    themes: list[str] = Field(..., min_length=3, max_length=10)


class CourseOutline(BaseModel):
    """One-shot full-course outline for a coherent mind map."""

    central_topic: str = Field(
        ...,
        max_length=60,
        description="The central course subject",
    )
    branches: list[MindMapNode] = Field(
        ...,
        min_length=3,
        max_length=10,
        description="Major course modules with subtopics and leaf concepts",
    )


class SubtopicExpansion(BaseModel):
    """Intermediate: expanding one branch into subtopics + leaves."""

    subtopics: list[MindMapNode]
