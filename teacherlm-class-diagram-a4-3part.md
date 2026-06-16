# TeacherLM Class Diagram - 3 A4 Pages

## Page 1 of 3 - Conversation, Files, Documents, Sections, Chunks

```mermaid
%%{init: {
  "theme": "base",
  "themeCSS": ".classTitle, .classGroup text { font-size: 26px !important; font-weight: 800 !important; } .classGroup rect { stroke-width: 3px !important; rx: 8px !important; ry: 8px !important; } .edgeLabel, .label, .relation { font-size: 18px !important; font-weight: 700 !important; }",
  "themeVariables": {
    "background": "#ffffff",
    "primaryColor": "#f8fafc",
    "primaryTextColor": "#111827",
    "primaryBorderColor": "#111827",
    "lineColor": "#374151",
    "edgeLabelBackground": "#f3f4f6",
    "tertiaryColor": "#f8fafc",
    "tertiaryTextColor": "#111827",
    "tertiaryBorderColor": "#111827"
  },
  "classDiagram": {
    "useMaxWidth": true,
    "fontSize": 26,
    "titleFontSize": 28
  }
}}%%

classDiagram
  direction TB

  class Conversation
  class Message
  class UploadedFile
  class CourseDocument
  class CourseSection
  class SearchChunk
  class LearnerState

  %% Relationships
  Conversation "1" --> "0..*" Message : has
  Conversation "1" --> "0..*" UploadedFile : owns
  Conversation "1" --> "0..*" CourseDocument : contains
  Conversation "1" --> "0..*" SearchChunk : indexes
  Conversation "1" --> "1" LearnerState : stores

  UploadedFile "1" --> "1" CourseDocument : parses_into
  CourseDocument "1" --> "0..*" CourseSection : has
  CourseDocument "1" --> "0..*" SearchChunk : chunks_into
  CourseSection "1" --> "0..*" SearchChunk : produces
  CourseSection "0..1" --> "0..*" CourseSection : parent_of
```

## Page 2 of 3 - Learning Path, Lessons, Quizzes, Knowledge Checks

```mermaid
%%{init: {
  "theme": "base",
  "themeCSS": ".classTitle, .classGroup text { font-size: 26px !important; font-weight: 800 !important; } .classGroup rect { stroke-width: 3px !important; rx: 8px !important; ry: 8px !important; } .edgeLabel, .label, .relation { font-size: 18px !important; font-weight: 700 !important; }",
  "themeVariables": {
    "background": "#ffffff",
    "primaryColor": "#f8fafc",
    "primaryTextColor": "#111827",
    "primaryBorderColor": "#111827",
    "lineColor": "#374151",
    "edgeLabelBackground": "#f3f4f6",
    "tertiaryColor": "#f8fafc",
    "tertiaryTextColor": "#111827",
    "tertiaryBorderColor": "#111827"
  },
  "classDiagram": {
    "useMaxWidth": true,
    "fontSize": 26,
    "titleFontSize": 28
  }
}}%%

classDiagram
  direction TB

  class Conversation
  class CourseConcept
  class CourseLearningPhase
  class CourseLearningObjective
  class CourseChapter
  class CourseLesson
  class CourseLessonBlock
  class ChapterQuiz
  class ChapterAttempt
  class KnowledgeCheck
  class KnowledgeAttempt

  %% Relationships
  Conversation "1" --> "0..*" CourseConcept : extracts
  Conversation "1" --> "0..*" CourseLearningPhase : structures
  Conversation "1" --> "0..*" CourseLearningObjective : defines
  Conversation "1" --> "0..*" CourseChapter : organizes
  Conversation "1" --> "0..*" CourseLesson : contains
  Conversation "1" --> "0..*" CourseLessonBlock : has
  Conversation "1" --> "0..*" ChapterQuiz : generates
  Conversation "1" --> "0..*" ChapterAttempt : records
  Conversation "1" --> "0..*" KnowledgeCheck : creates
  Conversation "1" --> "0..*" KnowledgeAttempt : tracks

  CourseLearningPhase "1" --> "0..*" CourseLearningObjective : contains
  CourseLearningPhase "1" --> "0..*" CourseChapter : organizes
  CourseChapter "1" --> "0..*" CourseLesson : contains
  CourseLearningObjective "1" --> "0..*" CourseLesson : targets
  CourseLesson "1" --> "0..*" CourseLessonBlock : contains
  CourseChapter "1" --> "0..1" ChapterQuiz : has
  ChapterQuiz "1" --> "0..*" ChapterAttempt : attempted_by
  CourseChapter "1" --> "0..*" ChapterAttempt : receives

  CourseConcept "1" --> "0..*" KnowledgeCheck : assessed_by
  CourseConcept "1" --> "0..*" KnowledgeAttempt : attempted_for
  KnowledgeCheck "1" --> "0..*" KnowledgeAttempt : answered_by
```

## Page 3 of 3 - Knowledge Graph, Course Builder, Runtime Settings

```mermaid
%%{init: {
  "theme": "base",
  "themeCSS": ".classTitle, .classGroup text { font-size: 26px !important; font-weight: 800 !important; } .classGroup rect { stroke-width: 3px !important; rx: 8px !important; ry: 8px !important; } .edgeLabel, .label, .relation { font-size: 18px !important; font-weight: 700 !important; }",
  "themeVariables": {
    "background": "#ffffff",
    "primaryColor": "#f8fafc",
    "primaryTextColor": "#111827",
    "primaryBorderColor": "#111827",
    "lineColor": "#374151",
    "edgeLabelBackground": "#f3f4f6",
    "tertiaryColor": "#f8fafc",
    "tertiaryTextColor": "#111827",
    "tertiaryBorderColor": "#111827"
  },
  "classDiagram": {
    "useMaxWidth": true,
    "fontSize": 26,
    "titleFontSize": 28
  }
}}%%

classDiagram
  direction TB

  class Conversation
  class KnowledgeNode
  class KnowledgeEdge
  class CourseBuilderCourse
  class CourseBuilderChapter
  class CourseBuilderLesson
  class CourseBuilderLessonBlock
  class CourseBuilderQuiz
  class CourseBuilderQuizQuestion
  class CourseBuilderChapterAttempt
  class AppRuntimeSettings

  %% Relationships
  Conversation "1" --> "0..*" KnowledgeNode : maps
  Conversation "1" --> "0..*" KnowledgeEdge : links
  Conversation "1" --> "0..1" CourseBuilderCourse : builds

  KnowledgeNode "1" --> "0..*" KnowledgeEdge : source
  KnowledgeNode "1" --> "0..*" KnowledgeEdge : target

  CourseBuilderCourse "1" --> "0..*" CourseBuilderChapter : has
  CourseBuilderCourse "1" --> "0..*" CourseBuilderLesson : has
  CourseBuilderCourse "1" --> "0..*" CourseBuilderQuiz : has
  CourseBuilderCourse "1" --> "0..*" CourseBuilderChapterAttempt : records
  CourseBuilderChapter "1" --> "0..*" CourseBuilderLesson : contains
  CourseBuilderChapter "1" --> "0..1" CourseBuilderQuiz : has
  CourseBuilderChapter "1" --> "0..*" CourseBuilderQuizQuestion : contains
  CourseBuilderChapter "1" --> "0..*" CourseBuilderChapterAttempt : attempted
  CourseBuilderLesson "1" --> "0..*" CourseBuilderLessonBlock : contains
  CourseBuilderQuiz "1" --> "0..*" CourseBuilderQuizQuestion : contains
  CourseBuilderQuiz "1" --> "0..*" CourseBuilderChapterAttempt : attempted_by
```
