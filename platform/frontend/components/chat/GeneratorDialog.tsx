"use client";

import { useEffect, useState } from "react";

import { Sparkles } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/Button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/Dialog";
import { Input, Label } from "@/components/ui/Input";
import { useGenerateStream } from "@/hooks/useChatStream";
import type { OutputType } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useConversationStore } from "@/stores/conversationStore";
import { useUiStore } from "@/stores/uiStore";

const TITLES: Record<OutputType, string> = {
  text: "Ask your teacher",
  quiz: "Generate a quiz",
  report: "Generate a report",
  presentation: "Generate a presentation",
  flashcards: "Generate flashcards",
  chart: "Generate a diagram",
  podcast: "Generate a podcast",
};

const DESCRIPTIONS: Record<OutputType, string> = {
  text: "Type a question and your teacher will answer it.",
  quiz: "Pick how many questions, difficulty, and which types to include.",
  report: "Choose a format and how deep the write-up should go.",
  presentation: "Choose a length and style for your slide deck.",
  flashcards: "Choose a count and what the cards should focus on.",
  chart: "Pick a diagram style — the teacher will extract the relationships.",
  podcast: "Pick a duration and presenter style.",
};

export function GeneratorDialog() {
  const { open, outputType } = useUiStore((s) => s.generatorDialog);
  const closeDialog = useUiStore((s) => s.closeGeneratorDialog);
  const activeConversationId = useConversationStore((s) => s.activeConversationId);
  const runGenerate = useGenerateStream();

  if (!outputType) {
    return <Dialog open={open} onOpenChange={(v) => !v && closeDialog()}><span /></Dialog>;
  }

  const handleSubmit = async (options: Record<string, unknown>, topic: string) => {
    if (!activeConversationId) {
      toast.error("No active conversation");
      return;
    }
    closeDialog();
    try {
      await runGenerate(activeConversationId, {
        output_type: outputType,
        options,
        topic: topic.trim() || null,
      });
    } catch (err) {
      toast.error(`Generation failed: ${(err as Error).message}`);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(v) => !v && closeDialog()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{TITLES[outputType]}</DialogTitle>
          <DialogDescription>{DESCRIPTIONS[outputType]}</DialogDescription>
        </DialogHeader>
        <OptionsForm outputType={outputType} onSubmit={handleSubmit} onCancel={closeDialog} />
      </DialogContent>
    </Dialog>
  );
}

// ---------- forms ----------

interface FormProps {
  outputType: OutputType;
  onSubmit: (options: Record<string, unknown>, topic: string) => void;
  onCancel: () => void;
}

function OptionsForm({ outputType, onSubmit, onCancel }: FormProps) {
  const [topic, setTopic] = useState("");
  const [values, setValues] = useState<Record<string, unknown>>(() =>
    defaultsFor(outputType),
  );

  useEffect(() => {
    setValues(defaultsFor(outputType));
    setTopic("");
  }, [outputType]);

  const fields = FIELDS_BY_TYPE[outputType];

  return (
    <form
      className="flex flex-col gap-4"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(values, topic);
      }}
    >
      <div className="flex flex-col gap-3">
        {fields.map((field) => (
          <Field
            key={field.name}
            field={field}
            value={values[field.name]}
            onChange={(v) => setValues((prev) => ({ ...prev, [field.name]: v }))}
          />
        ))}

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="generator-topic">Topic (optional)</Label>
          <Input
            id="generator-topic"
            placeholder="e.g. photosynthesis, recursion, the French Revolution"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
          />
          <p className="text-[11px] text-muted-foreground">
            Narrows retrieval to content about this topic.
          </p>
        </div>
      </div>

      <DialogFooter>
        <Button type="button" variant="secondary" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit">
          <Sparkles className="h-4 w-4" />
          Generate
        </Button>
      </DialogFooter>
    </form>
  );
}

function Field({
  field,
  value,
  onChange,
}: {
  field: FieldSpec;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={`opt-${field.name}`}>{field.label}</Label>
      {field.kind === "number" && (
        <Input
          id={`opt-${field.name}`}
          type="number"
          min={field.min}
          max={field.max}
          step={field.step ?? 1}
          value={Number(value ?? field.default)}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      )}
      {field.kind === "select" && (
        <select
          id={`opt-${field.name}`}
          value={String(value ?? field.default)}
          onChange={(e) => onChange(e.target.value)}
          className={cn(
            "h-9 rounded-md border border-border bg-background px-3 text-sm",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          )}
        >
          {field.options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      )}
      {field.kind === "multiselect" && (
        <div className="flex flex-wrap gap-1.5">
          {field.options.map((opt) => {
            const selected = Array.isArray(value) && value.includes(opt.value);
            return (
              <button
                key={opt.value}
                type="button"
                onClick={() => {
                  const list = Array.isArray(value) ? [...(value as string[])] : [];
                  const next = selected
                    ? list.filter((v) => v !== opt.value)
                    : [...list, opt.value];
                  onChange(next);
                }}
                className={cn(
                  "rounded-full border px-2.5 py-0.5 text-xs transition-colors",
                  selected
                    ? "border-primary bg-primary/15 text-primary"
                    : "border-border text-muted-foreground hover:border-primary/40 hover:text-foreground",
                )}
              >
                {opt.label}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ---------- field specs ----------

type Option = { value: string; label: string };

type FieldSpec =
  | {
      kind: "number";
      name: string;
      label: string;
      default: number;
      min?: number;
      max?: number;
      step?: number;
    }
  | {
      kind: "select";
      name: string;
      label: string;
      default: string;
      options: Option[];
    }
  | {
      kind: "multiselect";
      name: string;
      label: string;
      default: string[];
      options: Option[];
    };

const FIELDS_BY_TYPE: Record<OutputType, FieldSpec[]> = {
  text: [],
  quiz: [
    { kind: "number", name: "question_count", label: "Number of questions", default: 8, min: 1, max: 30 },
    {
      kind: "select",
      name: "difficulty",
      label: "Difficulty",
      default: "medium",
      options: [
        { value: "easy", label: "Easy" },
        { value: "medium", label: "Medium" },
        { value: "hard", label: "Hard" },
      ],
    },
    {
      kind: "multiselect",
      name: "question_types",
      label: "Question types",
      default: ["multiple_choice"],
      options: [
        { value: "multiple_choice", label: "Multiple choice" },
        { value: "true_false", label: "True / false" },
        { value: "short_answer", label: "Short answer" },
      ],
    },
  ],
  report: [
    {
      kind: "select",
      name: "format",
      label: "Format",
      default: "structured",
      options: [
        { value: "structured", label: "Structured (headings + bullets)" },
        { value: "narrative", label: "Narrative prose" },
        { value: "outline", label: "Outline only" },
      ],
    },
    {
      kind: "select",
      name: "depth",
      label: "Depth",
      default: "standard",
      options: [
        { value: "brief", label: "Brief" },
        { value: "standard", label: "Standard" },
        { value: "deep", label: "Deep dive" },
      ],
    },
  ],
  presentation: [
    { kind: "number", name: "slide_count", label: "Slides", default: 10, min: 3, max: 30 },
    {
      kind: "select",
      name: "style",
      label: "Style",
      default: "academic",
      options: [
        { value: "academic", label: "Academic" },
        { value: "minimal", label: "Minimal" },
        { value: "storytelling", label: "Storytelling" },
      ],
    },
  ],
  flashcards: [
    { kind: "number", name: "count", label: "Number of cards", default: 12, min: 3, max: 50 },
    {
      kind: "select",
      name: "focus",
      label: "Focus",
      default: "key_concepts",
      options: [
        { value: "key_concepts", label: "Key concepts" },
        { value: "definitions", label: "Definitions" },
        { value: "struggling", label: "Concepts you're struggling with" },
      ],
    },
  ],
  podcast: [
    { kind: "number", name: "duration_minutes", label: "Duration (minutes)", default: 8, min: 2, max: 30 },
    {
      kind: "select",
      name: "style",
      label: "Presenter style",
      default: "two_host",
      options: [
        { value: "solo", label: "Solo narrator" },
        { value: "two_host", label: "Two-host conversation" },
        { value: "interview", label: "Interview" },
      ],
    },
  ],
  chart: [
    {
      kind: "select",
      name: "diagram_type",
      label: "Diagram type",
      default: "flowchart",
      options: [
        { value: "flowchart", label: "Flowchart" },
        { value: "mindmap", label: "Mind map" },
        { value: "sequence", label: "Sequence" },
        { value: "class", label: "Class / entity" },
        { value: "state", label: "State machine" },
      ],
    },
  ],
};

function defaultsFor(outputType: OutputType): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const field of FIELDS_BY_TYPE[outputType]) {
    out[field.name] = field.default;
  }
  return out;
}
