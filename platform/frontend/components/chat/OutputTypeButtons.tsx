"use client";

import { useMemo } from "react";

import {
  BarChart3,
  FileText,
  GraduationCap,
  Layers,
  Mic2,
  Network,
  Presentation,
  ScrollText,
} from "lucide-react";

import { Button } from "@/components/ui/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/Tooltip";
import { useGenerators } from "@/hooks/useGenerators";
import type { OutputType } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useUiStore } from "@/stores/uiStore";

interface ButtonSpec {
  outputType: OutputType;
  label: string;
  hint: string;
  Icon: React.ComponentType<{ className?: string }>;
}

const BUTTONS: ButtonSpec[] = [
  { outputType: "text", label: "Chat", hint: "Talk to your teacher", Icon: GraduationCap },
  { outputType: "quiz", label: "Quiz", hint: "Test yourself", Icon: FileText },
  { outputType: "report", label: "Report", hint: "Study report", Icon: ScrollText },
  { outputType: "flashcards", label: "Flashcards", hint: "Spaced-repetition cards", Icon: Layers },
  { outputType: "chart", label: "Diagram", hint: "Concept diagram", Icon: BarChart3 },
  { outputType: "mindmap", label: "Mind map", hint: "Bird's-eye view of your materials", Icon: Network },
  { outputType: "podcast", label: "Podcast", hint: "Listen-along audio", Icon: Mic2 },
  { outputType: "presentation", label: "Presentation", hint: "Slide deck", Icon: Presentation },
];

interface Props {
  onSelectChat?: () => void;
  className?: string;
}

export function OutputTypeButtons({ onSelectChat, className }: Props) {
  const openDialog = useUiStore((s) => s.openGeneratorDialog);
  const { data } = useGenerators(true);

  const availability = useMemo(() => {
    const map = new Map<string, { enabled: boolean; icon: string | null | undefined }>();
    for (const g of data?.items ?? []) {
      map.set(g.output_type, { enabled: g.enabled, icon: g.icon });
    }
    return map;
  }, [data]);

  return (
    <TooltipProvider delayDuration={200}>
      <div className={cn("flex flex-wrap items-center gap-1.5", className)}>
        {BUTTONS.map(({ outputType, label, hint, Icon }) => {
          const meta = availability.get(outputType);
          const registered = meta !== undefined;
          const enabled = outputType === "text" ? true : meta?.enabled === true;
          const disabled = registered && !enabled;

          const handleClick = () => {
            if (outputType === "text") {
              onSelectChat?.();
              return;
            }
            openDialog(outputType);
          };

          return (
            <Tooltip key={outputType}>
              <TooltipTrigger asChild>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={handleClick}
                  disabled={disabled}
                  aria-label={label}
                  className={cn(
                    "h-9 gap-1.5 px-3",
                    outputType === "text" && "border-primary/40 bg-primary/10",
                  )}
                >
                  {meta?.icon ? (
                    <span className="text-base leading-none">{meta.icon}</span>
                  ) : (
                    <Icon className="h-4 w-4" />
                  )}
                  <span className="hidden md:inline">{label}</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {disabled
                  ? `${label} · coming soon`
                  : `${label} — ${hint}`}
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>
    </TooltipProvider>
  );
}
