"use client";

import { useMemo } from "react";

import {
  BarChart3,
  FileText,
  GraduationCap,
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
  { outputType: "chart", label: "Diagram", hint: "Concept diagram", Icon: BarChart3 },
  { outputType: "mindmap", label: "Mind map", hint: "Bird's-eye view of your materials", Icon: Network },
  { outputType: "podcast", label: "Podcast", hint: "Listen-along audio", Icon: Mic2 },
  { outputType: "presentation", label: "Presentation", hint: "Slide deck", Icon: Presentation },
];

interface Props {
  onSelectChat?: () => void;
  className?: string;
  disabled?: boolean;
  disabledReason?: string;
}

export function OutputTypeButtons({
  onSelectChat,
  className,
  disabled = false,
  disabledReason = "Upload at least one course file first.",
}: Props) {
  const openDialog = useUiStore((s) => s.openGeneratorDialog);
  const { data } = useGenerators(false);

  const activeGenerators = useMemo(() => {
    const map = new Map<string, { icon: string | null | undefined }>();
    for (const g of data?.items ?? []) {
      map.set(g.output_type, { icon: g.icon });
    }
    return map;
  }, [data]);

  const visibleButtons = useMemo(
    () => BUTTONS.filter((button) => activeGenerators.has(button.outputType)),
    [activeGenerators],
  );

  return (
    <TooltipProvider delayDuration={200}>
      <div
        className={cn(
          "flex items-center gap-1.5 overflow-x-auto pb-0.5",
          className,
        )}
      >
        {visibleButtons.map(({ outputType, label, hint, Icon }) => {
          const meta = activeGenerators.get(outputType);

          const handleClick = () => {
            if (disabled) return;
            if (outputType === "text") {
              onSelectChat?.();
              return;
            }
            openDialog(outputType);
          };

          return (
            <Tooltip key={outputType}>
              <TooltipTrigger asChild>
                <span className="inline-flex shrink-0">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={handleClick}
                    disabled={disabled}
                    aria-label={label}
                    title={disabled ? disabledReason : hint}
                    className={cn(
                      "h-8 shrink-0 gap-1.5 px-2.5",
                      outputType === "text" &&
                        "border-primary/40 bg-primary/10",
                    )}
                  >
                    {meta?.icon ? (
                      <span className="text-base leading-none">{meta.icon}</span>
                    ) : (
                      <Icon className="h-4 w-4" />
                    )}
                    <span className="hidden md:inline">{label}</span>
                  </Button>
                </span>
              </TooltipTrigger>
              <TooltipContent>
                {disabled ? disabledReason : `${label} - ${hint}`}
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>
    </TooltipProvider>
  );
}
