"use client";

import { useState } from "react";

import { ChevronDown, ChevronRight, FileText } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { AssistantMarkdown } from "@/components/chat/MessageBubble";
import type { CourseBuilderLessonBlock } from "@/lib/types";
import { cn } from "@/lib/utils";

import { CourseBuilderCitationList } from "./CitationList";

interface Props {
  block: CourseBuilderLessonBlock;
  index?: number;
  open?: boolean;
  onToggle?: () => void;
}

export function LessonBlockRenderer({
  block,
  index,
  open: controlledOpen,
  onToggle,
}: Props) {
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const open = controlledOpen ?? uncontrolledOpen;
  const toggleOpen = onToggle ?? (() => setUncontrolledOpen((value) => !value));
  const title = block.title?.trim() || blockTypeLabel(block.block_type);

  return (
    <article className="content-selectable overflow-hidden rounded-md border border-border bg-surface">
      <button
        type="button"
        className={cn(
          "flex w-full items-start gap-2 px-3 py-2 text-left text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          open ? "bg-primary/10 text-primary" : "text-foreground hover:bg-muted/60",
        )}
        onClick={toggleOpen}
        aria-expanded={open}
      >
        {typeof index === "number" && (
          <span
            className={cn(
              "flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold",
              open ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
            )}
          >
            {index + 1}
          </span>
        )}
        <FileText
          className={cn(
            "mt-0.5 h-3.5 w-3.5 shrink-0",
            open ? "text-primary" : "text-muted-foreground",
          )}
        />
        <span className="min-w-0 flex-1 line-clamp-2 font-semibold leading-4">
          {title}
        </span>
        {open ? (
          <ChevronDown className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        )}
      </button>

      {open && (
        <div className="flex flex-col gap-3 border-t border-border px-3 py-3">
          <BlockBody block={block} />
          <CourseBuilderCitationList citations={block.source_citations} />
        </div>
      )}
    </article>
  );
}

function BlockBody({ block }: Props) {
  if (block.block_type === "table") return <TableBlock data={block.data_json} content={block.content} />;
  if (block.block_type === "equation") return <AssistantMarkdown content={mathBlock(block.content)} />;
  if (block.block_type === "chart") return <ChartBlock data={block.data_json} fallback={block.content} />;
  if (block.block_type === "diagram") {
    return <AssistantMarkdown content={`\`\`\`mermaid\n${block.content}\n\`\`\``} />;
  }
  return (
    <div className="course-markdown text-xs leading-5 text-muted-foreground">
      <AssistantMarkdown content={block.content} />
    </div>
  );
}

function TableBlock({ data, content }: { data: Record<string, unknown>; content: string }) {
  const columns = Array.isArray(data.columns) ? data.columns.map(String) : [];
  const rows = Array.isArray(data.rows) ? data.rows : [];
  if (columns.length === 0 || rows.length === 0) {
    return <AssistantMarkdown content={content} />;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column} className="border border-border bg-muted px-2 py-1 text-left">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {columns.map((column) => (
                <td key={column} className="border border-border px-2 py-1">
                  {cellValue(row, column)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChartBlock({ data, fallback }: { data: Record<string, unknown>; fallback: string }) {
  const chartType = String(data.chart_type ?? "bar");
  const chartData = Array.isArray(data.data) ? (data.data as Record<string, unknown>[]) : [];
  const xKey = String(data.x_key ?? "label");
  const yKeys = Array.isArray(data.y_keys) ? data.y_keys.map(String) : ["value"];
  if (chartData.length === 0) return <AssistantMarkdown content={fallback} />;

  return (
    <div className="h-52 w-full">
      <ResponsiveContainer width="100%" height="100%">
        {chartType === "line" ? (
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={xKey} tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip />
            {yKeys.map((key, index) => (
              <Line key={key} type="monotone" dataKey={key} stroke={COLORS[index % COLORS.length]} />
            ))}
          </LineChart>
        ) : chartType === "pie" ? (
          <PieChart>
            <Tooltip />
            <Pie data={chartData} dataKey={yKeys[0]} nameKey={xKey} outerRadius={72}>
              {chartData.map((_, index) => (
                <Cell key={index} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
          </PieChart>
        ) : (
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={xKey} tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip />
            {yKeys.map((key, index) => (
              <Bar key={key} dataKey={key} fill={COLORS[index % COLORS.length]} />
            ))}
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

function mathBlock(content: string) {
  const trimmed = content.trim();
  if (trimmed.startsWith("$$")) return trimmed;
  return `$$\n${trimmed}\n$$`;
}

function cellValue(row: unknown, column: string) {
  if (row && typeof row === "object" && column in row) {
    return String((row as Record<string, unknown>)[column] ?? "");
  }
  if (Array.isArray(row)) return String(row[0] ?? "");
  return String(row ?? "");
}

const COLORS = ["#2563eb", "#16a34a", "#f97316", "#9333ea", "#0891b2"];

function blockTypeLabel(type: string) {
  return type
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
