"use client";

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

import { CourseBuilderCitationList } from "./CitationList";

interface Props {
  block: CourseBuilderLessonBlock;
}

export function LessonBlockRenderer({ block }: Props) {
  return (
    <article className="content-selectable rounded-md border border-border bg-surface p-3">
      {block.title && (
        <h5 className="mb-2 text-xs font-semibold text-foreground">{block.title}</h5>
      )}
      <BlockBody block={block} />
      <CourseBuilderCitationList citations={block.source_citations} />
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
