import type { Metadata, Viewport } from "next";

import { Providers } from "@/app/providers";

import "katex/dist/katex.min.css";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "TeacherLM",
    template: "%s · TeacherLM",
  },
  description:
    "Upload your course files and learn with an AI teacher grounded in what you provide.",
};

export const viewport: Viewport = {
  themeColor: "#0f0f14",
  width: "device-width",
  initialScale: 1,
};

const themeScript = `
(function () {
  try {
    var stored = window.localStorage.getItem("teacherlm-ui");
    var theme = "dark";
    if (stored) {
      var parsed = JSON.parse(stored);
      var value = parsed && parsed.state && parsed.state.theme;
      if (value === "light" || value === "dark") theme = value;
    }
    var root = document.documentElement;
    root.classList.toggle("dark", theme === "dark");
    root.style.colorScheme = theme;
  } catch (_) {
    document.documentElement.classList.add("dark");
    document.documentElement.style.colorScheme = "dark";
  }
})();
`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body
        className="min-h-dvh bg-background font-sans text-foreground antialiased"
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
