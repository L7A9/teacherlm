import type { Metadata, Viewport } from "next";

import { Providers } from "@/app/providers";

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

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className="min-h-dvh bg-background font-sans text-foreground antialiased"
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
