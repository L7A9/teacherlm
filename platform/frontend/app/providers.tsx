"use client";

import { useEffect, useMemo, useState } from "react";

import { CssBaseline, ThemeProvider, createTheme } from "@mui/material";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";

import { useUiStore } from "@/stores/uiStore";

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 30_000,
            refetchOnWindowFocus: false,
            retry: 1,
          },
        },
      }),
  );

  const theme = useUiStore((s) => s.theme);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  const muiTheme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: theme,
          background: {
            default: "hsl(var(--background))",
            paper: "hsl(var(--surface))",
          },
          text: {
            primary: "hsl(var(--foreground))",
            secondary: "hsl(var(--muted-foreground))",
          },
          primary: {
            main: "hsl(var(--primary))",
            contrastText: "hsl(var(--primary-foreground))",
          },
          secondary: {
            main: "hsl(var(--accent))",
            contrastText: "hsl(var(--accent-foreground))",
          },
          divider: "hsl(var(--border))",
          error: {
            main: "hsl(var(--danger))",
          },
          warning: {
            main: "hsl(var(--warning))",
          },
          success: {
            main: "hsl(var(--success))",
          },
        },
        shape: {
          borderRadius: 8,
        },
        typography: {
          fontFamily: "var(--font-sans)",
          button: {
            textTransform: "none",
            fontWeight: 600,
          },
        },
        components: {
          MuiCssBaseline: {
            styleOverrides: {
              body: {
                backgroundColor: "hsl(var(--background))",
                color: "hsl(var(--foreground))",
              },
            },
          },
          MuiPaper: {
            styleOverrides: {
              root: {
                backgroundImage: "none",
              },
            },
          },
        },
      }),
    [theme],
  );

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={muiTheme}>
        <CssBaseline />
        {children}
        <Toaster
          theme={theme}
          position="bottom-right"
          richColors
          closeButton
        />
      </ThemeProvider>
    </QueryClientProvider>
  );
}
