import * as React from "react";

import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium",
  {
    variants: {
      variant: {
        default:
          "border-border bg-surface text-surface-foreground",
        muted:
          "border-transparent bg-muted text-muted-foreground",
        primary:
          "border-transparent bg-primary/15 text-primary",
        success:
          "border-transparent bg-[hsl(var(--success)/0.18)] text-[hsl(var(--success))]",
        warning:
          "border-transparent bg-[hsl(var(--warning)/0.18)] text-[hsl(var(--warning))]",
        danger:
          "border-transparent bg-[hsl(var(--danger)/0.18)] text-[hsl(var(--danger))]",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { badgeVariants };
