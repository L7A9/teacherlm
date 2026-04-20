"use client";

import { useState } from "react";

import Link from "next/link";
import { useRouter } from "next/navigation";

import {
  GraduationCap,
  Loader2,
  MessageSquare,
  Plus,
  Sparkles,
  Trash2,
} from "lucide-react";
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
import {
  useConversations,
  useCreateConversation,
  useDeleteConversation,
} from "@/hooks/useConversations";
import type { Conversation } from "@/lib/types";
import { cn, formatRelativeTime } from "@/lib/utils";

export default function HomePage() {
  const { data, isLoading, error } = useConversations();
  const create = useCreateConversation();
  const router = useRouter();

  const handleNew = async () => {
    try {
      const created = await create.mutateAsync({});
      router.push(`/c/${created.id}`);
    } catch (err) {
      toast.error(`Could not create conversation: ${(err as Error).message}`);
    }
  };

  return (
    <main className="min-h-dvh bg-background text-foreground">
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-6 py-5">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary/15 text-primary">
              <GraduationCap className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-lg font-semibold">TeacherLM</h1>
              <p className="text-xs text-muted-foreground">
                Your AI teacher, grounded in the files you upload.
              </p>
            </div>
          </div>

          <Button
            variant="primary"
            onClick={handleNew}
            disabled={create.isPending}
          >
            {create.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Plus className="h-4 w-4" />
            )}
            New conversation
          </Button>
        </div>
      </header>

      <div className="mx-auto max-w-4xl px-6 py-8">
        {isLoading ? (
          <LoadingState />
        ) : error ? (
          <ErrorState message={(error as Error).message} />
        ) : (data?.items.length ?? 0) === 0 ? (
          <EmptyState onCreate={handleNew} pending={create.isPending} />
        ) : (
          <ConversationGrid items={data?.items ?? []} />
        )}
      </div>
    </main>
  );
}

function LoadingState() {
  return (
    <div className="flex items-center justify-center gap-2 py-16 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" />
      Loading conversations…
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-[hsl(var(--danger)/0.4)] bg-[hsl(var(--danger)/0.08)] px-4 py-6 text-sm text-[hsl(var(--danger))]">
      Couldn't load conversations: {message}
    </div>
  );
}

function EmptyState({
  onCreate,
  pending,
}: {
  onCreate: () => void;
  pending: boolean;
}) {
  return (
    <div className="flex flex-col items-center gap-4 rounded-lg border border-dashed border-border bg-surface px-6 py-16 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/15 text-primary">
        <Sparkles className="h-5 w-5" />
      </div>
      <div className="flex flex-col gap-1">
        <h2 className="text-base font-semibold">Start your first conversation</h2>
        <p className="max-w-md text-sm text-muted-foreground">
          Create a conversation, upload your course files, and chat with a
          teacher that stays grounded in what you gave it.
        </p>
      </div>
      <Button variant="primary" onClick={onCreate} disabled={pending}>
        {pending ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Plus className="h-4 w-4" />
        )}
        New conversation
      </Button>
    </div>
  );
}

function ConversationGrid({ items }: { items: Conversation[] }) {
  return (
    <ul className="grid gap-3 md:grid-cols-2">
      {items.map((c) => (
        <ConversationCard key={c.id} conversation={c} />
      ))}
    </ul>
  );
}

function ConversationCard({ conversation }: { conversation: Conversation }) {
  const [confirming, setConfirming] = useState(false);
  const del = useDeleteConversation();

  const handleDelete = async () => {
    try {
      await del.mutateAsync(conversation.id);
      setConfirming(false);
    } catch (err) {
      toast.error(`Delete failed: ${(err as Error).message}`);
    }
  };

  return (
    <li
      className={cn(
        "group relative rounded-lg border border-border bg-surface transition-colors",
        "hover:border-primary/50",
      )}
    >
      <Link
        href={`/c/${conversation.id}`}
        className="flex flex-col gap-2 px-4 py-4"
      >
        <div className="flex items-start gap-2">
          <MessageSquare className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-medium">
              {conversation.title}
            </div>
            <div className="text-[11px] text-muted-foreground">
              Updated {formatRelativeTime(conversation.updated_at)}
            </div>
          </div>
        </div>
      </Link>

      <button
        type="button"
        aria-label="Delete conversation"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setConfirming(true);
        }}
        className={cn(
          "absolute right-2 top-2 rounded-md p-1.5 text-muted-foreground opacity-0 transition",
          "group-hover:opacity-100 hover:bg-muted hover:text-[hsl(var(--danger))]",
          "focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        )}
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>

      <Dialog
        open={confirming}
        onOpenChange={(v) => !v && setConfirming(false)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete conversation?</DialogTitle>
            <DialogDescription>
              "{conversation.title}" and all of its files, messages, and
              generated items will be removed. This can't be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="secondary"
              onClick={() => setConfirming(false)}
              disabled={del.isPending}
            >
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={handleDelete}
              disabled={del.isPending}
            >
              {del.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Trash2 className="h-4 w-4" />
              )}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </li>
  );
}
