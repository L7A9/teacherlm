import { useQuery } from "@tanstack/react-query";

import { generatorsApi } from "@/lib/api";
import type { GeneratorListResponse, GeneratorView } from "@/lib/types";

const ROOT_KEY = ["generators"] as const;

export function useGenerators(includeDisabled = false) {
  return useQuery<GeneratorListResponse>({
    queryKey: [...ROOT_KEY, "list", includeDisabled],
    queryFn: () => generatorsApi.list(includeDisabled),
    staleTime: 60_000,
  });
}

export function useGenerator(id: string | null | undefined) {
  return useQuery<GeneratorView>({
    queryKey: [...ROOT_KEY, "detail", id],
    queryFn: () => generatorsApi.get(id as string),
    enabled: Boolean(id),
    staleTime: 60_000,
  });
}
