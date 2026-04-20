# Multi-stage Next.js 14 build producing a minimal standalone runtime.
# `output: "standalone"` in next.config.mjs emits a self-contained .next/standalone
# server that includes only the deps it actually uses.

# ---------- deps ----------
FROM node:20-alpine AS deps

WORKDIR /app

# Corepack-less install; package-lock.json is optional in dev repos, so fall
# back to plain `npm install` when a lock file isn't present.
COPY package.json package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

# ---------- builder ----------
FROM node:20-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# NEXT_PUBLIC_* vars are inlined at build time by Next.js, so this must be a
# build-arg — setting it as a runtime env in docker-compose has no effect.
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL} \
    NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---------- runner ----------
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Drop privileges.
RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

# `next build` with output:standalone writes .next/standalone/server.js plus the
# minimum node_modules subset. The public/ and .next/static/ dirs still need to
# be copied separately — the runtime reads them at well-known paths.
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

CMD ["node", "server.js"]
