# Builds @catan/server. Run `docker build .` from the repo root (not
# packages/server) — the image needs the whole pnpm workspace, since
# @catan/server depends on @catan/core and @catan/ai as workspace packages
# that are consumed directly as TypeScript source (see each package's
# package.json "main"), not as prebuilt dist artifacts.
#
# Debian-based (not Alpine): onnxruntime-node only ships prebuilt native
# bindings for glibc, not musl.
FROM node:24-bookworm-slim

WORKDIR /app

# onnxruntime-node's postinstall needs to run; allowed explicitly in
# pnpm-workspace.yaml's allowBuilds (pnpm refuses unknown postinstall
# scripts by default since v10).
RUN corepack enable

COPY . .

# Server-side AI seats load .onnx model files from packages/web/public/models
# by default (packages/server/src/bots.ts) — that directory is gitignored
# (see .gitignore's `models/`), so it is NOT fetched from git. It must exist
# on this machine's checkout before `docker build` runs, since COPY above
# only sees what .dockerignore lets through, not what git tracks. If your
# models live elsewhere, skip this and set MODEL_DIR at `docker run` time.
RUN pnpm install --frozen-lockfile

ENV PORT=8080
EXPOSE 8080

CMD ["pnpm", "--filter", "@catan/server", "start"]
