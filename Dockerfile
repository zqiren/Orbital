# syntax=docker/dockerfile:1
# Orbital daemon for a Linux server. See docs/deploy-linux-server.md, which also
# covers the access model.

FROM node:22-bookworm-slim AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
COPY web/vendor ./vendor
RUN npm ci --no-audit --no-fund
COPY web ./
RUN npm run build

FROM node:22-bookworm-slim AS node

FROM python:3.12-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl wget git openssh-client less jq procps \
      unzip xz-utils file tzdata tini util-linux iptables ripgrep \
      xvfb x11vnc openbox novnc websockify supervisor x11-utils \
      fonts-noto-cjk fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && npm install -g --no-audit --no-fund \
         @anthropic-ai/claude-code @openai/codex @google/gemini-cli \
    && npm cache clean --force

# Dependencies and Chromium first: changing the source must not re-download them.
WORKDIR /app
COPY pyproject.toml README.md ./
RUN mkdir agent_os && touch agent_os/__init__.py \
    && pip install -e ".[browser,office]" keyrings.alt tzlocal uv \
    && python -m patchright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/* \
    && chmod -R a+rX /opt/ms-playwright
COPY agent_os ./agent_os
COPY --from=web /src/web/dist /app/web/dist

COPY docker/entrypoint.sh /usr/local/bin/orbital-entrypoint
COPY docker/open-browser.sh /usr/local/bin/orbital-browser
COPY docker/supervisord.conf /etc/orbital/supervisord.conf
RUN chmod 0755 /usr/local/bin/orbital-entrypoint /usr/local/bin/orbital-browser \
    && useradd -u 10001 -m -s /bin/bash orbital

ENV HOME=/home/orbital \
    USER=orbital \
    DISPLAY=:99 \
    BROWSER=/usr/local/bin/orbital-browser \
    AGENT_OS_SPA_DIR=/app/web/dist \
    AGENT_OS_DATA_DIR=/home/orbital/.orbital \
    AGENT_OS_UPDATE_CHECK=0 \
    PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring \
    NPM_CONFIG_PREFIX=/home/orbital/.npm-global \
    PATH=/home/orbital/.npm-global/bin:/home/orbital/.local/bin:/usr/local/bin:/usr/bin:/bin

WORKDIR /home/orbital
EXPOSE 8000 6080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS -o /dev/null http://127.0.0.1:8000/api/v2/settings || exit 1
ENTRYPOINT ["/usr/local/bin/orbital-entrypoint"]
