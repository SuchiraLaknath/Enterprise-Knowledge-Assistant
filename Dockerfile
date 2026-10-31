# One image for the api, ui, mcp and ingest services, which differ only in their command and in
# the environment that Compose gives each one.
FROM python:3.12-slim

# Pinned, so that a rebuild installs the same tools as `uv.lock` was made with.
RUN pip install --no-cache-dir uv==0.11.21

WORKDIR /app

# Dependencies come first, so that a code change does not reinstall them.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# The tokenizer data is baked in, because rule 6 allows no download at run time.
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN uv run --no-sync python -m nltk.downloader -d "$NLTK_DATA" punkt_tab stopwords

COPY . .

# Containers run without root, so that a sandbox escape would gain as little as possible.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
