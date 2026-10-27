#!/usr/bin/env sh
# Fetch the raw data into data/raw (runs in the dev container via ./run download).
set -eu
mkdir -p data/raw
cd data/raw

base=https://downloads.tatoeba.org/exports/per_language
curl -fsSLO "$base/spa/spa_sentences.tsv.bz2"
curl -fsSLO "$base/fra/fra_sentences.tsv.bz2"

# Links file (which Spanish sentence translates which French one). The name below follows
# Tatoeba's per-language pattern; if it fails, copy the exact URL from
# https://tatoeba.org/downloads into TATOEBA_LINKS_URL in .env.
links_url="${TATOEBA_LINKS_URL:-$base/spa/spa-fra_links.tsv.bz2}"
curl -fsSL -o spa-fra_links.tsv.bz2 "$links_url" \
  || echo "WARNING: links file not found at $links_url (set TATOEBA_LINKS_URL in .env)"

curl -fsSL -o cocoa.zip https://codeload.github.com/amazon-science/contrastive-controlled-mt/zip/refs/heads/main
unzip -qo cocoa.zip

echo "data/raw now contains:"
ls -lh
