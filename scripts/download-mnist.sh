#!/usr/bin/env bash
# Download original MNIST IDX files. No aliases, conversion or Python packages.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/data"
cd "$root/data"
tmp=$(mktemp -d .mnist-download.XXXXXX)
trap 'rm -rf "$tmp"' EXIT

verify() {
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s  %s\n' "$2" "$1" | sha256sum --check --status
  else
    printf '%s  %s\n' "$2" "$1" | shasum -a 256 --check --status
  fi
}

while read -r name digest; do
  if [ -e "$name" ] || [ -L "$name" ]; then
    if [ -L "$name" ] || ! verify "$name" "$digest"; then
      printf 'Refusing to replace existing or invalid file: data/%s\n' "$name" >&2
      exit 1
    fi
    printf 'Verified data/%s\n' "$name"
    continue
  fi
  printf 'Downloading %s\n' "$name"
  curl --fail --location --retry 3 --connect-timeout 20 --max-time 180 \
    "https://storage.googleapis.com/cvdf-datasets/mnist/$name.gz" \
    --output "$tmp/$name.gz"
  gzip -dc "$tmp/$name.gz" > "$tmp/$name"
  if ! verify "$tmp/$name" "$digest"; then
    printf 'Checksum failed: %s\n' "$name" >&2
    exit 1
  fi
  mv "$tmp/$name" "$name"
done <<'FILES'
train-images-idx3-ubyte ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db
train-labels-idx1-ubyte 65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5
t10k-images-idx3-ubyte 0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7
t10k-labels-idx1-ubyte ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2
FILES
printf 'MNIST ready. Compile and run ./main from the repository root.\n'
