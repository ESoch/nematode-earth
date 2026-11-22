#!/bin/bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

TAGS=(bacterivores fungivores herbivores omnivores predators totalnumber_per100g bootstrap_coefvar)
URLS=(
	"https://www.research-collection.ethz.ch/server/api/core/bitstreams/6394a9df-fe27-4bef-b712-f4f3aeb2189f/content"
	"https://www.research-collection.ethz.ch/server/api/core/bitstreams/2eb513f5-dbd5-4dd8-becd-ae0699e6840a/content"
	"https://www.research-collection.ethz.ch/server/api/core/bitstreams/c4de346b-d8ae-4b02-a955-52ddb0222f4e/content"
	"https://www.research-collection.ethz.ch/server/api/core/bitstreams/ed1a9f1e-1e45-4224-8969-3236fa1b3365/content"
	"https://www.research-collection.ethz.ch/server/api/core/bitstreams/62f5d623-90a7-43dc-a87f-c84f89fb9210/content"
	"https://www.research-collection.ethz.ch/server/api/core/bitstreams/9c6f6a0d-a7cd-4261-aa1c-48924fb53374/content"
	"https://www.research-collection.ethz.ch/server/api/core/bitstreams/137ecc0c-2673-40fd-a9db-a4b4839bde33/content"
)

for i in "${!TAGS[@]}"; do
	tag="${TAGS[$i]}"
	url="${URLS[$i]}"
	echo "=== $(date '+%H:%M:%S') starting $tag ==="
	uv run fetch/eth_raster.py extract "$url" --out-width 512 --out-height 256 \
		--row-coverage 0.025 --out-dir out/ --tag "$tag"
	echo "=== $(date '+%H:%M:%S') finished $tag ==="
done
echo "=== ALL DONE ==="
