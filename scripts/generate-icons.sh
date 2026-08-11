#!/usr/bin/env sh
set -eu
src="tool/web/static/capyy.png"
out_dir="tool/web/static"
render() { size="$1"; output="$out_dir/capyy-${size}.png"; if command -v magick >/dev/null 2>&1; then magick "$src" -resize "${size}x${size}" "$output"; elif command -v convert >/dev/null 2>&1; then convert "$src" -resize "${size}x${size}" "$output"; elif command -v sips >/dev/null 2>&1; then sips -Z "$size" "$src" --out "$output" >/dev/null; else echo "Need ImageMagick or macOS sips." >&2; exit 1; fi; }
render 32
render 180
render 512
cp "$out_dir/capyy-32.png" "$out_dir/favicon.png"
echo "Generated Capyy PNG icons in $out_dir"
