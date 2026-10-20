#!/usr/bin/env bash
# Encode video/demo.mp4 from the frames video/make_slides.py produced.
#
#   .venv/bin/python video/make_slides.py && bash video/make_video.sh
#
# Inputs : video/frames/sNN_*.png, video/frames/manifest.txt (name|seconds|zoom|telop),
#          video/telop/*.txt
# Output : video/demo.mp4 — 1280x720, 30 fps, H.264 High / yuv420p, silent AAC
#
# The target is a Devpost upload: 2-3 minutes, under 30 MB, mp4 with a faststart
# index.  Every frame is a picture of a file this repository produced, so the
# video cannot claim anything the measurement did not.

set -euo pipefail

cd "$(dirname "$0")/.."
FRAMES="video/frames"
MANIFEST="$FRAMES/manifest.txt"
BUILD="$FRAMES/build"
OUT="video/demo.mp4"

FONT_TELOP="/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc"
FPS=30
CRF=22

command -v ffmpeg  >/dev/null || { echo "ffmpeg not found" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe not found" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "$MANIFEST missing — run: .venv/bin/python video/make_slides.py" >&2; exit 1; }
[[ -f "$FONT_TELOP" ]] || { echo "telop font $FONT_TELOP missing (install noto-fonts-cjk)" >&2; exit 1; }

mkdir -p "$BUILD"
rm -f "$BUILD"/*.mp4 "$BUILD/list.txt"

echo "== encoding segments"
: > "$BUILD/list.txt"
total=0
while IFS='|' read -r name dur zoom telop; do
  [[ -z "${name:-}" ]] && continue
  frame="$FRAMES/$name.png"
  [[ -f "$frame" ]] || { echo "missing frame $frame" >&2; exit 1; }
  [[ -f "$telop" ]] || { echo "missing telop $telop" >&2; exit 1; }

  fade_out="$(python3 -c "print(round($dur-0.6,3))")"

  if [[ "$zoom" == "1" ]]; then
    # Slow push-in on screen frames: motion without cutting away from the text.
    rate="$(python3 -c "print(f'{0.06/($dur*$FPS):.8f}')")"
    video="[0:v]scale=2560:1440,zoompan=z='min(1+${rate}*on,1.06)'\
:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1280x720:fps=$FPS,setsar=1"
  else
    video="[0:v]scale=1280:720,setsar=1,fps=$FPS"
  fi

  filter="$video,\
fade=t=in:st=0:d=0.45,fade=t=out:st=$fade_out:d=0.6,\
drawbox=x=0:y=594:w=1280:h=2:color=0x2b3542@0.9:t=fill,\
drawtext=fontfile=$FONT_TELOP:textfile=$telop:expansion=none:fontcolor=0xf2f5f9:fontsize=27\
:x=(w-text_w)/2:y=630:line_spacing=8:box=1:boxcolor=0x000000@0.55:boxborderw=14[vo]"

  # -nostdin: ffmpeg would otherwise swallow the manifest being read by this loop
  ffmpeg -nostdin -y -hide_banner -loglevel error \
    -framerate "$FPS" -loop 1 -t "$dur" -i "$frame" \
    -f lavfi -t "$dur" -i "anullsrc=r=48000:cl=mono" \
    -filter_complex "$filter" \
    -map "[vo]" -map 1:a \
    -c:v libx264 -preset medium -crf "$CRF" -pix_fmt yuv420p -profile:v high -g 60 \
    -c:a aac -b:a 96k -ar 48000 -shortest \
    "$BUILD/$name.mp4"

  printf "file '%s'\n" "$name.mp4" >> "$BUILD/list.txt"
  total=$(( total + dur ))
  printf "  %-16s %3ss  zoom=%s\n" "$name" "$dur" "$zoom"
done < "$MANIFEST"

echo "== concatenating (stream copy)"
ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i "$BUILD/list.txt" \
  -c copy -movflags +faststart "$OUT"

echo "== result"
ffprobe -v error -show_entries format=duration,size,format_name \
  -show_entries stream=codec_name,profile,width,height,r_frame_rate,pix_fmt \
  -of default=nw=1 "$OUT"
printf "planned: %ss (%dm%02ds)\n" "$total" $(( total / 60 )) $(( total % 60 ))
