# Build notes: Volume launch film (v3)

v3 is a new cut on the v2 pipeline: a locked 15-line script, new footage (DVIDS, NASA,
Pexels), a new narrator (Microsoft neural TTS), and a new music edit (Scott Buckley,
"Uprising"). The film is 63.0 s. v1 (`logistics/`, root `cuesheet.json`) is untouched; v2's
cut is in git history.

## Story

The locked script, one narrator, every line cut against the picture it names:

| at | line | picture |
|---|---|---|
| 0:00 | Everyone's talking about AI. | glowing fibre in a supercomputer; server boards |
| 0:03 | Nobody's talking about how it gets built. | steel frame going up (aerial); concrete pour |
| 0:06 | Transformers. Switchgear. Chillers. Racks. | one shot per noun, cut on each word |
| 0:10 | Built on three continents. Needed on one site. | steel mill; ship on open ocean; one mega site from above |
| 0:14 | One missing part, and a gigawatt sits dark. | empty hall; power lines at dusk |
| 0:18 | And AI is just the start. | industry at sunrise, music lifts |
| 0:20 | Chip fabs. Power plants. Drone factories. Drug plants. Shipyards. | one shot per noun |
| 0:25 | The biggest build in a generation. | forest of tower cranes; foundations from above |
| 0:28 | It goes to whoever builds fastest. | robotic welding; structure rising (time-lapse) |
| 0:31 | And nothing gets built until it arrives. | heavy haul on a mountain road; crane sets the unit |
| 0:34 | Logistics used to be the back office. | desk, typewriter, paperwork |
| 0:37 | Now it's the front line. | ship-to-shore crane lift; container port at night |
| 0:40 | Air. Ocean. Ground. Customs. | C-5M lift-off, ship at sea, semi at dusk, inspection truck |
| 0:43 | Built for the build. | container hoisted against the sky |
| 0:46 | (music breath) | port at sunset |
| 0:48 | *They build the future.* (type only) | grinder sparks; ironworkers at height |
| 0:51 | *We deliver it.* (type only) | ship heading out to sea |
| 0:53 | Volume. | logo on the downbeat |

End card (0:56): "Built for the build.", `volumeba.com`, and the DoD visual-information
non-endorsement notice. Everything is in `cuesheet.json` (`narration.lines`, `titles`,
`reveal`, `endcard`).

## Pipeline

```
sources.yaml ──fetch.py──> cache/sources/             (git-ignored downloads)
cuesheet.json ─┬─narrate.py─> cache/work/narration/    (one WAV per narration line)
               ├─audio.py───> cache/work/mix.wav       (music + narration, mastered)
               └─picture.py─> out/*.mp4                (cut, grade, grain, blur, type, logo; muxes the mix)
credits.py ──> CREDITS.csv / CREDITS.md                (generated from the manifest + actual uses)
contact.py ──> out/contact_sheet_{16x9,9x16}.png
scout.py      footage search / preview helpers (DVIDS, NASA, Commons, Pexels)
voice_eval.py narrator screening (engine/voice candidates scored on the script)
```

- **One tempo map.** 90 BPM in 4/4 (1 bar = 2.667 s, 1 beat = 0.667 s), the track's tempo.
  Shots, titles, narration and the reveal are placed in seconds (`at_s`); music segments in
  bars. List lines ("Transformers. Switchgear. ...") are one read, split at the voice's own
  pauses (`split_at_s`) so each noun lands on its cut.
- **Music edit.** Three whole-phrase segments of "Uprising", joined on downbeats:
  track 0:25.7-0:44.3 (7 bars, opening build), 1:32.3-2:12.3 (15 bars: the lift on "And AI is
  just the start", the break under the closing lines, the hit on the logo at 0:53.3), and
  3:14.0-3:18.4 (the final chord ringing out). No ducking or sweeps; the music bed is
  statically EQ-carved (-3 dB at 2.6 kHz, -1.5 dB at 500 Hz) so the voice sits in a fixed gap.
- **Narration.** `narrate.py` + `tts_edge.py`: Microsoft's `en-US-GuyNeural` voice via
  edge-tts. Each line gets four takes (the whole script read as one passage at two speaking
  rates, cut out at the service's word boundaries, plus the line read alone at two rates).
  The kept take must transcribe back to exactly the script (faster-whisper `medium.en`) and
  scores best on UTMOS among takes that pass. All 15 kept takes match the script; UTMOS
  4.24-4.53 (mean 4.44). Two fixes came out of checking the final mix, not the takes:
  the line trim now finds the onset at -38 dB re the loudest 10 ms (it was -26 dB, which
  shaved the voiced B off "Built for the build" so the mix read "Dilt"), and v07 has a
  `tts_text` ("Chip fabs; power plants; ...", same words, semicolons) because Guy's
  period-separated read gives a soft, Sh-like "Ch" that the recognizer hears as "Ship fabs"
  next to "shipyards". The semicolon read passes at every rate, alone and over the music. The engine runs in the Python named by `NARRATOR_PYTHON` (it needs
  torch for the scoring). A recorded human read replaces any line with no code change: drop
  `narration_recorded/<line id>.wav` and re-run `--audio`.
- **Voice choice.** `voice_eval.py` read the script with eight candidates: Edge Andrew,
  Andrew Multilingual, Brian, Christopher and Guy, Kokoro `am_michael`, and Orpheus `leo`
  and `dan`. Guy had the highest mean and minimum UTMOS (4.38 / 4.21) with natural pitch
  movement (4.7 semitone F0 s.d.); Andrew and Orpheus had more movement but dropped words
  or scored lower.
- **Picture.** Each shot is decoded by ffmpeg from the original source (accurate seek, crop to
  the output aspect, Lanczos scale, every source frame shown once so pans don't judder).
  Per-shot normalization (partial gray-world white balance, exposure match) makes DVIDS,
  NASA and stock sit together, then one warm film grade for everything: soft highlight
  shoulder, gentle S-curve, amber highlights / cool shadows, halation, printed blacks,
  vignette and fine grain. Optional per-shot push-in or pan. The 9:16 cut uses its own
  per-shot crop (`c9`, and `pan9` where a subject moves across frame), not a pillarbox.
- **Blur.** Readable brand marks are blurred in the source frame (`blur` in the shot:
  source-fraction boxes, optionally moving over source seconds): CMA CGM logo and owner code
  on the crane-lift container door, seven OOCL / Textainer marks on the stacked containers
  behind the hoist, the Yaskawa label on the welding robot, the Trane panel in the chiller
  hall, the Ford badge on the inspection truck, and the embroidered lettering on the
  grinder's collar. Boxes were measured on gridded source frames; the hoist shot's camera
  drift was measured by phase correlation on the static stacks.
- **Type.** Inter only. The two closing lines fade in over footage; no subtitles.
- **Logo and end card.** `logo.py` rasterizes `brand/volume-logo.svg` as-is (cream letters,
  gold square). The square lands on the 0:53.3 downbeat and the narrator says the name;
  the last shot dims to black, then the logo rises and the end line, URL and DoD notice
  fade up (`endcard.py`).
- **Master.** Narration chain: high-pass, chest and presence EQ, de-esser, compressor, one
  level for every line. Mix mastered by iterative gain to -14 LUFS integrated with a
  4x-oversampled true-peak limiter at -1.3 dBFS (measured with ffmpeg `ebur128`).

## Commands

```bash
sudo apt-get install -y ffmpeg fonts-inter
pip install --break-system-packages numpy scipy opencv-python-headless pillow pyyaml soundfile

# narration engine + scoring in their own environment
python3 -m venv ~/ttsvenv
~/ttsvenv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
~/ttsvenv/bin/pip install edge-tts faster-whisper soundfile librosa
export NARRATOR_PYTHON=~/ttsvenv/bin/python

python3 -m launchfilm.render --all --artifacts /opt/cursor/artifacts   # everything
python3 -m launchfilm.render --fetch                                   # downloads only
python3 -m launchfilm.render --narration                               # voice any new/changed lines
python3 -m launchfilm.render --audio                                   # narration + re-mix
python3 -m launchfilm.render --format 16x9                             # one cut (reuses the mix)
python3 -m launchfilm.render --stills 22.5,52.6 --format 9x16          # PNG stills -> out/stills/
python3 -m launchfilm.render --credits                                 # regenerate CREDITS.*
```

The end card URL is the single field `endcard.url` in `cuesheet.json`.

## Source and rights decisions

Every clip is a U.S. federal government work (DVIDS, NASA) or a Pexels video under the
Pexels License (free commercial use, no attribution required; credited anyway). The music is
CC BY 4.0 and its credit line is required wherever the film is published (for YouTube, in
the description): *'Uprising' by Scott Buckley - released under CC-BY 4.0.
www.scottbuckley.com.au*. The composer's licensing page says CC BY does not cover
placements that can't carry the credit (for example ads with no description); those need
his paid Web/Standard license. Each item's page was checked on 2026-09-28; quoted statements are in
`sources.yaml` and `CREDITS.*`. DVIDS creators are credited as units (license lines quoted
with the videographer's name replaced by "..."). The DoD notice is on the end card because
the film includes DoD visual information.

Pexels pages sit behind a bot challenge from the build VM, so titles come from each page's
URL slug and creator names from the page text via a reader service; the video files
themselves download directly from `videos.pexels.com`.

### Rejected or adjusted

| Item | Reason |
|---|---|
| DVIDS 919537 in-point 0:14 | AIM Photonics logo on screen. Moved to 0:46 (cleanroom, unbranded). |
| DVIDS 993151 in-point 0:03 | Worker portrait as the subject. Moved to 4:52 (shipyard at golden hour). |
| DVIDS 712740 opening | Title splash. Moved to 0:22 (robotic arc welding). |
| DVIDS 891556 hoist at 1:02-1:04 | Whip pan; unusable. The hoist at 3:25 is used with its stack logos blurred. |
| Stacked OOCL / Textainer containers (891556 at 3:25) | Blurred rather than cropped: cropping would lose the hoist against the sky. |
| DVIDS 787947 switchgear stickers | Checked at full resolution; not legible. No blur. |
| Hard-hat marks on the grinder (993144) | Hand-written scrawl, not a brand. No blur. |
| Pixabay | Still HTTP 403 from the VM. Not used. |

## Checks run on the delivered files

`ffprobe` (codec, size, fps, duration, audio), `ebur128` loudness and true peak, a
frame-by-frame black-frame scan, speech recognition (`medium.en`) over the audio of both
delivered MP4s against the locked script, and review of contact sheets and stills of every
shot in both formats, including full-resolution crops of every blurred region.

| check | 16:9 | 9:16 |
|---|---|---|
| duration | 63.000 s | 63.000 s |
| size | 66.0 MB | 66.4 MB |
| video / audio | H.264 High 1920x1080 24 fps / AAC 48 kHz | H.264 High 1080x1920 24 fps / AAC 48 kHz |
| integrated loudness | -14.1 LUFS | -14.1 LUFS |
| true peak | -1.1 dBTP | -1.1 dBTP |
| dark frames | 56.6-63.0 s only (the end card: charcoal with type) | same |
| speech recognition vs script | exact match | exact match |

The recognizer also emits a one-word "You" at 0:59.6, over the end-card music; the voice
stem is digital silence after 0:55, so that is a recognizer hallucination, not speech.
