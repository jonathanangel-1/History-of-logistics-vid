# Build notes: Volume launch film (v2)

v2 replaces v1's procedural picture and synthesized score with real, license-safe
footage, one narrator and a free commercially licensed music track. v1 (`logistics/`,
root `cuesheet.json`) is untouched.

## Story (v2.1)

One narrator carries the whole film, and every line is cut against the picture it names.
The history (a harbor being built, men hauling cargo, rails, steam, the box, wings, the
Saturn V stage leaving its factory) sets up the thesis: the world is building again (data
centers, microchips, robots, power, defense), it is the biggest build of our lifetime, and
every piece of it has to move, so logistics has to step up. The closing lines ("Software ate
the world." / "Somebody has to ship it.") land on black, then the gold square lands on the
track's outro downbeat and the narrator says the name. The script lives in
`cuesheet.json` (`narration.lines`); there are no subtitles, only the two closing lines as
type.

## Pipeline

```
sources.yaml ──fetch.py──> cache/sources/             (git-ignored downloads)
cuesheet.json ─┬─narrate.py─> cache/work/narration/    (one WAV per narration line)
               ├─audio.py───> cache/work/mix.wav       (music + narration + SFX, mastered)
               └─picture.py─> out/*.mp4                (cut, grade, grain, type, logo; muxes the mix)
credits.py ──> CREDITS.csv / CREDITS.md                (generated from the manifest + actual uses)
contact.py ──> out/contact_sheet_{16x9,9x16}.png
```

- **One tempo map.** `cuesheet.json` is 128 BPM in 4/4 (1 bar = 1.875 s), matching the
  music track's measured tempo (128.00 BPM, first beat at 0.035 s). Shots, titles and music
  segments are placed in bars; narration lines and SFX in seconds (`at_s` = onset of the
  first word) so words land on beats. Moving a number in the sheet moves picture and sound
  together.
- **Picture.** Each shot is decoded by ffmpeg straight from the source (accurate seek, square
  pixels for anamorphic files, crop to the output aspect inside the source's `active` area,
  Lanczos scale, conform to 24 fps), then graded in numpy: warm B&W for the archival section,
  desaturated color for the box, wings and Apollo beats, full color from "Now, we're building
  again", a single light grain/vignette pass over everything. Optional per-shot slow push-in,
  3-frame white flash on the big downbeats, and a 2-frame chromatic glitch on two cuts in the
  final burst. The 9:16 cut uses its own per-shot crop (`c9`), not a pillarbox.
- **Narration.** `narrate.py` + `tts_chatterbox.py` run a voice-over "session" with
  Chatterbox (Resemble AI, MIT). Every line is read in several seeded takes (4 per line,
  8 for "Volume."), and the kept take must transcribe back to exactly the script
  (faster-whisper), sit in the narrator's pitch range (85-125 Hz median F0, pYIN), and
  score best on UTMOS (an automatic naturalness predictor) among the takes that pass. The
  five single words ("Data centers. Microchips. Robots. Power. Defense.") are one read,
  split at its pauses and placed on their beats, so they keep a spoken cadence. The
  narrator's timbre comes from a reference built in the same session: Chatterbox's own
  stock voice reading a neutral paragraph, lowered 2.9 semitones with formants shifted.
  It is a synthetic voice, not a recording or clone of any person, and every generated
  file carries Resemble AI's imperceptible Perth watermark. The session is reproducible
  (fixed seeds) and cached per line. Chatterbox pins its own torch/numpy, so it runs in
  a separate environment named by `CHATTERBOX_PYTHON` (setup below). **A recorded human
  read replaces it without code changes:** put `narration_recorded/<line id>.wav` (for
  example `n07.wav`) in the repo root and re-run `--audio`. v2.1's Kokoro voice was
  dropped after client review ("sounds like AI").
- **Type.** Inter and Manrope (the site's faces). Only the two closing lines appear as
  type over footage/black, revealed word by word at the narrator's pace.
- **Closing sequence (`endcard.py`).** Built from the volumeba.com design language
  rather than a centered card: after "Volume." the logo glides to the top-left while the
  frame closes into the site's rounded photographic stage (22 px radius, charcoal
  surround) over a slowed dusk-at-sea shot. Then, on the site's strong ease-out: the mono
  eyebrow (INTERNATIONAL FREIGHT FORWARDING · SINCE 2000), the large Manrope headline
  "Global freight. / Clear visibility." with a gold-to-cream gradient and masked line
  reveals, the service line, a gold route that draws through the site's three gateway
  facts (Ben Gurion Airport; Haifa & Ashdod ports; in-house brokerage at the ports) with
  the gold square riding its head, the coverage line (Israel · U.S. · worldwide) and a
  smoked-glass URL button with the site's arrow tile. Every string is a live-site claim
  and lives in `endcard` in the cue sheet; the URL is still one field. The 9:16 version
  stacks the same elements inside the 250 px safe zones with a vertical route.
- **Logo.** `logo.py` rasterizes `brand/volume-logo.svg` itself (the file only uses absolute
  M/H/V/L/Q/Z paths, a rounded rect and translate/scale transforms). Letterforms render cream
  `#eee9de`, the square stays gold `#baa88a`. Nothing is redrawn. On the final hit the gold
  square lands first, then the letters resolve outward from it.
- **Sound.** The music plays as written, at one constant level: no ducking, no filter sweeps,
  no drop-outs. The film opens on a phrase start (track bar 48, after the track's own
  one-beat breath) and runs through the quiet section and the build into the big section
  (bar 64 lands at 0:30.0). There is one edit: the big section repeats in 8-bar blocks, and
  the film skips one repeat (end of bar 65 straight into bar 74; bar-to-bar chroma and band
  similarity 0.977, so the join is the same music). The track's own quieter two bars
  (82-83) are the breath under the closing lines, and its outro downbeat (bar 84) is the logo
  hit at 0:52.5; the outro plays out to 1:00 with a short tail fade. The music is statically
  EQ-carved (-3.5 dB at 2.5 kHz, -1.5 dB at 450 Hz) so the voice sits in a gap that never
  moves. Narration chain: high-pass, chest and presence EQ, de-esser, compressor, one level
  for every line. Master: iterative gain to -14 LUFS integrated with a 4x-oversampled
  true-peak limiter at -1.3 dBFS (measured with ffmpeg `ebur128`).

## Commands

```bash
sudo apt-get install -y ffmpeg fonts-inter        # or drop Inter + JetBrains Mono TTFs into ./fonts/
pip install --break-system-packages numpy scipy opencv-python-headless pillow pyyaml soundfile

# narration engine in its own environment (it pins torch 2.6 / numpy < 2)
python3 -m venv ~/cbvenv
~/cbvenv/bin/pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
~/cbvenv/bin/pip install chatterbox-tts faster-whisper soundfile
export CHATTERBOX_PYTHON=~/cbvenv/bin/python

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

Every clip in the film is a U.S. federal government work (Library of Congress early films
with no known restrictions, NASA, DVIDS), a CC0 recording, or the one CC BY 4.0 music track.
Each item's own page was checked on 2026-09-28; the quoted statements are in `sources.yaml`
and `CREDITS.*`. Creators are credited as institutions / units (DVIDS license lines are quoted
with the videographer's name replaced by "..."); the only personal name in the credits is the
composer, whose credit line is required by the CC BY 4.0 license. CC0 contributors are
credited as "Wikimedia Commons contributor" with the file URL (CC0 requires no attribution).
The narration engine (Chatterbox, MIT) is credited as a source, and every narration line
has its own row marked "synthetic voice". Manrope is under the SIL Open Font License.

v2.0 used public-domain soundbites (President Kennedy at Rice University, 1962; Apollo 11
Launch Control, 1969) as the voice track. Client review: one old speech reused across the
film did not match the pictures, captions were not being read, and the ducking under each
soundbite pulled the music down. v2.1 replaces them with the single narrator above; neither
recording is in the film any more.

### Rejected or unavailable

| Item | Reason |
|---|---|
| NASA Artemis II core stage drone / aerial-at-water clips (MAF, 2024-07-16) | Public domain, but the NASA item IDs (and so the credit URLs) contain photographers' personal names. Replaced with the Michoud 2022 resource reel. |
| NASA RadPC, LCRD cleanroom, Mars 2020 "pit crew" / "twin" reels, VIPER time-lapse | Non-NASA co-credit (RadPC), presenter or interviews on camera, burned-in lower thirds, or extreme fish-eye. Not used. |
| DVIDS Robotics at Robins, Carderock welding lab, DLA Distribution visit, data system administrators, radiation-hardened electronics, UAV blood delivery | Interview-driven, faces as the subject, a large robot-maker logo, or SD only. Not used. |
| DVIDS microgrid items (Fort McCoy, Fort Cavazos, promo series) | Ceremonies, interviews, or a utility's logos. Not used. |
| DVIDS roll-on/roll-off arrival (881375) | Commercial ship name fills the frame; armored vehicles. Not used. |
| Pexels / Pixabay (again, for AI/data-center/robotics stock) | Still HTTP 403 from the VM. Modern-build shots come from NASA and DVIDS instead. |
| Pixabay Music, Pixabay video, Pexels | pixabay.com and pexels.com return HTTP 403 (bot wall) from the build VM. Not bypassed. Music comes from incompetech (CC BY 4.0); modern footage from DVIDS/NASA. |
| Freesound SFX | Downloads require a login. Not used; SFX are CC0 files from Wikimedia Commons or natural sound from the PD footage. |
| loc.gov HTML item pages | Behind a browser challenge; not bypassed. Rights text taken from the LOC JSON API for the same items. |
| Commons CC BY / CC BY-SA "own work" port videos (Malmö, Malta Freeport, Rauma, etc.) | Attribution would put private individuals' names in the credits; many Commons CC-BY videos are also YouTube-sourced. Avoided. |
| DVIDS "Night Port Operations" (893855) | Armored vehicles / weapons. Not used. |
| DVIDS "Rapid C-17 Globemaster III takeoff" (890820) | Mostly identifiable personnel close-ups. Not used. |
| DVIDS CBP inspection shots with officers, patches, "US CUSTOMS" signage | Identifiable people and possible implied government affiliation. Not used. |
| DVIDS C-17 B-roll at 0:04 | Legible "U.S. AIR FORCE" / unit markings (DoW terms restrict insignia in commerce). Used the unmarked engine/wing angle instead; C-5 nose cropped away from its lettering. |
| Port of Savannah hoist at 0:59 | Carrier logo (CMA CGM) is the subject of the shot. Replaced with an unbranded container lift. |
| Long Beach package 2 picture | Carrier-branded containers dominate the frame. Only its natural sound (a container set on a chassis: the clang) is used. |
| Savannah rail shots | Crane-maker branding across the top of frame. Cropped out (`c16`/`c9` zoomed to the lower frame); the 0:27 in-point (readable carrier name on a container) dropped entirely. |
| Long Beach package 1 cranes and yard trucks (0:49, 2:57, 3:53) and package 3 yard (0:34) | Readable carrier / crane-maker / fleet names on crane booms, trucks, or a hoisted container. Replaced with the bridge vista, gate truck cropped to its unbadged side, and military container-handler shots. |
| Port of Savannah hoisted containers (1:05-1:08) | Carrier logo readable on the container doors in the vertical crop. Not used. |
| USDA Wando Welch Terminal clip | Static, carrier name on the hull. Not used. |
| Apollo 11 press-site feed after liftoff | Commentary continues and the feed carries no launch roar. Roar comes from the NASA RS-25 hot-fire test (natural sound, no speech). |
| NASA Artemis I isolated views / slow-motion liftoff audio | Commentary on the audio track. Not used for SFX. |
| LOC "The Touaregs in their country" (1908) | Rights fine; staged ethnographic scenes. Not used. |
| LOC "S.S. Queen loading" (1897) | Rights fine; too dark to read. Not used. |
| Newsreels (e.g. LOC Pathé News, Universal Newsreel) | News broadcast material, excluded by the brief. |

## Checks run on the delivered files

`ffprobe` (codec, profile, size, fps, audio), `ebur128` loudness, a frame-by-frame scan for
unintended black frames, and review of the contact sheets and stills of every shot in both
formats, plus a speech-recognition pass over the final mix to confirm every narration line
is intelligible over the music. The two style references described in the brief
were never downloaded. The third reference the client linked for v2.1 (a 60-second brand
film on X) was downloaded once to a scratch folder outside the repo, only to study its
structure (one narrator, no subtitles, music never ducked); nothing from it is in the repo
or the film.
