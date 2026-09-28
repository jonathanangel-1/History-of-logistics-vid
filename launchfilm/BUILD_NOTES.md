# Build notes: Volume launch film (v2)

v2 replaces v1's procedural picture and synthesized score with real, license-safe
footage, public-domain voices and a free commercially licensed music track. v1
(`logistics/`, root `cuesheet.json`) is untouched.

## Pipeline

```
sources.yaml ──fetch.py──> cache/sources/            (git-ignored downloads)
cuesheet.json ─┬─audio.py──> cache/work/mix.wav       (music edit + voices + SFX, mastered)
               └─picture.py─> out/*.mp4               (cut, grade, grain, type, logo; muxes the mix)
credits.py ──> CREDITS.csv / CREDITS.md               (generated from the manifest + actual uses)
contact.py ──> out/contact_sheet_{16x9,9x16}.png
```

- **One tempo map.** `cuesheet.json` is 128 BPM in 4/4 (1 bar = 1.875 s), matching the
  music track's measured tempo (128.00 BPM, first beat at 0.035 s). Shots, titles and music
  segments are placed in bars; voices and SFX in seconds (`at_s`) where they must line up
  with a word or transient. Moving a number in the sheet moves picture and sound together.
- **Picture.** Each shot is decoded by ffmpeg straight from the source (accurate seek, crop to
  the output aspect inside the source's `active` area, Lanczos scale, conform to 24 fps),
  then graded in numpy: warm B&W for the archival section, desaturated color for the box and
  air beats, full color for "now", a single light grain/vignette pass over everything.
  Optional per-shot slow push-in, 3-frame white flash on the clang / slam / liftoff, and a
  2-frame chromatic glitch on three modern cuts only. The 9:16 cut uses its own per-shot crop
  (`c9`), not a pillarbox; captions sit at y = 1390 of 1920 (well inside the 250 px margins).
- **Type.** Inter (site font). Lowercase kinetic lines revealed word by word on the beat;
  voice captions in sentence case with a small monospace historical label ("RICE UNIVERSITY ·
  1962", "APOLLO 11 LAUNCH CONTROL · 1969") so speakers read as historical speeches, not
  endorsements.
- **Logo.** `logo.py` rasterizes `brand/volume-logo.svg` itself (the file only uses absolute
  M/H/V/L/Q/Z paths, a rounded rect and translate/scale transforms). Letterforms render cream
  `#eee9de`, the square stays gold `#baa88a`. Nothing is redrawn. On the final hit the gold
  square lands first, then the letters resolve outward from it.
- **Sound.** The track is cut on bar boundaries (6 ms pre-roll so downbeats stay intact):
  build (track bars 6–15.75) with a low-pass sweep into a half-beat of near-silence, the clang
  on the downbeat as the big section enters (track bar 16), a full drop-out under "We choose
  to go to the Moon", the slam back in on track bar 61, a hard stop on "Liftoff!", then the
  track's biggest downbeat (bar 68) as the logo hit, ringing out under the end card. Voices
  are high-passed, lightly de-noised (ffmpeg `afftdn`, a classic spectral filter), EQ'd,
  compressed and level-matched; the music ducks 11–16 dB under each line with a 50 ms release
  so it slams back in. Master: iterative gain to -14 LUFS integrated with a 4x-oversampled
  true-peak limiter at -1.3 dBFS (measured with ffmpeg `ebur128`).

## Commands

```bash
sudo apt-get install -y ffmpeg fonts-inter        # or drop Inter + JetBrains Mono TTFs into ./fonts/
pip install --break-system-packages numpy scipy opencv-python-headless pillow pyyaml

python3 -m launchfilm.render --all --artifacts /opt/cursor/artifacts   # everything, ~10 min on 4 cores
python3 -m launchfilm.render --fetch                                   # downloads only (~3.5 GB)
python3 -m launchfilm.render --audio                                   # re-mix only
python3 -m launchfilm.render --format 16x9                             # one cut (reuses the mix)
python3 -m launchfilm.render --stills 22.5,52.6 --format 9x16          # PNG stills -> out/stills/
python3 -m launchfilm.render --credits                                 # regenerate CREDITS.*
```

The end card URL is the single field `endcard.url` in `cuesheet.json`.

## Source and rights decisions

Every clip in the film is a U.S. federal government work (Library of Congress early films
with no known restrictions, NASA, DVIDS), a CC0 recording, or the one CC BY 4.0 music track.
Each item's own page was checked on 2026-09-28; the quoted statements are in `sources.yaml`
and `CREDITS.*`. Creators are credited as institutions / units; the only personal names in the
credits are the historical speaker (President Kennedy) and the composer, whose credit line is
required by the CC BY 4.0 license. CC0 contributors are credited as "Wikimedia Commons
contributor" with the file URL (CC0 requires no attribution).

Voice lines were located with an offline speech recognizer run once during editing to get
word timings; it is not part of the pipeline and nothing it produced is in the film.

### Rejected or unavailable

| Item | Reason |
|---|---|
| Pixabay Music, Pixabay video, Pexels | pixabay.com and pexels.com return HTTP 403 (bot wall) from the build VM. Not bypassed. Music comes from incompetech (CC BY 4.0); modern footage from DVIDS/NASA. |
| Freesound SFX | Downloads require a login. Not used; SFX are CC0 files from Wikimedia Commons or natural sound from the PD footage. |
| jfklibrary.org asset pages | HTTP 403 from the VM. Used the JFK Library recording mirrored on Wikimedia Commons (PD-USGov, source stated as the JFK Library). |
| Commons "President Kennedys Speech at Rice University" (.wav / .ogv) | File pages credit a YouTube upload as the source. Rejected (no YouTube rips) in favor of the JFK Library copy. |
| loc.gov HTML item pages | Behind a browser challenge; not bypassed. Rights text taken from the LOC JSON API for the same items. |
| Commons CC BY / CC BY-SA "own work" port videos (Malmö, Malta Freeport, Rauma, etc.) | Attribution would put private individuals' names in the credits; many Commons CC-BY videos are also YouTube-sourced. Avoided. |
| DVIDS "Night Port Operations" (893855) | Armored vehicles / weapons. Not used. |
| DVIDS "Rapid C-17 Globemaster III takeoff" (890820) | Mostly identifiable personnel close-ups. Not used. |
| DVIDS CBP inspection shots with officers, patches, "US CUSTOMS" signage | Identifiable people and possible implied government affiliation. Not used. |
| DVIDS C-17 B-roll at 0:04 | Legible "U.S. AIR FORCE" / unit markings (DoW terms restrict insignia in commerce). Used the unmarked engine/wing angle instead; C-5 nose cropped away from its lettering. |
| Port of Savannah hoist at 0:59 | Carrier logo (CMA CGM) is the subject of the shot. Replaced with an unbranded container lift. |
| Long Beach package 2 picture | Carrier-branded containers dominate the frame. Only its natural sound (a container set on a chassis: the clang) is used. |
| Savannah rail shots | Crane-maker branding across the top of frame. Cropped out (`c16`/`c9` zoomed to the lower frame); the 0:27 in-point (readable carrier name on a container) dropped entirely. |
| Long Beach package 1 cranes and yard trucks (0:49, 2:57, 3:53, 5:51) and a Long Beach yard at 0:34 | Readable carrier / crane-maker / fleet names on crane booms, trucks, or a hoisted container. Replaced with the bridge vista, gate truck cropped to its unbadged side, and military container-handler shots. |
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
formats. The two style-reference videos were never downloaded or opened on the build machine.
