# Data Sources

## Goldberg Variations

Initial direct searches did not find a clean public ABC-format source for the
full Goldberg Variations.

The best source found so far is:

https://github.com/ksnortum/bach-goldberg

Local copy:

```text
data/raw/ksnortum-bach-goldberg
```

This repository contains LilyPond source files for the Aria and Variations 1-30.
The README says the work is based in part on Open Goldberg Variations by Kimiko
Ishizaka, and the repository is licensed CC BY-SA 4.0.

The release also includes separate MIDI files for the Aria, Aria da capo, and
Variations 1-30. These have been downloaded here:

```text
data/raw/ksnortum-bach-goldberg-midi
```

Current status:

- no `.abc` files found in the source repository;
- no `.musicxml` / `.mxl` files found in the source repository;
- local conversion tools are not installed yet (`lilypond`, `midi2abc`,
  `abc2midi`, and Python MIDI libraries were not available globally);
- `music21` is listed in the project requirements and can be used after setting
  up the virtual environment.

Likely conversion options:

1. Find a MusicXML/MuseScore source if one exists, then convert to ABC or a
   structured internal representation.
2. Use the LilyPond source directly as the highest-fidelity score source and
   write a parser/extractor for the subset we need.
3. Convert the released MIDI files to an internal note-event representation.
   This is easiest, but it may lose some notation, voice, ornament, and hand
   assignment information.

For the Goldberg-structure reward, the LilyPond source is probably more useful
than MIDI because it preserves bar structure, voices, repeats, ornaments, and
hand separation.
