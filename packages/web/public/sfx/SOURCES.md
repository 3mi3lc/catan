# Sound effects

Drop files in this folder using the **exact names** below (the manager in
[`src/audio/sound.ts`](../../src/audio/sound.ts) references these paths
directly — no manifest to edit, just add the file and it's picked up). Every
cue degrades silently if its file is missing, so the game stays fully
playable — just silent — until real audio lands.

`.mp3` is the only format referenced right now (broadest browser/WKWebView
support without extra `<source>` fallbacks). Keep clips **short** (under ~1s
for UI/build cues, under ~2.5s for the victory fanfare) and **normalized to a
similar loudness** as each other — a sound effect set where one cue is twice
as loud as the rest reads as broken, not characterful.

| File | Used for | Vibe |
|---|---|---|
| `dice-roll.mp3` | Dice rolled | Quick rattle/clatter, ~0.4–0.7s |
| `build-settlement.mp3` | Settlement placed | Light wooden "plonk" / construction tap |
| `build-city.mp3` | City built (upgrade) | Slightly heavier/grander version of the settlement sound |
| `build-road.mp3` | Road placed | Short wood-on-wood click/snap |
| `robber-move.mp3` | Robber moved, nobody to steal from | Low ominous thud/shift |
| `steal.mp3` | Robber moved AND stole a card | Sharper sting — a quick "snatch" |
| `card-buy.mp3` | Development card bought | Crisp card-flip / paper riffle |
| `trade-executed.mp3` | Player-to-player trade completed | Coin/handshake chime |
| `trade-bank.mp3` | Bank (or port) trade | Lighter single coin clink |
| `discard.mp3` | A 7 forced a discard | Soft card-drop / shuffle-out |
| `award.mp3` | Longest Road / Largest Army changes hands | Small triumphant flourish, shorter than victory |
| `victory.mp3` | Game won | Short fanfare, ~1.5–2.5s |
| `click.mp3` | Any button press (delegated, every surface) | Very short, soft UI tick — this one plays *constantly*, so keep it the quietest/least fatiguing of the set |
| `your-turn.mp3` | Online only: it becomes your turn (and wasn't a moment ago) | Gentle notification chime, distinct from dice-roll |

## Where to get them (CC0 / no-attribution-needed)

- **[Kenney.nl](https://kenney.nl/assets?q=audio)** — UI Audio pack and
  Interface Sounds pack are CC0 and have most of the "click," "card," "coin"
  textures above ready-made.
- **[Freesound.org](https://freesound.org/)** — filter by **CC0** license
  specifically (other licenses on the site need attribution). Good searches:
  "dice roll," "wood click," "coin chime," "card flip," "fanfare short,"
  "notification bell."
- **[Pixabay Sound Effects](https://pixabay.com/sound-effects/)** — free for
  commercial use, no attribution required, decent dice/coin/UI selection.

## Volume/normalization tip

If using Audacity (free): `Effect → Loudness Normalization`, target around
**-18 to -16 LUFS** for short SFX, applied to every clip in the set so
none jumps out. Trim leading/trailing silence so cues feel instant on tap.
