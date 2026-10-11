# Example commands

## Process a single song

```bash
separate-vocals ~/Music/my_song.mp3
```

The output goes to `./out/my_song/...`.

## Process a directory of songs

```bash
for f in ~/Music/dataset/*.mp3; do
  separate-vocals "$f" --out ./results
done
```

## Use speechbrain instead of pyannote

```bash
separate-vocals my_song.mp3 --diarizer speechbrain
```

## Use a different Demucs model

```bash
separate-vocals my_song.mp3 --model htdemucs
```

## Skip diagnostic plots (faster iteration)

```bash
separate-vocals my_song.mp3 --no-diagnostics
```

## Custom pitch range

```bash
separate-vocals my_song.mp3 --fmin A2 --fmax G5
```
