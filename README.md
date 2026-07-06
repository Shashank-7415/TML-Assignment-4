# Reproducing Our Best Leaderboard Result (Score: 0.361)

## Requirements

```
pip install numpy pillow scipy
```

## Steps

1. Place the dataset so the following structure exists relative to
   `task_template.py`:
   ```
   Dataset/
     watermarked_sources/
       WM_1/ ... WM_8/   (25 .png images each)
     clean_targets/
       1.png ... 200.png
   ```
   If you instead have `Dataset.zip`, set `ZIP_FILE` in
   `task_template.py` to its path — the script will extract it
   automatically on first run.

2. Run:
   ```
   python task_template.py
   ```
   This generates `submission.zip` containing the 200 forged images.
   The per-group watermark strengths (`ALPHA_OVERRIDES` in the script)
   are already set to the values that produced our best score of
   0.361 — do not change them if you want to reproduce this result.

3. Submit the result:
   ```
   python submission.py
   ```
   Make sure `FILE_PATH` in `submission.py` points to the
   `submission.zip` generated in step 2.

4. Check the leaderboard to confirm the score.

No other configuration or files are required to reproduce this
result.
