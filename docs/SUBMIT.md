# Submitting to the VidTide leaderboard

Leaderboard: https://vidtide.fly.dev/leaderboard (submission form: `/leaderboard/submit`).

## What is scored

Each release `Mk` (M1–M7) has two evaluation lists:

| File | Contents | Labels |
|---|---|---|
| `manifests/Mk/eval_balanced.jsonl` | public evaluation list, equal generated / real | released |
| `manifests/Mk/private_unlabeled.jsonl` | private list from the same publication month | kept on the server |

A submission is scored on both lists. The leaderboard ranks entries by AUROC on the private list.
The private labels of a release are published when ranking moves to a newer release.

## How to submit

1. Fetch the clips listed in both files of one release (`id`, `source_url`). Use the
   repository download scripts.
2. Score every clip with your detector. A higher score means "more likely generated".
3. Write one JSON object per line:

   ```
   {"id": "0006bf3dba744d078a178dfa0e152d7f", "score": 0.93}
   ```

   At least 95% of the clips in the two lists must be scored. Clips that can no longer be
   retrieved may be omitted.
4. Upload the file at `/leaderboard/submit` with a team name, a contact e-mail (not shown), and a
   one-line method description.

## Limits

- One submission per team per release every 7 days.
- At most 5 submissions per day from one network address.

Public-list AUROC can be checked locally with `scripts/score_submission.py` before submitting.
