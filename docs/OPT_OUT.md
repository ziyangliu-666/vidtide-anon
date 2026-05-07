# Opt-out & take-down policy

VidTide treats every clip in a frozen monthly slice as removable on request.

## Policy

- **Eligibility.** The original uploader of a clip, or a copyright/likeness rights-holder for the content of the clip, may request removal of any record from any frozen slice.
- **Turnaround.** **24 hours** from a well-formed request.
- **Effect.** The corresponding row is deleted from the canonical `manifests/<slice>/metadata.jsonl` and from any `splits/*.jsonl` files that reference it; the deletion immediately propagates to anyone who re-runs `scripts/download_videos.py` against the slice (the script will not attempt to fetch URLs that are no longer in the manifest).
- **Errata.** Removals are recorded in `manifests/<slice>/ERRATA.md` so that older paper-cited counts remain auditable.

## How to file a request (during double-blind review)

Please open an issue on this anonymous repository's issue tracker with:

1. The `id` field (or `source_url`) of the affected clip(s).
2. A brief statement of the basis for the request (uploader / rights-holder / other).
3. A contact channel for confirming completion (an email is sufficient; no ID verification beyond proof-of-control of the original platform account is required).

We do **not** require legal threats, and we do not push back on opt-out requests; the 24-hour turnaround is unconditional.

## How to file a request after acceptance

A persistent contact endpoint will be published in the camera-ready version. The 24-hour turnaround commitment will continue to apply.
