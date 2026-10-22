# Study 5: notes after the test stage

These notes record events after the stage-4 freeze. They change no code or data. `AMENDMENTS.md` lists the amendments and is itself hashed in their freeze records, so these notes live here.

- **The replay check failed.** Four IOI replays differed from the test outputs by 1.1–1.6 × 10⁻³ logits, against a tolerance of 10⁻³. Three are primary: study engine B1 and C2, and reference engine C2. The fourth is the secondary L4 replay. All agreement replays were within 10⁻⁴. Under the protocol a failed check stops the study for the author's review, so Study 5 stops here. Its results are reported in `RESULTS.md` and wait for that decision. A post hoc bound (`posthoc/replay_noise.py`, `results/posthoc/replay-noise.json`) shows that noise of this size cannot move any normalised effect across τ, so no flag or primary count depends on it.
- **Order of stages.** The analysis (stage 8) was started at 11:26 UTC, while the replays (stage 7, started 11:25 UTC) were running. The outcomes were therefore seen about 18 minutes before the replay result. The analysis is the frozen code, and no choice depended on it.
- **Billing watchdog.** During the test stage, `watch_billing.sh` (not frozen) read the billing report every 15 minutes, ready to stop the app at $10. It never needed to.
- **The author's decision** ([redacted] session, about 2026-10-17T12:00Z, after RESULTS.md and the post hoc bound were shown): "Accept Study 5's results with the replay failure and the post hoc noise bound disclosed, and write Study 5 into the AI-03 manuscript as a new version."
