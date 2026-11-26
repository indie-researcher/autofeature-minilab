# Abstract and post drafts

## Abstract (about 190 words)

LLM-driven feature-engineering loops typically keep a candidate feature whenever validation performance improves, which raises the worry of multiple-testing overfitting. We run a pre-registered case study on the TalkingData AdTracking dataset with a fixed feature grammar, a sealed holdout day opened once, and planted decoy features as canaries. Under the sealed day, validation gains transferred almost fully: a keep-if-better gate did not overfit, and random proposals matched or beat Qwen2.5 3B/7B/14B proposers; the random-vs-LLM difference traced to one rare feature found by chance. Planted decoys show both error types of a gate: keep-if-better accepted 13 of 390 decoys (3.3%) versus 0 of 390 for an online-FDR (LORD++) gate, yet 78% of the candidates the FDR gate rejected had positive sealed gains versus 3% of decoys. A regime map from 7k to 712k training rows finds that no gate tested (LORD++ with 12 or 24 blocks, and a decoy-calibrated gate) beats keep-if-better on net holdout gain; strict gates starve, and the decoy-calibrated gate ties keep-if-better from 144k rows with about 2.7x fewer accepted decoys. Limits: one dataset, a random proposer in the regime map, and a reused holdout day.

## Post draft (about 150 words)

I tried to break my own agentic feature search.

Pre-registered, a sealed holdout day opened once, planted decoys as canaries, on the TalkingData dataset.

What I found:
- The sealed day confirmed validation almost exactly. "Keep it if validation improves" didn't overfit.
- Random search matched LLM proposers (Qwen 3B/7B/14B). The gap traced to ONE feature, proposed once, by chance.
- Keep-if-better accepted 3.3% of planted noise. A strict FDR gate accepted 0%, but it also rejected real signal (78% of its rejects had positive sealed gains).
- From 7k to 712k rows, no gate I tried beat keep-if-better on net holdout gain. A decoy-calibrated gate tied it with about 2.7x fewer false accepts.

One dataset, small cells, honest limits. Every verdict, including the nulls, is in the repo.

[repo link]
