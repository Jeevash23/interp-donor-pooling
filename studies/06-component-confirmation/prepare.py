"""Generate Study 4's agreement prompts without consulting any model output.

The agreement generator is copied from Study 3 (studies/05-head-confirmation/prepare.py), which copied it
from Study 2, so the word pools and templates are the same. New seeds; every prompt string used by any
earlier AI-03 study (P1, P2, Study 1, Study 2, Study 3) is excluded, and the three splits do not overlap.

Outputs (data/): sva-calibration.json, sva-development.json, sva-test.json, generation-audit.json
"""
from __future__ import annotations

import random
from collections import Counter

from transformers import AutoTokenizer

from common import MODEL_PATHS, ROOT, dump, prior_prompt_strings, read, sha, utc

# ----------------------------------------------------------------------------- SVA (October generator)
NOUN_CANDIDATES = ("pilot guard teacher doctor farmer lawyer driver nurse singer artist dancer sailor worker "
                   "actor student soldier chef player writer friend cousin officer leader captain owner author "
                   "reader parent boy girl guest").split()
ADJECTIVE_CANDIDATES = ("quiet calm clever tired proud kind brave polite friendly busy careful patient curious "
                        "tall short new eager gentle bright honest cheerful modest nervous serious silent wealthy "
                        "hungry thirsty happy sleepy lonely helpful playful").split()
SVA_TEMPLATES = ["At the station, the {adj} {head} near the {distr}",
                 "At the office, the {adj} {head} beside the {distr}"]


def sva_prompt(template, adjective, head, distractor, cls, corrupt):
    head_word = head + ("s" if bool(cls) ^ bool(corrupt) else "")
    distractor_word = distractor + ("s" if cls == 0 else "")
    text = SVA_TEMPLATES[template].format(adj=adjective, head=head_word, distr=distractor_word)
    start = text.index(head_word, text.index(", the ") + 6)
    return {"text": text, "anchor_span": [start, start + len(head_word)], "head_word": head_word,
            "end_word": distractor_word, "adjective": adjective}


def sva_encode(tok, config, p, good, bad):
    enc = tok(p["text"], add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    targets = []
    for word in [good, bad]:
        joined = tok.encode(p["text"] + " " + word, add_special_tokens=False)
        if len(joined) != len(ids) + 1 or joined[:-1] != ids:
            raise ValueError("continuation is not one appended token")
        targets.append(joined[-1])
    start, end = p["anchor_span"]
    hits = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < end]
    if len(hits) != 1:
        raise ValueError("head noun must occupy one token")
    word_start = len(p["text"]) - len(p["end_word"])
    if [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > word_start] != [len(ids) - 1]:
        raise ValueError("distractor must occupy exactly the final token")
    adj_start = p["text"].index(" " + p["adjective"] + " ", p["text"].index(", the ")) + 1
    adj_hits = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > adj_start and a < adj_start + len(p["adjective"])]
    if adj_hits != [hits[0] - 1]:
        raise ValueError("adjective must occupy the token before the head")
    bos = config.get("bos_token_id")
    if bos is None:
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return {**p, "ids": [bos] + ids, "adj": hits[0], "anchor": hits[0] + 1, "end": len(ids),
            "good_id": targets[0], "bad_id": targets[1]}


def sva_group(index, split, seed, nouns, adjectives, attempt=0):
    rng = random.Random(seed + index * 1009 + attempt * 10000019)
    template, cls = index % 2, (index // 2) % 2
    head, recipient_distractor = rng.sample(nouns, 2)
    recipient_adjective = rng.choice(adjectives)
    context_adjectives = rng.sample([x for x in adjectives if x != recipient_adjective], 8)
    labels = rng.sample([x for x in nouns if x not in [head, recipient_distractor]], 8)
    prompts = []

    def add(adjective, distractor, corrupt):
        prompts.append(sva_prompt(template, adjective, head, distractor, cls, corrupt))
        return len(prompts) - 1

    assert [add(recipient_adjective, recipient_distractor, c) for c in [False, True]] == [0, 1]
    grid = [[[add(adj, label, c) for c in [False, True]] for label in labels] for adj in context_adjectives]
    good, bad = ("is", "are") if cls == 0 else ("are", "is")
    return {"id": f"{split}-sva-{index:03d}", "split": split, "group": index, "template": template, "class": cls,
            "head": head, "good": good, "bad": bad, "recipient": [0, 1], "prompts": prompts, "grid": grid,
            "context_adjectives": context_adjectives, "labels": labels, "generation_seed": seed,
            "generation_attempt": attempt}


def sva_validate(row):
    assert len(row["prompts"]) == 130 and len({p["text"] for p in row["prompts"]}) == 130
    for model, encoded in row["encoded"].items():
        assert len({len(p["ids"]) for p in encoded}) == 1, (row["id"], model)
        assert len({(p["good_id"], p["bad_id"]) for p in encoded}) == 1
        assert len({p["anchor"] for p in encoded}) == 1 and len({p["adj"] for p in encoded}) == 1
        for slot in range(8):
            for d in [0, 1]:
                ref = encoded[row["grid"][slot][0][d]]
                for column in range(8):
                    cur = encoded[row["grid"][slot][column][d]]
                    assert cur["ids"][:-1] == ref["ids"][:-1]
        for column in range(8):
            for d in [0, 1]:
                ends = {encoded[row["grid"][slot][column][d]]["ids"][-1] for slot in range(8)}
                assert len(ends) == 1
    return True


def make_sva(tokenizers, configs, split, n, seed, used):
    nouns = [w for w in NOUN_CANDIDATES if all(len(t.encode(" " + w + sfx, add_special_tokens=False)) == 1
                                               for t in tokenizers.values() for sfx in ["", "s"])]
    adjectives = [w for w in ADJECTIVE_CANDIDATES if all(len(t.encode(" " + w, add_special_tokens=False)) == 1
                                                         for t in tokenizers.values())]
    rows, local, pairs, rejects = [], set(), set(), Counter()
    for index in range(n):
        for attempt in range(10000):
            row = sva_group(index, split, seed, nouns, adjectives, attempt)
            texts = {p["text"] for p in row["prompts"]}
            pair = tuple(p["text"] for p in row["prompts"][:2])
            if texts & used or pair in pairs:
                rejects["overlap"] += 1
                continue
            try:
                row["encoded"] = {name: [sva_encode(tok, configs[name], p, row["good"], row["bad"]) for p in row["prompts"]]
                                  for name, tok in tokenizers.items()}
                sva_validate(row)
            except (ValueError, AssertionError):
                rejects["tokenization"] += 1
                continue
            rows.append(row)
            pairs.add(pair)
            local.update(texts)
            break
        else:
            raise RuntimeError(f"SVA generation failed: {split}/{index}")
    return rows, local, dict(rejects), nouns, adjectives





def main():
    out = ROOT / "data"
    if (out / "generation-audit.json").exists():
        raise RuntimeError("refusing to overwrite prepared data")
    prior, prior_files = prior_prompt_strings()
    used = set(prior)
    tokenizers = {name: AutoTokenizer.from_pretrained(path, local_files_only=True) for name, path in MODEL_PATHS.items()}
    configs = {name: read(path / "config.json") for name, path in MODEL_PATHS.items()}
    audit = {"created_utc": utc(), "prior_files": prior_files, "prior_unique_strings": len(prior), "splits": {}}
    for split, n, seed in [("sva-calibration", 4, 26100661), ("sva-development", 4, 26100662), ("sva-test", 64, 26100663)]:
        rows, local, rejects, nouns, adjectives = make_sva(tokenizers, configs, split, n, seed, used)
        used |= local
        dump(out / f"{split}.json", rows)
        audit["splits"][split] = {"groups": len(rows), "rejections": rejects}
    audit.update(sva_nouns=nouns, sva_adjectives=adjectives, uses_model_outputs=False,
                 note="Word pools filtered with both the GPT-2 and Qwen3 tokenizers, as in Studies 2 and 3; only GPT-2 is run.",
                 files={p.name: sha(p) for p in sorted(out.glob("*.json"))})
    dump(out / "generation-audit.json", audit)
    print(audit["splits"], flush=True)


if __name__ == "__main__":
    main()
