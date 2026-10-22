"""Outcome-independent matched donor banks; no model inference is used here."""
from __future__ import annotations
import argparse
import random
from collections import Counter
from common import *
from transformers import AutoTokenizer

NOUN_CANDIDATES = "pilot guard teacher doctor farmer lawyer driver nurse singer artist dancer sailor worker actor student soldier chef player writer friend cousin officer leader captain owner author reader parent boy girl guest".split()
ADJECTIVE_CANDIDATES = "quiet calm clever tired proud kind brave polite friendly busy careful patient curious tall short new eager gentle bright honest cheerful modest nervous serious silent wealthy hungry thirsty happy sleepy lonely helpful playful".split()
TEMPLATES = [
    "At the station, the {adj} {head} near the {distr}",
    "At the office, the {adj} {head} beside the {distr}",
]

def make_prompt(template, adjective, head, distractor, cls, corrupt):
    head_word = head + ("s" if bool(cls) ^ bool(corrupt) else "")
    distractor_word = distractor + ("s" if cls == 0 else "")
    text = TEMPLATES[template].format(adj=adjective, head=head_word, distr=distractor_word)
    start = text.index(head_word, text.index(", the ") + 6)
    return {"text": text, "anchor_span": [start, start + len(head_word)], "head_word": head_word, "end_word": distractor_word, "adjective": adjective}

def encode_prompt(tok, config, p, good, bad):
    encoded = tok(p["text"], add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    targets = []
    for word in [good, bad]:
        joined = tok.encode(p["text"] + " " + word, add_special_tokens=False)
        if len(joined) != len(ids) + 1 or joined[:-1] != ids:
            raise ValueError("continuation is not one appended token")
        targets.append(joined[-1])
    start, end = p["anchor_span"]
    hits = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if b > start and a < end]
    if len(hits) != 1:
        raise ValueError("head noun must occupy one token")
    word_start = len(p["text"]) - len(p["end_word"])
    end_hits = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if b > word_start]
    if end_hits != [len(ids) - 1]:
        raise ValueError("distractor must occupy exactly the final token")
    bos = config.get("bos_token_id")
    if bos is None:
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return {**p, "ids": [bos] + ids, "anchor": hits[0] + 1, "end": len(ids), "good_id": targets[0], "bad_id": targets[1]}

def qualify_pool(tokenizers):
    nouns = [w for w in NOUN_CANDIDATES if all(len(t.encode(" " + w + suffix, add_special_tokens=False)) == 1 for t in tokenizers.values() for suffix in ["", "s"])]
    adjectives = [w for w in ADJECTIVE_CANDIDATES if all(len(t.encode(" " + w, add_special_tokens=False)) == 1 for t in tokenizers.values())]
    assert len(nouns) >= 18 and len(adjectives) >= 17
    return nouns, adjectives

def make_group(index, split, seed, nouns, adjectives, attempt=0):
    rng = random.Random(seed + index * 1009 + attempt * 10000019)
    template = index % 2
    cls = (index // 2) % 2
    head, recipient_distractor = rng.sample(nouns, 2)
    recipient_adjective = rng.choice(adjectives)
    context_adjectives = rng.sample([x for x in adjectives if x != recipient_adjective], 8)
    labels = rng.sample([x for x in nouns if x not in [head, recipient_distractor]], 8)
    prompts = []
    def add(adjective, distractor, corrupt):
        p = make_prompt(template, adjective, head, distractor, cls, corrupt)
        prompts.append(p)
        return len(prompts) - 1
    assert [add(recipient_adjective, recipient_distractor, c) for c in [False, True]] == [0, 1]
    grid = [[[add(adj, label, c) for c in [False, True]] for label in labels] for adj in context_adjectives]
    banks = []
    for k in DOSES:
        for rotation in range(8):
            columns = [(rotation + (slot * k // 8)) % 8 for slot in range(8)]
            pairs = [grid[slot][column] for slot, column in enumerate(columns)]
            banks.append({"rotation": rotation, "k": k, "pairs": pairs, "columns": columns, "assignments": [labels[column] for column in columns], "adjectives": context_adjectives})
    good, bad = ("is", "are") if cls == 0 else ("are", "is")
    return {"id": f"{split}-token-{index:03d}", "split": split, "group": index, "seed_block": index // 16, "template": template, "class": cls, "head": head, "good": good, "bad": bad, "recipient": [0, 1], "prompts": prompts, "banks": banks, "grid": grid, "context_adjectives": context_adjectives, "labels": labels, "generation_seed": seed, "generation_attempt": attempt}

def validate_row(row):
    assert len(row["banks"]) == 8 * len(DOSES)
    assert len(row["prompts"]) == 130
    assert len({p["text"] for p in row["prompts"]}) == 130
    for model, encoded in row["encoded"].items():
        assert len({len(p["ids"]) for p in encoded}) == 1, (row["id"], model, "length")
        assert len({(p["good_id"], p["bad_id"]) for p in encoded}) == 1
        assert len({p["anchor"] for p in encoded}) == 1
        assert all(p["anchor"] < p["end"] for p in encoded)
        for k in DOSES:
            selected = [b for b in row["banks"] if b["k"] == k]
            assert len(selected) == 8
            for direction in [0, 1]:
                pooled = [pair[direction] for bank in selected for pair in bank["pairs"]]
                expected = [row["grid"][i][j][direction] for i in range(8) for j in range(8)]
                assert sorted(pooled) == sorted(expected), "Every donor occurs once at every dose"
            for bank in selected:
                assert len(set(bank["assignments"])) == k
                assert set(Counter(bank["assignments"]).values()) == {8 // k}
                for direction in [0, 1]:
                    ix = [pair[direction] for pair in bank["pairs"]]
                    assert len({encoded[i]["text"] for i in ix}) == 8
                    assert len({encoded[i]["ids"][-1] for i in ix}) == k
                    assert all(encoded[i]["ids"][-1] != encoded[direction]["ids"][-1] for i in ix)
        for slot in range(8):
            for direction in [0, 1]:
                reference = encoded[row["grid"][slot][0][direction]]
                for column in range(8):
                    current = encoded[row["grid"][slot][column][direction]]
                    # END is the only varying token within each context row.
                    assert current["ids"][:-1] == reference["ids"][:-1]
                    assert current["ids"][:current["anchor"] + 1] == reference["ids"][:reference["anchor"] + 1]
    return True

def old_strings():
    result = set()
    files = [STUDY / "data" / f"{split}.json" for split in ["calibration", "development", "test"]]
    files += [STUDY.parent / "data" / f for f in ["calibration.json", "evaluation.json"]]
    def descend(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("text"), str):
                result.add(obj["text"])
            for key, value in obj.items():
                if key != "encoded":
                    descend(value)
        elif isinstance(obj, list):
            for value in obj:
                descend(value)
    for path in files:
        if path.exists():
            descend(read(path))
    return result, [str(p.relative_to(STUDY.parent)) for p in files if p.exists()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--n-calibration", type=int, default=4)
    ap.add_argument("--n-development", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    args = ap.parse_args()
    if args.n_test % 16 or args.n_calibration % 4 or args.n_development % 4:
        raise ValueError("Use complete template/class blocks; test blocks contain 16 groups")
    tokenizers = {name: AutoTokenizer.from_pretrained(path, local_files_only=True) for name, path in MODEL_PATHS.items()}
    configs = {name: read(path / "config.json") for name, path in MODEL_PATHS.items()}
    nouns, adjectives = qualify_pool(tokenizers)
    previous, previous_files = old_strings()
    used = set(previous)
    split_strings = {}
    reject = {}
    counts = {}
    reuse = {}
    for split, n, seed in [("calibration", args.n_calibration, 26100211), ("development", args.n_development, 26100212), ("test", args.n_test, 26100213)]:
        dest = args.out / f"{split}.json"
        if dest.exists():
            raise RuntimeError(f"Refusing to overwrite {dest}")
        rows = []
        recipient_pairs = set()
        strings = set()
        reject[split] = {"overlap": 0, "tokenization": 0}
        for index in range(n):
            for attempt in range(10000):
                row = make_group(index, split, seed, nouns, adjectives, attempt)
                texts = {p["text"] for p in row["prompts"]}
                recipient_pair = tuple(p["text"] for p in row["prompts"][:2])
                if texts & used or recipient_pair in recipient_pairs:
                    reject[split]["overlap"] += 1
                    continue
                try:
                    row["encoded"] = {name: [encode_prompt(tok, configs[name], p, row["good"], row["bad"]) for p in row["prompts"]] for name, tok in tokenizers.items()}
                    validate_row(row)
                except (ValueError, AssertionError):
                    reject[split]["tokenization"] += 1
                    continue
                rows.append(row)
                recipient_pairs.add(recipient_pair)
                # Only split-level exclusion: within-split chance overlap is
                # allowed and documented, avoiding rejection-induced dependence.
                strings.update(texts)
                break
            else:
                raise RuntimeError(f"Generation failed: {split}/{index}")
        counts[split] = len(rows)
        donor_counts = Counter(p["text"] for row in rows for p in row["prompts"][2:])
        recipient_counts = Counter(p["text"] for row in rows for p in row["prompts"][:2])
        all_counts = Counter(p["text"] for row in rows for p in row["prompts"])
        reuse[split] = {"recipient_pair_duplicates": len(rows) - len(recipient_pairs), "recipient_prompt_duplicate_occurrences": sum(c - 1 for c in recipient_counts.values()), "donor_prompt_duplicate_occurrences": sum(c - 1 for c in donor_counts.values()), "unique_donor_prompts_reused": sum(c > 1 for c in donor_counts.values()), "maximum_donor_prompt_multiplicity": max(donor_counts.values()), "all_prompt_duplicate_occurrences": sum(c - 1 for c in all_counts.values()), "recipient_donor_overlap_unique_strings": len(set(recipient_counts) & set(donor_counts))}
        dump(dest, rows)
        used.update(strings)
        split_strings[split] = strings
        print(f"Prepared {split}: {n} groups", flush=True)
    audit = {"created_utc": utc(), "counts": counts, "within_split_reuse": reuse, "rejections": reject, "models": list(MODEL_PATHS), "nouns": nouns, "adjectives": adjectives, "templates": TEMPLATES, "previous_files": previous_files, "old_unique_strings": len(previous), "old_exact_overlap": {s: len(v & previous) for s, v in split_strings.items()}, "cross_split_exact_overlap": {a + "/" + b: len(split_strings[a] & split_strings[b]) for a in split_strings for b in split_strings if a < b}, "unique_strings": {s: len(v) for s, v in split_strings.items()}, "bank_size": 8, "distinct_end_tokens": DOSES, "rotations_per_dose": 8, "grid_shape": [8, 8], "balanced_donor_composition_across_doses": True, "constant_prompt_length_within_group_per_model": True, "constant_prefix_across_doses_within_donor_context": True, "unique_prompts_per_bank": 8, "unit_of_inference": "recipient group, averaging eight correlated partition banks per dose; rotations and donors are not independent observations", "scope": "Randomized partition policy of a fixed 64-donor grid changes within-bank lexical diversity and dispersion; global donor composition and global resample mean are exactly constant across doses. Not a pure token-identity mechanism."}
    dump(args.out / "generation-audit.json", audit)

if __name__ == "__main__":
    main()
