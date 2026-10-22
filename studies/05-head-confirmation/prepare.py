"""Generate the confirmation study's prompts without consulting any model output.

The generators are copied from Study 2 (studies/04-reference-and-sweeps/prepare.py) so that this
folder is self-contained. New seeds; every prompt string used by any earlier AI-03 study is excluded.

Outputs (data/):
  sva-calibration.json, sva-development.json, sva-test.json     agreement groups (8 x 8 donor grids)
  ioi-calibration.json, ioi-development.json, ioi-test.json     IOI prompts with 64 canonical ABC donors
  generation-audit.json
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter

from transformers import AutoTokenizer

from common import MODEL_PATHS, ROOT, dump, prior_prompt_strings, read, sha, utc

# ----------------------------------------------------------------------------- IOI
NAMES = ("Alice Bob Carol David Emma Frank Grace Henry Jack James Jane John Julia Kate Laura Lucy Luke "
         "Mark Mary Michael Mike Nick Oliver Paul Peter Rachel Richard Robert Rose Sam Sarah Simon Sophie "
         "Steve Susan Thomas Tom Victor William Anna Ben Daniel Edward George Helen Joe Kevin Linda Lisa "
         "Maria Martin Nancy Patrick Rebecca Roger Steven Tony Walter Wendy").split()
PLACES = ("store garden restaurant school hospital office house station park market hotel museum "
          "library beach church theater").split()
OBJECTS = ("ring kiss bone basketball computer necklace drink snack book ball cup bag box hat map toy "
           "key letter flower ticket bottle").split()
# Family O: the four October templates. Family W: the thirteen Wang et al. (2023) templates
# that contain both a place and an object (two of their fifteen have neither).
IOI_TEMPLATES = {
    "O0": "When {n1} and {n2} went to the {place}, {s} gave a {obj} to",
    "O1": "After {n1} and {n2} visited the {place}, {s} handed a {obj} to",
    "O2": "While {n1} and {n2} were at the {place}, {s} passed a {obj} to",
    "O3": "Before {n1} and {n2} left the {place}, {s} offered a {obj} to",
    "W0": "Then, {n1} and {n2} went to the {place}. {s} gave a {obj} to",
    "W1": "Then, {n1} and {n2} had a lot of fun at the {place}. {s} gave a {obj} to",
    "W2": "Then, {n1} and {n2} were working at the {place}. {s} decided to give a {obj} to",
    "W3": "Then, {n1} and {n2} were thinking about going to the {place}. {s} wanted to give a {obj} to",
    "W4": "After {n1} and {n2} went to the {place}, {s} gave a {obj} to",
    "W5": "When {n1} and {n2} got a {obj} at the {place}, {s} decided to give it to",
    "W6": "When {n1} and {n2} got a {obj} at the {place}, {s} decided to give the {obj} to",
    "W7": "While {n1} and {n2} were working at the {place}, {s} gave a {obj} to",
    "W8": "While {n1} and {n2} were commuting to the {place}, {s} gave a {obj} to",
    "W9": "After the lunch, {n1} and {n2} went to the {place}. {s} gave a {obj} to",
    "W10": "Afterwards, {n1} and {n2} went to the {place}. {s} gave a {obj} to",
    "W11": "The {place} {n1} and {n2} went to had a {obj}. {s} gave it to",
    "W12": "Friends {n1} and {n2} found a {obj} at the {place}. {s} gave it to",
}
FAMILIES = {"O": [k for k in IOI_TEMPLATES if k.startswith("O")], "W": [k for k in IOI_TEMPLATES if k.startswith("W")]}
BANK_SIZES = {"N8": 8, "NPO8": 8, "N128": 128, "NPO128": 128}


def ioi_encode(tok, text, target, subject):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    def token_at(start, word):
        hits = [i for i, (a, b) in enumerate(offsets) if b > start and a < start + len(word)]
        if len(hits) != 1:
            raise ValueError("name is not one token in context")
        return hits[0]

    starts = [i for i in range(len(text)) if text.startswith(subject, i)
              and (i == 0 or not text[i - 1].isalpha()) and (i + len(subject) == len(text) or not text[i + len(subject)].isalpha())]
    if len(starts) != 2:
        raise ValueError("subject must occur exactly twice")
    t_starts = [i for i in range(len(text)) if text.startswith(target, i)
                and (i == 0 or not text[i - 1].isalpha()) and (i + len(target) == len(text) or not text[i + len(target)].isalpha())]
    if len(t_starts) != 1:
        raise ValueError("target must occur once")
    s1, s2, io = token_at(starts[0], subject), token_at(starts[1], subject), token_at(t_starts[0], target)
    answers = []
    for name in [target, subject]:
        ext = tok.encode(text + " " + name, add_special_tokens=False)
        if len(ext) != len(ids) + 1 or ext[:-1] != ids:
            raise ValueError("answer is not one appended token")
        answers.append(ext[-1])
    # No BOS token, matching Wang et al. and the October circuit assay.
    return {"text": text, "ids": ids, "io": io, "s1": s1, "s2": s2, "s1_plus1": s1 + 1,
            "end": len(ids) - 1, "target_id": answers[0], "contrast_id": answers[1]}


def ioi_recipient(tok, template, order, rng):
    target, subject = rng.sample(NAMES, 2)
    place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
    n1, n2 = (target, subject) if order == 0 else (subject, target)
    text = IOI_TEMPLATES[template].format(n1=n1, n2=n2, s=subject, place=place, obj=obj)
    enc = ioi_encode(tok, text, target, subject)
    return {"template": template, "family": template[0], "order": "ABBA" if order == 0 else "BABA",
            "target": target, "subject": subject, "place": place, "object": obj, "recipient": enc}


def abc_donor(tok, row, rng, content, used_triplets):
    """One ABC donor: three new distinct names; with content 'NPO' the place and object also change."""
    rec = row["recipient"]
    available = [n for n in NAMES if n not in (row["target"], row["subject"])]
    for _ in range(1000):
        triplet = tuple(rng.sample(available, 3))
        if triplet not in used_triplets:
            break
    else:
        raise RuntimeError("could not draw a new name triple")
    place, obj = row["place"], row["object"]
    if content == "NPO":
        place = rng.choice([p for p in PLACES if p != row["place"]])
        obj = rng.choice([o for o in OBJECTS if o != row["object"]])
    text = IOI_TEMPLATES[row["template"]].format(n1=triplet[0], n2=triplet[1], s=triplet[2], place=place, obj=obj)
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) != len(rec["ids"]):
        raise ValueError("donor length differs from recipient")
    changed = [k for k, (x, y) in enumerate(zip(ids, rec["ids"])) if x != y]
    names_only = sorted({rec["io"], rec["s1"], rec["s2"]})
    if content == "N" and changed != names_only:
        raise ValueError("names-only donor changes a non-name position")
    if content == "NPO" and not set(names_only) <= set(changed):
        raise ValueError("donor keeps a recipient name position")
    return triplet, {"text": text, "ids": ids, "place": place, "object": obj, "names": list(triplet)}


def ioi_banks(tok, row, prior):
    """Deterministic nested donor banks for one selected recipient (frozen rule).

    N128 / NPO128 each hold 128 distinct donors; N8 / NPO8 are their first eight.
    """
    banks = {}
    for content in ["N", "NPO"]:
        seed = int(hashlib.sha256(f"study4-banks-{row['id']}-{content}".encode()).hexdigest()[:12], 16)
        rng = random.Random(seed)
        donors, triplets, texts = [], set(), set()
        while len(donors) < 128:
            triplet, donor = abc_donor(tok, row, rng, content, triplets)
            triplets.add(triplet)
            if donor["text"] in texts or donor["text"] in prior:
                continue
            texts.add(donor["text"])
            donors.append(donor)
        banks[content + "128"] = donors
        banks[content + "8"] = donors[:8]
    return banks


def make_ioi_split(tok, split, n_per_cell, templates, seed, used, prior):
    rows, local, rejects = [], set(), Counter()
    cells = [(t, o) for t in templates for o in (0, 1)]
    for g in range(n_per_cell * len(cells)):
        template, order = cells[g % len(cells)]
        rng = random.Random(seed + g * 7919)
        for attempt in range(1000):
            try:
                row = ioi_recipient(tok, template, order, rng)
            except ValueError:
                rejects["encoding"] += 1
                continue
            text = row["recipient"]["text"]
            if text in used or text in local:
                rejects["overlap"] += 1
                continue
            row.update(id=f"{split}-ioi-{g:05d}", split=split, group=g, stratum=f"{template}-{row['order']}",
                       selection_priority=hashlib.sha256(f"study4-priority-{split}-{g}".encode()).hexdigest())
            rows.append(row)
            local.add(text)
            break
        else:
            raise RuntimeError(f"could not generate {split} row {g}")
    return rows, local, dict(rejects)


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




def ioi_panel(tok, split, n_per_cell, seed, used, prior, n_donors=64):
    """IOI prompts in templates O0 and W0, both orders, each with 64 canonical (names-place-object) ABC donors."""
    rows, local, rejects = make_ioi_split(tok, split, n_per_cell, ["O0", "W0"], seed, used, prior)
    for row in rows:
        rng = random.Random(int(hashlib.sha256(f"study5-donors-{row['id']}".encode()).hexdigest()[:12], 16))
        donors, triplets, texts = [], set(), set()
        while len(donors) < n_donors:
            triplet, donor = abc_donor(tok, row, rng, "NPO", triplets)
            triplets.add(triplet)
            if donor["text"] in texts or donor["text"] in prior or donor["text"] in used or donor["text"] in local:
                continue
            texts.add(donor["text"])
            donors.append(donor)
        row["donors"] = donors
        # Token at the repeated-subject position: the grouping key for heads patched there.
        row["s2_tokens"] = [d["ids"][row["recipient"]["s2"]] for d in donors]
        local.update(texts)
    return rows, local, rejects


def main():
    out = ROOT / "data"
    if (out / "generation-audit.json").exists():
        raise RuntimeError("refusing to overwrite prepared data")
    prior, prior_files = prior_prompt_strings()
    used = set(prior)
    gpt2 = AutoTokenizer.from_pretrained(MODEL_PATHS["gpt2-small"], local_files_only=True)
    tokenizers = {name: AutoTokenizer.from_pretrained(path, local_files_only=True) for name, path in MODEL_PATHS.items()}
    configs = {name: read(path / "config.json") for name, path in MODEL_PATHS.items()}
    audit = {"created_utc": utc(), "prior_files": prior_files, "prior_unique_strings": len(prior), "splits": {}}
    for split, n, seed in [("sva-calibration", 4, 26100621), ("sva-development", 4, 26100622), ("sva-test", 64, 26100623)]:
        rows, local, rejects, nouns, adjectives = make_sva(tokenizers, configs, split, n, seed, used)
        used |= local
        dump(out / f"{split}.json", rows)
        audit["splits"][split] = {"groups": len(rows), "rejections": rejects}
    for split, per_cell, seed in [("ioi-calibration", 1, 26100631), ("ioi-development", 2, 26100632), ("ioi-test", 16, 26100633)]:
        rows, local, rejects = ioi_panel(gpt2, split, per_cell, seed, used, prior)
        used |= local
        dump(out / f"{split}.json", rows)
        audit["splits"][split] = {"rows": len(rows), "rejections": rejects, "donors_per_row": 64}
    audit.update(sva_nouns=nouns, sva_adjectives=adjectives, ioi_templates={k: IOI_TEMPLATES[k] for k in ["O0", "W0"]},
                 uses_model_outputs=False, files={p.name: sha(p) for p in sorted(out.glob("*.json"))})
    dump(out / "generation-audit.json", audit)
    print(audit["splits"], flush=True)


if __name__ == "__main__":
    main()
