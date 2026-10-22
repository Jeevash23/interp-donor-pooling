"""Design invariants for Study 4 (pytest). These read only the frozen inputs."""
from __future__ import annotations

import numpy as np
import pytest
from transformers import AutoTokenizer

from common import MODEL_PATHS, ROOT, read
from prepare import BANK_SIZES, ioi_banks
from run_sweep import ADJ, MIX, NOUN


def cells(partition):
    return sorted((s, n) for bank in partition for s, n in bank)


@pytest.mark.parametrize("partition", [MIX, NOUN, ADJ])
def test_every_partition_uses_each_donor_once(partition):
    assert cells(partition) == [(s, n) for s in range(8) for n in range(8)]
    assert all(len(set(b)) == 8 for b in partition)


def test_partition_contents():
    assert all(len({n for _, n in b}) == 8 and len({s for s, _ in b}) == 8 for b in MIX)
    assert all(len({n for _, n in b}) == 1 for b in NOUN)
    assert all(len({s for s, _ in b}) == 1 for b in ADJ)


def test_sva_groups_share_prefix_within_context_rows():
    rows = read(ROOT / "data/sva-test.json")
    for row in rows[:8]:
        for model, enc in row["encoded"].items():
            for s in range(8):
                ids = [enc[row["grid"][s][n][0]]["ids"] for n in range(8)]
                assert all(x[:-1] == ids[0][:-1] for x in ids)
                assert len({x[-1] for x in ids}) == 8


@pytest.fixture(scope="module")
def banks():
    tok = AutoTokenizer.from_pretrained(MODEL_PATHS["gpt2-small"], local_files_only=True)
    from common import prior_prompt_strings
    prior, _ = prior_prompt_strings()
    rows = read(ROOT / "data/ioi-selected-test.json")[:6]
    return tok, [(row, ioi_banks(tok, row, prior)) for row in rows]


def test_banks_are_nested_and_sized(banks):
    _, pairs = banks
    for _, b in pairs:
        assert {k: len(v) for k, v in b.items()} == BANK_SIZES
        assert b["N8"] == b["N128"][:8] and b["NPO8"] == b["NPO128"][:8]
        for k in ("N128", "NPO128"):
            assert len({d["text"] for d in b[k]}) == 128


def test_names_only_donors_change_exactly_the_name_slots(banks):
    _, pairs = banks
    for row, b in pairs:
        rec = row["recipient"]
        slots = sorted({rec["io"], rec["s1"], rec["s2"]})
        for d in b["N128"]:
            changed = [i for i, (x, y) in enumerate(zip(d["ids"], rec["ids"])) if x != y]
            assert changed == slots
            assert row["target"] not in d["names"] and row["subject"] not in d["names"]
        for d in b["NPO128"]:
            assert d["place"] != row["place"] and d["object"] != row["object"]
            assert len(d["ids"]) == len(rec["ids"])


def test_selected_recipients_carry_population_weights():
    rows = read(ROOT / "data/ioi-selected-test.json")
    audit = read(ROOT / "results/reference/screen-selection.json")
    for key, s in audit["strata"].items():
        fam, label = key.split("/")
        sel = [r for r in rows if r["family"] == fam and r["stratum_label"] == label]
        assert len(sel) == s["selected"]
        assert np.allclose([r["weight"] for r in sel], s["population"] / s["selected"])
