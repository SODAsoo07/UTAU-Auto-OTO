"""Unit tests for 2-token transition alias → adjacent filename-token slot matching.

Targets the 2026-05-17 val-gate failure mode where v-cv / vc rows like
``u k`` on 8-mora tense+coda wavs collapse to slot 0 (worst-case offset
−1990 ms). The new `_match_compound_alias_to_token_transition` recovers
these by checking adjacent filename-token pairs, opt-in via
``UTOA_BOUNDARY_COMPOUND_TOKEN_SLOT_MATCH_ENABLE``.
"""
from __future__ import annotations

from core.coarse_crnn.boundary_targets import (
    _compound_token_slot_match_enabled,
    _match_compound_alias_to_token_transition,
    _strip_color_suffix,
)


# ---------------------------------------------------------------------------
# _strip_color_suffix
# ---------------------------------------------------------------------------

def test_strip_color_suffix_removes_underscore_tail():
    assert _strip_color_suffix("b a_W") == "b a"
    assert _strip_color_suffix("hi_W") == "hi"
    assert _strip_color_suffix("ka_X") == "ka"


def test_strip_color_suffix_keeps_plain_token():
    assert _strip_color_suffix("ka") == "ka"
    assert _strip_color_suffix("eu") == "eu"


def test_strip_color_suffix_handles_empty():
    assert _strip_color_suffix("") == ""
    assert _strip_color_suffix(None) == ""


# ---------------------------------------------------------------------------
# _match_compound_alias_to_token_transition
# ---------------------------------------------------------------------------

def test_compound_match_returns_none_for_single_token_alias():
    tokens = ["geu", "geo", "ge", "ga"]
    assert _match_compound_alias_to_token_transition("geu", tokens) is None
    assert _match_compound_alias_to_token_transition("", tokens) is None


def test_compound_match_unambiguous_ko_v_cv_8mora_tense_coda():
    """The original failing case: `u k` should land on slot 4 (kku → kkeu)."""
    tokens = ["kkung", "kkun", "kkum", "kkul", "kku", "kkeu", "kka", "kkeu"]
    assert _match_compound_alias_to_token_transition("u k", tokens) == 4


def test_compound_match_unambiguous_ko_v_cv_8mora_tt():
    tokens = ["ttung", "ttun", "ttum", "ttul", "ttu", "tteu", "tta", "tteu"]
    assert _match_compound_alias_to_token_transition("u t", tokens) == 4


def test_compound_match_unambiguous_ko_v_cv_8mora_pp():
    tokens = ["pping", "ppin", "ppim", "ppil", "ppi", "ppeu", "ppo", "ppeu"]
    assert _match_compound_alias_to_token_transition("i p", tokens) == 4


def test_compound_match_vowel_chain_single_match():
    """Vowel-only chain `[a,a,i,a,u,a,e,a]`: `a a` has exactly one adjacency at slot 0."""
    tokens = ["a", "a", "i", "a", "u", "a", "e", "a"]
    assert _match_compound_alias_to_token_transition("a a", tokens) == 0


def test_compound_match_eu_distinct_from_u():
    """`u k` must NOT match `kkeu → kka`; `eu k` must match it."""
    tokens = ["kkung", "kkun", "kkum", "kkul", "kku", "kkeu", "kka", "kkeu"]
    assert _match_compound_alias_to_token_transition("u k", tokens) == 4
    assert _match_compound_alias_to_token_transition("eu k", tokens) == 5


def test_compound_match_rejects_coda_bearing_tokens_on_left():
    """`u k` should only fire on the bare `kku → kkeu`, not coda-bearing kku{NG,N,M,L}."""
    tokens = ["kkung", "kkun", "kkum", "kkul", "kku", "kkeu", "kka", "kkeu"]
    # If the matcher counted coda variants, this would have 5 matches → None.
    assert _match_compound_alias_to_token_transition("u k", tokens) == 4


def test_compound_match_ja_color_tag_stripped():
    """`a m1_W` (color tag) should still match the m-onset adjacency."""
    tokens = ["ma", "mi", "mu", "me", "mo", "ma", "n", "ma"]
    # `a m1` → left "a", right "m1" → onset "m". Adjacent ma→mi (idx 0),
    # ma→n (idx 5 — n not m). So only idx 0 → unambiguous slot 0.
    # But also ma→n doesn't match m onset, so still single.
    # Actually mo→ma (idx 4): mo doesn't end with a. So only ma at idx 0
    # → mi (idx 1) starts with m. And ma at idx 5 → n (idx 6) starts with n not m.
    # So exactly one match.
    assert _match_compound_alias_to_token_transition("a m1_W", tokens) == 0


def test_compound_match_returns_none_for_three_token_alias():
    tokens = ["a", "b", "c"]
    assert _match_compound_alias_to_token_transition("a b c", tokens) is None


def test_compound_match_returns_none_when_left_does_not_end_token():
    tokens = ["geu", "geo", "ge", "ga"]
    # `o k` — no token ends with 'o' followed by token starting with k.
    assert _match_compound_alias_to_token_transition("o k", tokens) is None


def test_compound_match_falls_back_to_one_char_onset():
    """Right onset `kk` (cluster) should still match a `kk` token start."""
    tokens = ["geung", "geun", "geum", "geul", "geu", "kka"]
    # `eu kk`: token 'geu' (idx 4) has trailing vowel 'eu'; next 'kka' starts with 'kk'.
    assert _match_compound_alias_to_token_transition("eu kk", tokens) == 4
    # `eu k` (single-char onset) should also match via 1-char fallback.
    assert _match_compound_alias_to_token_transition("eu k", tokens) == 4


# ---------------------------------------------------------------------------
# env helper
# ---------------------------------------------------------------------------

def test_compound_token_match_default_off(monkeypatch):
    monkeypatch.delenv("UTOA_BOUNDARY_COMPOUND_TOKEN_SLOT_MATCH_ENABLE", raising=False)
    assert _compound_token_slot_match_enabled() is False


def test_compound_token_match_env_on(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_COMPOUND_TOKEN_SLOT_MATCH_ENABLE", "on")
    assert _compound_token_slot_match_enabled() is True


def test_compound_token_match_env_unknown_value_off(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_COMPOUND_TOKEN_SLOT_MATCH_ENABLE", "nonsense")
    assert _compound_token_slot_match_enabled() is False
