"""Tests for weak Chomsky normal form conversion."""

from itertools import product

import pytest
from pyformlang.cfg import CFG, Production, Terminal, Variable

from project.task6 import cfg_to_weak_normal_form


def assert_weak_normal_form(grammar):
    for production in grammar.productions:
        body = production.body
        assert (
            not body
            or (len(body) == 1 and isinstance(body[0], Terminal))
            or (len(body) == 2 and all(isinstance(symbol, Variable) for symbol in body))
        )


@pytest.mark.parametrize(
    "text",
    [
        "S -> a S b | $",
        "S -> A B C\nA -> a | $\nB -> b | $\nC -> c | $",
        "S -> A\nA -> B\nB -> A | a | $",
        "S -> a b c d | a",
        "S -> S S | a | $",
        "S -> A\nA -> A",
        "S -> $",
    ],
)
def test_conversion_preserves_short_words_and_empty_word(text):
    grammar = CFG.from_text(text)
    before = set(grammar.productions)
    normalized = cfg_to_weak_normal_form(grammar)
    assert normalized.start_symbol == grammar.start_symbol
    assert_weak_normal_form(normalized)
    for length in range(4):
        for word in product("abcd", repeat=length):
            assert normalized.contains(word) == grammar.contains(word)
    assert set(grammar.productions) == before


def test_fresh_variables_do_not_collide_with_input_variables():
    start = Variable("Root")
    reserved = [Variable("__WCNF_0"), Variable("__WCNF_1")]
    grammar = CFG(
        start_symbol=start,
        productions={
            Production(start, [reserved[0], Terminal("b"), reserved[1], Terminal("d")]),
            Production(reserved[0], [Terminal("a")]),
            Production(reserved[1], [Terminal("c")]),
        },
    )
    normalized = cfg_to_weak_normal_form(grammar)
    assert_weak_normal_form(normalized)
    assert normalized.contains(["a", "b", "c", "d"])
    assert not normalized.contains(["a", "a", "c", "d"])
    assert normalized.start_symbol == start


def test_conversion_preserves_nullable_non_start_variables():
    grammar = CFG.from_text("S -> A B\nA -> a | $\nB -> b | $")
    normalized = cfg_to_weak_normal_form(grammar)
    assert {
        Variable("S"),
        Variable("A"),
        Variable("B"),
    } <= normalized.get_nullable_symbols()
    assert_weak_normal_form(normalized)


def test_empty_grammar_and_weak_form_are_supported():
    for grammar in [CFG(), CFG.from_text("S -> A B | $\nA -> a\nB -> b")]:
        normalized = cfg_to_weak_normal_form(grammar)
        assert_weak_normal_form(normalized)
        assert normalized.start_symbol == grammar.start_symbol
        assert normalized.is_empty() == grammar.is_empty()
