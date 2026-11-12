"""Tests for CFG/EBNF conversion to recursive state machines."""

import pytest
from networkx import MultiDiGraph
from pyformlang.cfg import CFG, Production, Terminal, Variable
from pyformlang.finite_automaton import Symbol
from pyformlang.rsa import RecursiveAutomaton

from project.task8 import cfg_to_rsm, ebnf_to_rsm, tensor_based_cfpq


def test_cfg_boxes_recognize_production_bodies():
    grammar = CFG.from_text(
        "Root -> A B | $\nA -> hello\nB -> world", start_symbol="Root"
    )
    before = set(grammar.productions)
    rsm = cfg_to_rsm(grammar)
    assert isinstance(rsm, RecursiveAutomaton)
    assert rsm.initial_label == Symbol("Root")
    assert rsm.get_box("Root").dfa.accepts(["A", "B"])
    assert rsm.get_box("Root").dfa.accepts([])
    assert rsm.get_box("A").dfa.accepts(["hello"])
    assert rsm.get_box("B").dfa.accepts(["world"])
    assert set(grammar.productions) == before


def test_cfg_literal_terminal_is_not_confused_with_box_name():
    start = Variable("S")
    grammar = CFG(start_symbol=start, productions={Production(start, [Terminal("S")])})
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="S")
    assert tensor_based_cfpq(cfg_to_rsm(grammar), graph) == {(10, 20)}


def test_cfg_escapes_regex_metacharacters_by_building_transitions_directly():
    start = Variable("S")
    grammar = CFG(
        start_symbol=start,
        productions={
            Production(start, [Terminal("a|b"), Terminal("*"), Terminal("hello world")])
        },
    )
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a|b")
    graph.add_edge(20, 30, label="*")
    graph.add_edge(30, 40, label="hello world")
    assert tensor_based_cfpq(cfg_to_rsm(grammar), graph) == {(10, 40)}


@pytest.mark.parametrize("grammar", [CFG(), CFG.from_text("S -> A\nA -> A")])
def test_cfg_empty_language(grammar):
    graph = MultiDiGraph()
    graph.add_edge(10, 20, label="a")
    assert tensor_based_cfpq(cfg_to_rsm(grammar), graph) == set()


def test_ebnf_repeated_heads_and_regular_operators():
    rsm = ebnf_to_rsm("S -> a*\nS -> b c")
    assert rsm.get_box("S").dfa.accepts([])
    assert rsm.get_box("S").dfa.accepts(["a", "a"])
    assert rsm.get_box("S").dfa.accepts(["b", "c"])
    assert not rsm.get_box("S").dfa.accepts(["b"])


def test_ebnf_selects_entry_box_without_assuming_first_head_is_s():
    assert ebnf_to_rsm("A -> a\nS -> A").initial_label == Symbol("S")
    assert ebnf_to_rsm("Root -> hello").initial_label == Symbol("Root")


def test_empty_and_invalid_ebnf():
    assert not ebnf_to_rsm(" \n").boxes
    with pytest.raises(ValueError):
        ebnf_to_rsm("not a production")
