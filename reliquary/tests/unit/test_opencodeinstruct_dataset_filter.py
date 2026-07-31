"""Tests for the OpenCodeInstruct subset filter pipeline.

The filter functions are pure (no HF / no network), tested directly.
The push-to-Hub side is exercised manually when running the script.
"""

import json
import subprocess

import pytest


def test_keep_row_filters_low_test_score():
    from scripts.build_opencodeinstruct_subset import keep_row
    row = {"average_test_score": 0.9, "unit_tests": "[]", "input": "p", "output": "x"}
    assert keep_row(row) is False


def test_keep_row_accepts_perfect_score():
    from scripts.build_opencodeinstruct_subset import keep_row
    row = {
        "average_test_score": 1.0,
        "unit_tests": '["assert f(1) == 1"]',
        "input": "p", "output": "x",
    }
    assert keep_row(row) is True


def test_parse_unit_tests_handles_string_list():
    from scripts.build_opencodeinstruct_subset import parse_unit_tests
    raw = '["assert f(1) == 1", "assert f(2) == 2"]'
    assert parse_unit_tests(raw) == ["assert f(1) == 1", "assert f(2) == 2"]


def test_parse_unit_tests_returns_none_on_garbage():
    from scripts.build_opencodeinstruct_subset import parse_unit_tests
    assert parse_unit_tests("not json") is None
    assert parse_unit_tests("[unterminated") is None


def test_has_nondeterministic_pattern_detects_random():
    from scripts.build_opencodeinstruct_subset import has_nondeterministic_pattern
    assert has_nondeterministic_pattern("import random\nassert random.random() > 0") is True
    assert has_nondeterministic_pattern("import time; assert time.time() > 0") is True
    assert has_nondeterministic_pattern("import socket") is True
    assert has_nondeterministic_pattern("import urllib.request") is True
    assert has_nondeterministic_pattern("import requests") is True
    assert has_nondeterministic_pattern("import subprocess") is True
    assert has_nondeterministic_pattern("import threading") is True


def test_has_nondeterministic_pattern_clean_code():
    from scripts.build_opencodeinstruct_subset import has_nondeterministic_pattern
    assert has_nondeterministic_pattern("assert sum([1,2,3]) == 6") is False
    assert has_nondeterministic_pattern("assert sorted([3,1,2]) == [1,2,3]") is False


def test_filter_tests_drops_nondeterministic():
    from scripts.build_opencodeinstruct_subset import filter_tests
    tests = ["assert f(1) == 1", "import random; assert random.random() > 0"]
    kept = filter_tests(tests)
    assert kept == ["assert f(1) == 1"]


def test_structure_test_function_equality():
    from scripts.build_opencodeinstruct_subset import structure_test

    assert structure_test("assert add(1, 2) == 3") == {
        "entry": {"kind": "function", "name": "add"},
        "args": [1, 2],
        "kwargs": {},
        "expected": 3,
        "compare": "exact",
    }


def test_structure_test_reversed_equality():
    from scripts.build_opencodeinstruct_subset import structure_test

    assert structure_test("assert [1, 2] == f('x')") == {
        "entry": {"kind": "function", "name": "f"},
        "args": ["x"],
        "kwargs": {},
        "expected": [1, 2],
        "compare": "exact",
    }


def test_structure_test_truthy_and_falsy():
    from scripts.build_opencodeinstruct_subset import structure_test

    assert structure_test("assert is_even(2)") == {
        "entry": {"kind": "function", "name": "is_even"},
        "args": [2],
        "kwargs": {},
        "expected": True,
        "compare": "exact",
    }
    assert structure_test("assert not is_even(3)") == {
        "entry": {"kind": "function", "name": "is_even"},
        "args": [3],
        "kwargs": {},
        "expected": False,
        "compare": "exact",
    }


def test_structure_test_solution_method():
    from scripts.build_opencodeinstruct_subset import structure_test

    assert structure_test("assert Solution().twoSum([2,7,11,15], 9) == [0, 1]") == {
        "entry": {"kind": "method", "class_name": "Solution", "method": "twoSum"},
        "args": [[2, 7, 11, 15], 9],
        "kwargs": {},
        "expected": [0, 1],
        "compare": "exact",
    }


def test_structure_test_rejects_arbitrary_or_unsafe_asserts():
    from scripts.build_opencodeinstruct_subset import structure_test

    assert structure_test("import os\nassert f(1) == 1") is None
    assert structure_test("assert f(g(1)) == 1") is None
    assert structure_test("assert f(1) < 2") is None
    assert structure_test("assert f(1) == object()") is None
    assert structure_test("assert inspect.stack()") is None


def test_process_row_outputs_structured_cases_only(monkeypatch):
    from scripts import build_opencodeinstruct_subset as subset

    monkeypatch.setattr(subset, "double_execute", lambda code, cases: True)
    row = {
        "average_test_score": 1.0,
        "unit_tests": '["assert add(1, 2) == 3"]',
        "input": "Write add",
        "output": "def add(a,b): return a+b",
        "id": "row1",
    }
    out = subset.process_row(row)
    assert out is not None
    assert "structured_cases" in out
    assert "unit_tests_parsed" not in out
    assert "output" not in out
    assert json.loads(out["structured_cases"])[0]["expected"] == 3


def test_process_row_extracts_fenced_reference_solution_without_publishing_it():
    from scripts import build_opencodeinstruct_subset as subset

    row = {
        "average_test_score": 1.0,
        "unit_tests": '["assert add(1, 2) == 3"]',
        "input": "Implement add",
        "output": "```python\ndef add(a, b):\n    return a + b\n```",
    }
    out = subset.process_row(row)
    assert out is not None
    assert "output" not in out
    assert json.loads(out["structured_cases"])[0]["expected"] == 3


def test_process_row_can_include_reference_solution_for_lab_artifacts():
    from scripts import build_opencodeinstruct_subset as subset

    row = {
        "average_test_score": 1.0,
        "unit_tests": '["assert add(1, 2) == 3"]',
        "input": "Implement add",
        "output": "```python\ndef add(a, b):\n    return a + b\n```",
    }
    out = subset.process_row(row, include_reference_output=True)
    assert out is not None
    assert out["output"].startswith("def add")
    assert json.loads(out["structured_cases"])[0]["expected"] == 3


def test_prompt_only_rows_preserve_order_and_hide_cases():
    from scripts import build_opencodeinstruct_subset as subset

    rows = [
        {"input": "Prompt A", "id": "a", "structured_cases": "[secret]"},
        {"input": "Prompt B", "id": "b", "structured_cases": "[secret]", "output": "ref"},
    ]

    assert subset.prompt_only_rows(rows) == [
        {"input": "Prompt A", "id": "a"},
        {"input": "Prompt B", "id": "b"},
    ]


def test_double_execute_timeout_drops_row(monkeypatch):
    from scripts import build_opencodeinstruct_subset as subset

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=1)

    monkeypatch.setattr(subprocess, "run", _timeout)

    assert subset.double_execute("def add(a, b): return a + b", [
        {
            "entry": {"kind": "function", "name": "add"},
            "args": [1, 2],
            "kwargs": {},
            "expected": 3,
            "compare": "exact",
        }
    ]) is False
