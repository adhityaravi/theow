"""Tests for stats (meow)."""

from io import StringIO
from unittest.mock import MagicMock, patch

from theow._core._stats import meow


def test_meow_empty():
    chroma = MagicMock()
    chroma.get_all_rules_with_stats.return_value = []

    with patch("sys.stdout", new_callable=StringIO):
        meow(chroma)  # Should not raise


def test_meow_with_rules():
    chroma = MagicMock()
    chroma.get_all_rules_with_stats.return_value = [
        {
            "name": "rule1",
            "collection": "default",
            "success_count": 10,
            "fail_count": 2,
            "explored": False,
            "cost": 0.5,
        },
        {
            "name": "rule2",
            "collection": "default",
            "success_count": 5,
            "fail_count": 0,
            "explored": True,
            "cost": 1.0,
        },
    ]

    with patch("sys.stdout", new_callable=StringIO):
        meow(chroma)  # Should not raise


def test_meow_identifies_struggling():
    chroma = MagicMock()
    chroma.get_all_rules_with_stats.return_value = [
        {
            "name": "good_rule",
            "collection": "default",
            "success_count": 10,
            "fail_count": 1,
            "explored": False,
            "cost": 0.1,
        },
        {
            "name": "bad_rule",
            "collection": "default",
            "success_count": 2,
            "fail_count": 10,
            "explored": False,
            "cost": 0.5,
        },
    ]

    with patch("sys.stdout", new_callable=StringIO):
        meow(chroma)  # Should show bad_rule as struggling
