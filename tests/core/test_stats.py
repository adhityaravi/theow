"""Tests for stats (meow)."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

from rich.table import Table

from theow._core._stats import meow


def _capture_meow(rules: list[dict]) -> tuple[list[str], list[Table]]:
    """Run meow and capture printed strings and Table objects."""
    chroma = MagicMock()
    chroma.get_all_rules_with_stats.return_value = rules

    strings: list[str] = []
    tables: list[Table] = []

    with patch("theow._core._stats.Console") as mock_console_cls:
        mock_console = MagicMock()

        def _capture(x, **kw):
            if isinstance(x, Table):
                tables.append(x)
            else:
                strings.append(str(x))

        mock_console.print.side_effect = _capture
        mock_console_cls.return_value = mock_console
        meow(chroma)

    return strings, tables


def _table_rows(table: Table) -> list[list[str]]:
    """Extract rows from a Rich Table as list of [col0, col1, ...]."""
    n_rows = len(table.columns[0]._cells) if table.columns else 0
    rows = []
    for i in range(n_rows):
        row = [str(col._cells[i]) for col in table.columns]
        rows.append(row)
    return rows


def test_meow_empty():
    chroma = MagicMock()
    chroma.get_all_rules_with_stats.return_value = []

    with patch("sys.stdout", new_callable=StringIO):
        meow(chroma)  # Should not raise


def test_meow_with_rules():
    strings, tables = _capture_meow(
        [
            {
                "name": "rule1",
                "collection": "default",
                "success_count": 10,
                "fail_count": 2,
                "explored": False,
            },
            {
                "name": "rule2",
                "collection": "default",
                "success_count": 5,
                "fail_count": 0,
                "explored": True,
            },
        ]
    )

    header = strings[0]
    assert "15" in header  # total resolves
    assert "1" in header  # 1 explore

    assert len(tables) == 1
    rows = _table_rows(tables[0])
    rule_names = [r[0] for r in rows]
    assert "rule1" in rule_names
    assert "rule2" in rule_names


def test_meow_shows_hit_fix_rate():
    strings, tables = _capture_meow(
        [
            {
                "name": "license_llm_fallback",
                "collection": "default",
                "success_count": 4,
                "fail_count": 7,
                "explored": False,
            },
            {
                "name": "license_fallback_pkggodev",
                "collection": "default",
                "success_count": 0,
                "fail_count": 11,
                "explored": False,
            },
            {
                "name": "go_module_path_mismatch",
                "collection": "default",
                "success_count": 1,
                "fail_count": 1,
                "explored": False,
            },
        ]
    )

    assert len(tables) == 1
    rows = _table_rows(tables[0])

    # Sorted by hits descending: llm_fallback(11), pkggodev(11), mismatch(2)
    assert rows[0][0] == "license_llm_fallback"
    assert rows[0][1] == "11"  # hit
    assert rows[0][2] == "4"  # fix
    assert "36%" in rows[0][3]  # rate

    assert rows[1][0] == "license_fallback_pkggodev"
    assert rows[1][1] == "11"
    assert rows[1][2] == "0"
    assert "0%" in rows[1][3]

    assert rows[2][0] == "go_module_path_mismatch"
    assert rows[2][1] == "2"
    assert rows[2][2] == "1"
    assert "50%" in rows[2][3]


def test_meow_excludes_zero_hit_rules():
    strings, tables = _capture_meow(
        [
            {
                "name": "active_rule",
                "collection": "default",
                "success_count": 3,
                "fail_count": 1,
                "explored": False,
            },
            {
                "name": "inactive_rule",
                "collection": "default",
                "success_count": 0,
                "fail_count": 0,
                "explored": False,
            },
        ]
    )

    assert len(tables) == 1
    rows = _table_rows(tables[0])
    rule_names = [r[0] for r in rows]
    assert "active_rule" in rule_names
    assert "inactive_rule" not in rule_names
