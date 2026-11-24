"""Helpers shared by the XML parsers."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pandas as pd

_OFFSET = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def parse_times(values: list[str]) -> pd.Series:
    """Parse ISO 8601 strings. Returns UTC-aware times if every value has an offset,
    naive times if none do. A file that mixes both is rejected: we cannot tell which
    zone the naive ones are in."""
    with_offset = [bool(_OFFSET.search(v.strip())) for v in values]
    if all(with_offset):
        return pd.Series(pd.to_datetime(values, format="ISO8601", utc=True))
    if not any(with_offset):
        return pd.Series(pd.to_datetime(values, format="ISO8601"))
    raise ValueError("file mixes timestamps with and without a UTC offset")


def local(tag: str) -> str:
    """Tag name without its XML namespace."""
    return tag.rsplit("}", 1)[-1]


def iter_local(root: ET.Element, name: str):
    for elem in root.iter():
        if local(elem.tag) == name:
            yield elem


def child(elem: ET.Element, *path: str) -> ET.Element | None:
    """Follow a path of child tag names, ignoring namespaces."""
    node = elem
    for name in path:
        node = next((c for c in node if local(c.tag) == name), None)
        if node is None:
            return None
    return node


def child_float(elem: ET.Element, *path: str) -> float:
    node = child(elem, *path)
    if node is None or node.text is None or not node.text.strip():
        return float("nan")
    return float(node.text)


def descendant_float(elem: ET.Element, name: str) -> float:
    """First descendant with this tag name, at any depth. Extensions nest differently
    across vendors, so searching by name is more robust than a fixed path."""
    node = next(iter_local(elem, name), None)
    if node is None or node.text is None or not node.text.strip():
        return float("nan")
    return float(node.text)
