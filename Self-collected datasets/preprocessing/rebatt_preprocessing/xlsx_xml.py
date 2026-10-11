"""Read-only XLSX XML parsing reused by the two-sheet public workflow.

Original workbook header values are preserved using Unicode string escapes.
"""
from __future__ import annotations

import math
import posixpath
import unicodedata
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
CURVES = ("DisC", "CC", "DisCC2", "DiscOCV", "CcOCV", "Pulse", "CClast")
# Exact original workbook headers: current (A), voltage (V), capacity (Ah).
IVQ = ("\u7535\u6d41(A)", "\u7535\u538b(V)", "\u5bb9\u91cf(Ah)")
RV_HEADERS = ("Point", "Voltage(V)")


def normalized(value: object) -> str:
    return "".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def column_index(reference: str) -> int:
    number = 0
    for char in reference:
        if not char.isalpha():
            break
        number = number * 26 + ord(char.upper()) - 64
    if not number:
        raise ValueError(f"Cell has no column coordinate: {reference!r}")
    return number - 1


def cell_value(cell: ET.Element, shared: list[str]) -> object:
    kind = cell.get("t", "n")
    if kind == "inlineStr":
        return "".join(part.text or "" for part in cell.iter(NS + "t"))
    value = cell.find(NS + "v")
    if value is None or value.text is None:
        return None
    if kind == "s":
        return shared[int(value.text)]
    if kind in ("n", ""):
        return float(value.text)
    if kind == "b":
        return bool(int(value.text))
    return value.text


def worksheet_paths(archive: ZipFile) -> dict[str, str]:
    relationships = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    }
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    result = {}
    for sheet in workbook.find(NS + "sheets"):
        target = relationships[sheet.attrib[RID]]
        result[sheet.attrib["name"]] = (
            target.lstrip("/")
            if target.startswith("/")
            else posixpath.normpath("xl/" + target)
        )
    return result


def read_sheet(
    archive: ZipFile, xml_path: str, name: str, shared: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    expected = IVQ if name in CURVES else RV_HEADERS if name == "RV" else None
    headers = list(expected or ())
    indices = None
    header_row = None
    records: list[list[float]] = []
    excel_rows: list[int] = []
    nonnumeric: list[dict] = []
    formula_cells = 0
    previous_xml_row = 0
    with archive.open(xml_path) as stream:
        for _, row in ET.iterparse(stream, events=("end",)):
            if row.tag != NS + "row":
                continue
            row_number = int(row.attrib["r"])
            if row_number <= previous_xml_row:
                raise ValueError(f"{name}: non-increasing source Excel rows")
            previous_xml_row = row_number
            cells = {}
            for cell in row:
                if cell.tag != NS + "c":
                    continue
                reference = cell.attrib["r"]
                index = column_index(reference)
                suffix = reference.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
                if int(suffix) != row_number or index in cells:
                    raise ValueError(f"{name}: inconsistent/duplicate cell {reference}")
                cells[index] = cell_value(cell, shared)
                formula_cells += cell.find(NS + "f") is not None

            if indices is None:
                present = {index: value for index, value in cells.items() if value is not None}
                if expected:
                    matches = {}
                    wanted = {normalized(value) for value in expected}
                    for index, value in present.items():
                        key = normalized(value)
                        if key in wanted:
                            if key in matches:
                                raise ValueError(f"{name}: duplicate required header {value}")
                            matches[key] = index
                    if wanted == set(matches):
                        indices = [matches[normalized(value)] for value in expected]
                        headers = [str(cells[index]) for index in indices]
                        header_row = row_number
                elif present:
                    width = max(present) + 1
                    indices = list(range(width))
                    headers = [str(cells.get(index) or "") for index in indices]
                    header_row = row_number
                row.clear()
                continue

            last_row = excel_rows[-1] if excel_rows else header_row
            for missing_row in range(last_row + 1, row_number):
                records.append([math.nan] * len(indices))
                excel_rows.append(missing_row)
            values = []
            for index in indices:
                value = cells.get(index)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(float(value))
                else:
                    values.append(math.nan)
                    if value is not None and value != "":
                        nonnumeric.append({"row": row_number, "column": index + 1, "value": str(value)})
            records.append(values)
            excel_rows.append(row_number)
            row.clear()

    if indices is None and previous_xml_row:
        raise ValueError(f"{name}: no required header found in nonempty worksheet")
    width = len(headers)
    data = np.asarray(records, dtype=np.float64).reshape((-1, width)) if width else np.empty((0, 0), dtype=np.float64)
    rows = np.asarray(excel_rows, dtype=np.int64)
    extra = {
        "header_excel_row": header_row,
        "source_excel_columns": [index + 1 for index in indices or []],
        "blank_rows": int(np.isnan(data).all(axis=1).sum()) if data.shape[1] else 0,
        "nan_values": int(np.isnan(data).sum()),
        "formula_cells_using_cached_values": formula_cells,
        "nonnumeric_cells": nonnumeric,
    }
    return data, rows, headers, extra
