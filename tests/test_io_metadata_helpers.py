from __future__ import annotations

from timelapsedhrpqct.io.metadata import dicttolog, logtodict, parse_processing_log


def test_logtodict_parses_scalars_and_lists() -> None:
    log = """! comment\nMu_Scaling  1000\nOrig-ISQ-Dim-um  615  615  615\nHU: mu water  0.2409\nTextKey  abc\n"""
    d = logtodict(log)
    assert d["Mu_Scaling"] == 1000
    assert d["Orig-ISQ-Dim-um"] == [615, 615, 615]
    assert d["HU"] == "mu water  0.2409" or d.get("HU: mu water") == 0.2409


def test_dicttolog_roundtrip_contains_keys() -> None:
    d = {
        "Mu_Scaling": 1000,
        "Orig-ISQ-Dim-um": [615, 615, 615],
        "Density: slope": 1603.51904,
        "TextKey": "abc",
    }
    log = dicttolog(d)
    reparsed = logtodict(log)

    assert "Mu_Scaling" in reparsed
    assert "Orig-ISQ-Dim-um" in reparsed
    assert parse_processing_log(log) == reparsed
