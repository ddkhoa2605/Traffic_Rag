from __future__ import annotations

import orjson


def _cases(path):
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line]


def test_manual_gold_cases(registry, parsed_documents):
    files = [registry.root / "data/06_gold/law35_gold.jsonl", registry.root / "data/06_gold/law36_gold.jsonl"]
    for case in (item for path in files for item in _cases(path)):
        blocks, nodes = parsed_documents[case["document_id"]]
        expected = case["expected"]
        if "node_id" in expected:
            by_id = {node.id: node for node in nodes}
            node = by_id[expected["node_id"]]
            assert node.type == expected["node_type"], case["case_id"]
            if "page" in expected:
                assert node.source.page_start == expected["page"], case["case_id"]
            if "page_end" in expected:
                assert node.source.page_end == expected["page_end"], case["case_id"]
            if "point" in expected:
                assert node.hierarchy["point"] == expected["point"], case["case_id"]
            if "text_starts_with" in expected:
                assert node.text.startswith(expected["text_starts_with"]), case["case_id"]
            if "text_contains" in expected:
                assert expected["text_contains"] in node.text, case["case_id"]
        else:
            matches = [
                block for block in blocks
                if block.page == case["page"]
                and block.classification == expected["block_classification"]
                and expected["raw_contains"] in block.raw_text
            ]
            assert matches, case["case_id"]
            assert all(block.include_in_legal_text == expected["included"] for block in matches)


def test_gold_suite_has_planned_size(registry):
    assert len(_cases(registry.root / "data/06_gold/law35_gold.jsonl")) == 25
    assert len(_cases(registry.root / "data/06_gold/law36_gold.jsonl")) == 25
