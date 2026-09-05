from fastapi.testclient import TestClient

from src.demo_web import app as demo_app
from src.demo_web.app import build_polish_prompt, page, polish_with_gemini


def test_polish_prompt_only_contains_retrieved_evidence():
    prompt = build_polish_prompt(
        "Điều 2 nói gì?",
        [
            {
                "rank": 1,
                "document_id": "LAW_36_2024",
                "hierarchy": "Điều 2",
                "citation_node_ids": ["LAW_36_2024:ARTICLE:2"],
                "evidence_text": "Nội dung chứng cứ.",
            }
        ],
    )
    assert "không suy diễn" in prompt
    assert "Nội dung chứng cứ." in prompt
    assert "LAW_36_2024:ARTICLE:2" in prompt


def test_polish_prompt_is_limited_to_top_five_results():
    results = [
        {
            "rank": number,
            "document_id": "LAW_36_2024",
            "hierarchy": "Điều 2",
            "citation_node_ids": [f"NODE_{number}"],
            "evidence_text": f"Evidence {number}",
        }
        for number in range(1, 7)
    ]
    prompt = build_polish_prompt("Câu hỏi", results)
    assert "Evidence 5" in prompt
    assert "Evidence 6" not in prompt


def test_page_uses_top_labels_and_gemini_generate_wording():
    html = page()
    assert "Sử dụng Gemini để generate" in html
    assert "Top '+text(item.rank)" in html
    assert "hierarchyText(item.hierarchy)" in html
    assert "Demo B6/B7c" not in html


def test_search_endpoint_returns_canonical_evidence(monkeypatch, tmp_path):
    class Response:
        def model_dump(self, mode):
            return {
                "route": "DIRECT_REFERENCE",
                "resolution_status": "RESOLVED",
                "results": [{
                    "rank": 1,
                    "score": 1.0,
                    "source_strategy": "B7c",
                    "document_id": "LAW_36_2024",
                    "hierarchy": "Điều 2",
                    "result_type": "canonical_node",
                    "citation_node_ids": ["LAW_36_2024:ARTICLE:2"],
                    "evidence_text": "Nội dung chứng cứ.",
                }],
            }

    class Retrieval:
        def __init__(self, *args):
            pass

        def search(self, query, *, top_k):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(demo_app, "PostgresB6Application", Retrieval)
    with TestClient(demo_app.create_app(tmp_path)) as client:
        response = client.post(
            "/api/search",
            json={"query": "Điều 2 nói gì?", "top_k": 3, "polish": False},
        )
    assert response.status_code == 200
    assert response.json()["results"][0]["citation_node_ids"] == ["LAW_36_2024:ARTICLE:2"]


def test_polish_rejects_citations_outside_retrieved_evidence(monkeypatch):
    class Response:
        text = "Câu trả lời [NODE_KHONG_TON_TAI]"

    class Models:
        def generate_content(self, **kwargs):
            return Response()

    class Client:
        models = Models()

    monkeypatch.setattr("google.genai.Client", lambda api_key: Client())
    try:
        polish_with_gemini(
            "Câu hỏi",
            [{"citation_node_ids": ["NODE_HOP_LE"], "rank": 1, "document_id": "LAW", "hierarchy": "Điều 1", "evidence_text": "Text"}],
            "test-key",
            "test-model",
        )
    except RuntimeError as exc:
        assert "citation ngoài evidence" in str(exc)
    else:
        raise AssertionError("Expected a citation guard failure")
