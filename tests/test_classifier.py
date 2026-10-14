"""Offline tests: the Decisions API is mocked with httpx.MockTransport, so no API key is needed."""
import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main
from app.classifier import classify_dataframe


def fake_client(label="positive"):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/decisions"
        q = request.read()
        import json

        body = json.loads(q)
        assert body["questions"][0]["type"] == "choice"
        assert label in [c["value"] for c in body["questions"][0]["choices"]]
        return httpx.Response(200, json={"answers": [{"type": "choice", "name": "classification", "choice": label, "confidence": 0.93}]})

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1")


def test_adds_columns():
    df = pd.read_csv("sample_data/reviews.csv")
    out = classify_dataframe(df, ["positive", "negative"], "sentiment?", "sentiment", client=fake_client())
    assert list(out["sentiment"]) == ["positive"] * len(df)
    assert list(out["sentiment_confidence"]) == [0.93] * len(df)
    assert "sentiment" not in df.columns


def test_http_error_becomes_error_cell():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400, text="bad")), base_url="https://x/v1")
    out = classify_dataframe(pd.DataFrame({"a": ["x"]}), ["p", "q"], "i", client=client)
    assert out["classification"][0].startswith("ERROR")


def test_existing_column_rejected():
    df = pd.DataFrame({"a": ["x"], "label": ["y"]})
    with pytest.raises(ValueError):
        classify_dataframe(df, ["a", "b"], "i", "label", client=fake_client("a"))


def test_api_endpoint(monkeypatch):
    monkeypatch.setattr("app.classifier.get_client", lambda: fake_client("billing"))
    c = TestClient(main.app)
    with open("sample_data/support_tickets.csv", "rb") as f:
        r = c.post("/classify", files={"file": ("t.csv", f, "text/csv")}, data={"preset": "support_tickets"})
    assert r.status_code == 200
    lines = r.text.splitlines()
    assert lines[0].endswith(",category,category_confidence")
    assert lines[1].endswith(",billing,0.93")
