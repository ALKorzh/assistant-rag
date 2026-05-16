import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.services.rag_service import document_matches_hints, hints_from_question


@pytest.fixture(scope="session")
def rag_fixture_text() -> str:
    p = Path(__file__).resolve().parent.parent / "data" / "e2e_chat" / "rag.json"
    return json.loads(p.read_text(encoding="utf-8"))["text"]


def test_hints_from_question_rag_json(rag_fixture_text: str) -> None:
    hints = hints_from_question(rag_fixture_text)
    lowers = [h.lower() for h in hints]
    assert "e2e_upload_verify" in lowers


def test_hints_explicit_filename_with_extension() -> None:
    q = "Проверь в docker_test_doc.txt упоминание alpha."
    out = hints_from_question(q)
    assert any("docker_test_doc.txt".lower() in h.lower() for h in out)


@pytest.mark.parametrize(
    ("filename",),
    (
        ("e2e_upload_verify.txt",),
        ("E2e_Upload_Verify.txt",),
    ),
)
def test_document_matches_hints_by_filename_metadata(filename: str) -> None:
    doc = Document(
        page_content="E2E verify document token",
        metadata={"filename": filename, "source": f"/tmp/{filename}"},
    )
    assert document_matches_hints(doc, ["e2e_upload_verify", "noise"])


def test_document_matches_hints_by_source_only() -> None:
    doc = Document(
        page_content="only source",
        metadata={"source": "/app/data/raw/e2e_upload_verify.txt"},
    )
    assert document_matches_hints(doc, ["e2e_upload_verify"])
