import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from rag_app.app import app
import io

def test_ingest_rejects_non_pdf():
    c = TestClient(app)
    f = io.BytesIO(b"not a pdf")
    r = c.post("/ingest", files={"file": ("x.txt", f, "text/plain")}, data={"doc_id":"x"})
    assert r.status_code == 400
