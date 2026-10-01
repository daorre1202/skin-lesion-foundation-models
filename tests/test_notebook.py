import hashlib
import io
import json
import tokenize

from scripts.ce1_common import ROOT

NOTEBOOK = ROOT / "notebooks" / "ce1_pipeline.ipynb"
# SHA-256 of the code tokens (comments and layout ignored) of the notebook as it was executed.
CODE_HASH = "bafa42bc09acc427"


def code_hash(nb):
    h = hashlib.sha256()
    skip = (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        code = "\n".join("pass" if line.lstrip().startswith(("!", "%")) else line for line in src.splitlines())
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type not in skip:
                h.update(tok.string.encode())
                h.update(b"\x00")
    return h.hexdigest()


def test_code_tokens_match_the_executed_notebook():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert code_hash(nb).startswith(CODE_HASH)


def test_outputs_are_cleared():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")


def test_corrected_comments():
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "Mismo criterio de evidencia" not in text
    assert "0.8589" not in text and "0.8356" not in text
    assert "La elección se hizo con test" in text
    assert "Val 0.8613" in text and "Test 0.8379" in text
