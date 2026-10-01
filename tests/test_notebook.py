import hashlib
import io
import json
import tokenize

from scripts.ce1_common import ROOT

NOTEBOOK = ROOT / "notebooks" / "ce1_pipeline.ipynb"
# SHA-256 of the code of the notebook as it was executed, without comments, trailing spaces
# and blank lines. Only comment tokens are read from the tokenizer, so the value does not
# depend on the Python version.
CODE_HASH = "52e3a4183dbaca707f4333b6ea00091a3554dbd21e49d17922c4fa2a1e1c3fbb"


def code_text(nb):
    out = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        probe = ["pass" if line.lstrip().startswith(("!", "%")) else line for line in lines]
        for tok in tokenize.generate_tokens(io.StringIO("\n".join(probe)).readline):
            if tok.type == tokenize.COMMENT:
                row, col = tok.start
                lines[row - 1] = lines[row - 1][:col]
        out += [line.rstrip() for line in lines if line.strip()]
    return "\n".join(out)


def code_hash(nb):
    return hashlib.sha256(code_text(nb).encode("utf-8")).hexdigest()


def test_code_matches_the_executed_notebook():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert code_hash(nb) == CODE_HASH


def test_outputs_are_cleared():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")


def test_corrected_comments():
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "Mismo criterio de evidencia" not in text
    assert "0.8589" not in text and "0.8356" not in text
    assert "La elección se hizo con test" in text
    assert "Val 0.8613" in text and "Test 0.8379" in text
