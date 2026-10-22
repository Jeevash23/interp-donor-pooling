"""Validate an original frozen digest or an explicitly recorded path migration."""
import hashlib,json
from pathlib import Path

def frozen_matches(root, relative, expected):
    path=Path(root)/relative
    if not path.is_file():return False
    with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual==expected:return True
    ledger=Path(root)/'data/consolidation-source-relocations.json'
    if not ledger.is_file():return False
    record=json.loads(ledger.read_text())['files'].get(relative,{})
    return record.get('original_sha256')==expected and record.get('relocated_sha256')==actual
