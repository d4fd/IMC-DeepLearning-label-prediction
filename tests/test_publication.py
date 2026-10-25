import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import verify_publication as audit


@pytest.mark.parametrize('name', ['data.csv', 'workflow/data.parquet', 'workflow/model.pt',
                                  'report.ipynb', 'outputs/predictions.csv',
                                  'MULTITASK_LEARNING_PLAN.md', '../escape.py', 'data.zip'])
def test_rejects_private_artifact_paths(name):
    assert not audit.allowed_path(name)


def test_detects_secret_and_patient_pattern_without_real_records():
    fake_secret = 'ghp_' + 'A' * 36
    fake_id = 'rcc2025necrosis_' + 'deadbeef_' + '0000'
    assert audit.scan_text('toy.py', ('x = ' + repr(fake_secret)).encode())
    assert audit.scan_text('toy.py', ('x = ' + repr(fake_id)).encode())
    assert audit.scan_text('toy.py', b'\x00')
    assert audit.scan_text('toy.py', ('x = ' + repr('a' * 64)).encode())


def test_detects_private_paths_without_naming_real_infrastructure():
    assert audit.scan_text('toy.md', b'Location: /home/' + b'synthetic-user/dataset')
    assert audit.scan_text('toy.md', b'Location: shared/' + b'synthetic-input/images')
    assert audit.scan_text('toy.md', b'https://' + b'synthetic.internal/path')


def test_inventory_cannot_hide_extra_file():
    files = {n: b'' for n in audit.REQUIRED}
    files['PUBLIC_FILES.txt'] = ('\n'.join(sorted(files)) + '\n').encode()
    files['unexpected.csv'] = b'purely synthetic'
    errors = audit.audit(files, files.__getitem__)
    assert any('Inventory differs' in e for e in errors)
    assert any('forbidden path/type' in e for e in errors)


def init_toy_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    audit.git('init', '--initial-branch=main')
    audit.git('config', 'user.name', 'Synthetic Test')
    audit.git('config', 'user.email', 'synthetic@example.invalid')
    for name in audit.REQUIRED:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('')
    (tmp_path / 'PUBLIC_FILES.txt').write_text('\n'.join(sorted(audit.REQUIRED)) + '\n')
    audit.git('add', '--all')
    audit.git('commit', '-m', 'Synthetic safe initial commit')


def test_index_bytes_audited_not_just_worktree(tmp_path, monkeypatch):
    init_toy_repo(tmp_path, monkeypatch)
    file = tmp_path / 'demo.py'
    file.write_text('x = ' + repr('ghp_' + 'A' * 36))
    audit.git('add', 'demo.py')
    file.write_text('')
    assert audit.check_worktree() == []
    assert any('credential' in e for e in audit.check_index())


def test_deleted_file_in_old_commit_still_rejected(tmp_path, monkeypatch):
    init_toy_repo(tmp_path, monkeypatch)
    file = tmp_path / 'forbidden.csv'
    file.write_text('not real data')
    audit.git('add', 'forbidden.csv')
    audit.git('commit', '-m', 'Synthetic forbidden-file fixture')
    file.unlink()
    audit.git('add', '--all')
    audit.git('commit', '-m', 'Remove synthetic fixture')
    assert audit.check_index() == []
    assert any('forbidden path/type' in e for e in audit.check_history())


def test_launcher_list_and_no_data_preflight(tmp_path):
    import os
    env = dict(os.environ, PROJECT_WORK_DIR=str(tmp_path / 'empty'))
    launcher = ROOT / 'workflow/run_stage.py'
    listing = subprocess.run([sys.executable, str(launcher), 'list'], capture_output=True, text=True, env=env)
    assert listing.returncode == 0 and 'final-train' in listing.stdout
    check = subprocess.run([sys.executable, str(launcher), 'final-train', '--check'],
                           capture_output=True, text=True, env=env)
    assert check.returncode == 2 and 'Missing external prerequisites' in check.stderr
    assert not (tmp_path / 'empty').exists()
