"""Audit an explicit text-only inventory, the Git index, or every reachable commit.

This is a fail-closed guardrail, not a substitute for human source review.
It never stages files, rewrites history, publishes, or loads scientific data.
"""

import argparse
import ast
import re
import subprocess
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
PUBLIC_WEIGHT_DIGEST = '974ebbd557c3717d49d4bcc83ba97632e8980eb94ec6a72c3c5f6c307024e7d7'
REQUIRED = {'.gitignore', '.gitattributes', 'LICENSE', 'README.md', 'THIRD_PARTY_NOTICES.md',
            'PUBLIC_FILES.txt', 'demo.py', 'verify_publication.py', 'requirements.txt',
            'requirements-dev.txt', 'requirements-encoder.txt', '.github/workflows/check.yml'}
RULES = {
    'GitHub credential': r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}',
    'AWS access identifier': r'AKIA[0-9A-Z]{16}',
    'private key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'credential in URL': r'https?://[^/\s:@]+:[^/\s@]+@',
    'local Windows user path': r'[A-Za-z]:[\\/]Users[\\/]',
    'private runtime path': r'/(?:home|user|scratch)/[A-Za-z0-9._@-]+/',
    'private shared-data path': r'(?:shared|private)/[A-Za-z0-9._-]+/',
    'internal host': r'https?://[^/\s]+\.(?:local|internal)(?:[/:]|$)',
    'study tissue identifier': r'rcc2025necrosis_[a-f0-9]{8}_\d{4}',
    'embedded image': r'data:image/[a-z]+;base64,',
}


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def allowed_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        return False
    return (name in REQUIRED or
            (len(path.parts) == 2 and path.parts[0] == 'workflow' and path.suffix == '.py') or
            (len(path.parts) == 2 and path.parts[0] == 'tests' and path.name.startswith('test_') and path.suffix == '.py') or
            (len(path.parts) == 2 and path.parts[0] == 'docs' and path.suffix == '.md'))


def scan_text(name, payload):
    problems = []
    if len(payload) > 200_000:
        return [f'{name}: exceeds source-only file-size limit']
    try:
        text = payload.decode('utf-8')
    except UnicodeDecodeError:
        return [f'{name}: not UTF-8 text']
    if '\x00' in text:
        problems.append(f'{name}: binary/null content')
    for label, pattern in RULES.items():
        if re.search(pattern, text):
            problems.append(f'{name}: {label}')
    # The only permitted literal 64-hex digest is a public upstream weight checksum.
    for value in re.findall(r'\b[a-f0-9]{64}\b', text):
        if value != PUBLIC_WEIGHT_DIGEST:
            problems.append(f'{name}: unreviewed embedded digest')
    if name.endswith('.py'):
        try:
            ast.parse(text, filename=name)
        except SyntaxError as exc:
            problems.append(f'{name}: syntax error at line {exc.lineno}')
    return problems


def audit(files, read):
    errors = []
    try:
        inventory = read('PUBLIC_FILES.txt').decode('utf-8').splitlines()
    except (KeyError, FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
        return ['Missing or invalid PUBLIC_FILES.txt']
    if len(set(inventory)) != len(inventory) or any(not n.strip() for n in inventory):
        errors.append('Inventory contains duplicates or empty entries')
    expected, actual = set(inventory), set(files)
    if not REQUIRED <= expected:
        errors.append('Inventory omits required publication files')
    if expected != actual:
        errors.append('Inventory differs from audited files: ' + ', '.join(sorted(expected ^ actual)))
    for name in sorted(actual):
        if not allowed_path(name):
            errors.append(f'{name}: forbidden path/type')
        try:
            errors.extend(scan_text(name, read(name)))
        except (KeyError, FileNotFoundError, subprocess.CalledProcessError):
            errors.append(f'{name}: unreadable')
    return errors


def check_worktree():
    # Include every non-ignored file, not just the claimed allowlist.
    names = git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')
    names = sorted(set(n for n in names if n))
    symlinks = [n for n in names if (ROOT / n).is_symlink()]
    return audit(names, lambda n: (ROOT / n).read_bytes()) + [n + ': symlink' for n in symlinks]


def check_index():
    entries = git('ls-files', '--stage', '-z').decode().strip('\0').split('\0')
    names, errors = [], []
    for entry in filter(None, entries):
        meta, name = entry.split('\t', 1)
        mode, _, stage = meta.split()
        names.append(name)
        if mode != '100644' or stage != '0':
            errors.append(name + ': non-regular file or unresolved index entry')
    return errors + audit(names, lambda n: git('show', ':' + n))


def check_history():
    commits = git('rev-list', '--all').decode().splitlines()
    if not commits:
        return ['No commits to audit']
    errors = []
    for commit in commits:
        entries = git('ls-tree', '-r', '-z', commit).decode().strip('\0').split('\0')
        names = []
        for entry in filter(None, entries):
            meta, name = entry.split('\t', 1)
            mode, kind, _ = meta.split()
            names.append(name)
            if mode != '100644' or kind != 'blob':
                errors.append(commit[:12] + ': non-regular tracked file: ' + name)
        errors += [commit[:12] + ': ' + e for e in audit(names, lambda n: git('show', commit + ':' + n))]
        errors += scan_text('commit metadata', git('show', '-s', '--format=fuller', commit))
    if git('rev-list', '--max-parents=0', '--all').decode().count('\n') != 1:
        errors.append('Expected one fresh history root')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true', help='Audit exact index bytes')
    parser.add_argument('--history', action='store_true', help='Audit every reachable commit and metadata')
    args = parser.parse_args()
    errors = check_worktree()
    if args.staged:
        errors += check_index()
    if args.history:
        errors += check_history()
    if errors:
        for error in errors:
            print('FAIL:', error)
        return 1
    print('PUBLICATION_AUDIT_OK: explicit source-only inventory' +
          (', index' if args.staged else '') + (', complete reachable history' if args.history else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
