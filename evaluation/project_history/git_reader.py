"""Чтение Git без права что-либо менять.

Только команды чтения: log, show, diff-tree, status, branch, rev-parse,
cat-file. Без оболочки (аргументы списком), hash проверяется регулярным выражением и
`git cat-file -e <hash>^{commit}` до любого show. Вывод ограничен по
размеру: сообщение коммита может быть любым, diff — тем более.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .model import CommitFile, CommitRecord
from .safe_paths import looks_like_hash

# разделители полей и записей в формате log — символы, которых нет в тексте
_FS, _RS = '\x1f', '\x1e'
_LOG_FORMAT = _FS.join(['%H', '%h', '%P', '%an', '%aI', '%s', '%b']) + _RS
_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
DEFAULT_DIFF_BYTES = 200_000


class GitError(RuntimeError):
    """Команда не выполнилась или hash не прошёл проверку."""


class GitReader:
    def __init__(self, root: Path, git_exe: str = 'git', timeout: float = 60.0):
        self.root = Path(root)
        self.git_exe = git_exe
        self.timeout = timeout
        self._available: bool | None = None

    # ——— низкий уровень ———
    def _run(self, *args: str, max_bytes: int | None = None) -> str:
        """Запуск команды чтения; при обрезке возвращает первые max_bytes."""
        try:
            r = subprocess.run(
                [self.git_exe, '-C', str(self.root), *args],
                capture_output=True, timeout=self.timeout,
                creationflags=_NO_WINDOW, shell=False)
        except (OSError, subprocess.SubprocessError) as e:
            raise GitError(f'git {args[0]}: {e}') from e
        if r.returncode != 0:
            err = r.stderr.decode('utf-8', errors='replace').strip()
            raise GitError(f'git {" ".join(args[:2])}: код {r.returncode}: '
                           f'{err[:300]}')
        out = r.stdout
        if max_bytes is not None and len(out) > max_bytes:
            out = out[:max_bytes]
        return out.decode('utf-8', errors='replace')

    def available(self) -> bool:
        if self._available is None:
            try:
                self._run('rev-parse', '--is-inside-work-tree')
                self._available = True
            except GitError:
                self._available = False
        return self._available

    # ——— hash ———
    def commit_exists(self, hash_: str) -> bool:
        if not looks_like_hash(hash_):
            return False
        try:
            self._run('cat-file', '-e', f'{hash_}^{{commit}}')
        except GitError:
            return False
        return True

    def _checked(self, hash_: str) -> str:
        if not self.commit_exists(hash_):
            raise GitError(f'нет такого коммита: {hash_!r}')
        return hash_

    # ——— список коммитов ———
    def log(self, limit: int | None = None, ref: str = 'HEAD') -> list[CommitRecord]:
        args = ['log', f'--format={_LOG_FORMAT}', '--date=iso-strict']
        if limit:
            args.append(f'-n{int(limit)}')
        args.append(ref if ref == 'HEAD' else self._checked(ref))
        out = self._run(*args)
        commits: list[CommitRecord] = []
        for rec in out.split(_RS):
            rec = rec.strip('\r\n')
            if not rec.strip():
                continue
            parts = rec.split(_FS)
            if len(parts) < 7:
                continue
            full, short, parents, author, date, subject, body = parts[:7]
            plist = parents.split()
            commits.append(CommitRecord(
                hash=full.strip(), short=short.strip(), date=date.strip(),
                author=author.strip(), subject=subject.strip(),
                body=body.strip('\r\n'), parents=plist,
                is_merge=len(plist) > 1))
        return commits

    def head(self) -> str:
        return self._run('rev-parse', 'HEAD').strip()

    def branch(self) -> str:
        return self._run('branch', '--show-current').strip()

    def remote_refs(self) -> list[dict]:
        """Локальные копии удалённых веток: имя, hash, дата коммита (без fetch)."""
        out = self._run('for-each-ref',
                        '--format=%(refname:short)\x1f%(objectname:short)\x1f'
                        '%(committerdate:iso-strict)', 'refs/remotes')
        refs = []
        for line in out.splitlines():
            parts = line.split('\x1f')
            if len(parts) == 3 and not parts[0].endswith('/HEAD'):
                refs.append({'name': parts[0], 'short': parts[1], 'date': parts[2]})
        return refs

    # ——— один коммит ———
    def commit_detail(self, hash_: str) -> CommitRecord:
        h = self._checked(hash_)
        recs = self.log(limit=1, ref=h)
        if not recs:
            raise GitError(f'коммит не прочитан: {h}')
        c = recs[0]
        # статус файлов (A/M/D/R…) и построчная статистика — двумя чтениями
        status = self._run('diff-tree', '--no-commit-id', '-r', '-z',
                           '--name-status', '--root', '-M', h)
        numstat = self._run('diff-tree', '--no-commit-id', '-r', '-z',
                            '--numstat', '--root', '-M', h)
        files: dict[str, CommitFile] = {}
        toks = [t for t in status.split('\0')]
        i = 0
        while i < len(toks) - 1:
            st = toks[i]
            if not st:
                i += 1
                continue
            if st[0] in 'RC':               # переименование: старое имя, новое имя
                old, new = toks[i + 1], toks[i + 2] if i + 2 < len(toks) else ''
                files[new] = CommitFile(path=new, status=st[0])
                files[new].exists = (self.root / new).exists()
                i += 3
            else:
                p = toks[i + 1]
                files[p] = CommitFile(path=p, status=st[0])
                files[p].exists = (self.root / p).exists()
                i += 2
        added = deleted = 0
        toks = numstat.split('\0')
        i = 0
        while i < len(toks):
            t = toks[i]
            if not t:
                i += 1
                continue
            parts = t.split('\t')
            if len(parts) < 3:
                i += 1
                continue
            a, d, p = parts[0], parts[1], parts[2]
            if p == '':                     # переименование в -z: путь идёт отдельно
                p = toks[i + 2] if i + 2 < len(toks) else ''
                i += 3
            else:
                i += 1
            f = files.get(p) or CommitFile(path=p, status='M')
            files.setdefault(p, f)
            f.added = int(a) if a.isdigit() else None
            f.deleted = int(d) if d.isdigit() else None
            if f.added is not None:
                added += f.added
            if f.deleted is not None:
                deleted += f.deleted
        c.files = list(files.values())
        c.files_loaded = True
        c.n_files = len(c.files)
        c.added, c.deleted = added, deleted
        c.is_empty = c.n_files == 0
        return c

    def diff_preview(self, hash_: str, path: str | None = None,
                     max_bytes: int = DEFAULT_DIFF_BYTES) -> tuple[str, bool]:
        """Текст diff (без бинарных тел) и признак обрезки."""
        h = self._checked(hash_)
        args = ['show', '--format=', '--no-color', '--stat=0', '-p',
                '--no-ext-diff', '--ignore-submodules', '-M', h]
        if path:
            if path.startswith('-') or '\0' in path:
                raise GitError(f'недопустимый путь: {path!r}')
            args += ['--', path]
        text = self._run(*args, max_bytes=max_bytes + 1)
        truncated = len(text.encode('utf-8')) > max_bytes
        if truncated:
            text = text.encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore')
            text += f'\n\n… обрезано: показаны первые {max_bytes // 1000} КБ diff\n'
        return text, truncated

    # ——— рабочая директория ———
    def working_tree(self) -> dict:
        """Ветка, HEAD, изменённые и новые файлы. Всегда живьём, без кэша."""
        out = self._run('status', '--porcelain=v1', '-z', '--untracked-files=all')
        modified, untracked = [], []
        for entry in out.split('\0'):
            if len(entry) < 4:
                continue
            code, path = entry[:2], entry[3:]
            if code == '??':
                untracked.append(path)
            else:
                modified.append({'code': code.strip(), 'path': path})
        head = self.head()
        return {'branch': self.branch(), 'head': head, 'short': head[:7],
                'modified': modified, 'untracked': untracked,
                'is_clean': not modified and not untracked}
