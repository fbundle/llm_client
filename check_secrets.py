"""Scan the repo for accidentally committed secrets.

Checks:
  1. .env / dangerous files in git history
  2. Known API key patterns (AWS, GCP, GitHub, OpenAI, etc.)
  3. Private key blocks (PEM, SSH)
  4. URL-embedded credentials
  5. High-entropy strings as a catch-all heuristic
  6. File-based detection (dangerous filenames)
  7. Gitignore coverage for sensitive patterns

Usage:
  python check_secrets.py          # scan working tree
  python check_secrets.py --full   # scan full git history
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

# Prompt used to generate and extend this file:
PROMPT = """\
Write a Python script check_secrets.py that scans the repo for accidentally
committed confidential information. Include these detection techniques:

1. KNOWN API KEY PATTERNS — prefix-based regexes for common services:
   - Cloud: AWS (AKIA/ASIA), GCP (AIza), Azure, DigitalOcean
   - GitHub: PAT classic (ghp_), fine-grained (github_pat_), OAuth (gho_)
   - AI: OpenAI (sk-), Anthropic (sk-ant-), HuggingFace (hf_)
   - CI/CD: GitLab (glpat-), npm, PyPI (pypi-)
   - Chat: Slack (xoxb-), Discord webhooks, Telegram bot tokens
   - Payments: Stripe (sk_live/sk_test), Twilio (AC/SK)
   - Email: SendGrid (SG.)
   - Generic JWT (eyJ... pattern)

2. GENERIC PATTERNS:
   - Variable assignments with secret-like names (api_key, password, token, etc.)
   - URL-embedded credentials (https://user:pass@host)
   - Private key blocks (-----BEGIN ... PRIVATE KEY-----)

3. HIGH-ENTROPY STRING DETECTION as a catch-all:
   - Only examine quoted string literals (not arbitrary code substrings)
   - Use Shannon entropy with character-class-specific thresholds
   - Skip hex hashes, UUIDs, URLs, natural language
   - Skip binary files, lock files, generated files

4. DANGEROUS FILE DETECTION by filename:
   - Flag tracked files matching .env, *.pem, *.key, credentials.json,
     secrets.yml, id_rsa, *.p12, service-account*.json, etc.

5. GIT HISTORY SCANNING (--full flag):
   - Pipe 'git log -p --all' through all the above checks

6. ALLOWLIST to suppress false positives:
   - Stopword values: changeme, placeholder, example, test, TODO, etc.
   - Path exclusions: binary files, lock files, generated code

7. OUTPUT: human-readable with severity tags (HIGH/MEDIUM/LOW), or --json

To extend this file with more patterns, add entries to _NAMED_PATTERNS,
_DANGEROUS_FILES, or the generic regexes. Use 'git check-ignore' to
verify gitignore coverage instead of string matching.
"""

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Known API key patterns — prefix-based, very low false-positive rate
# ---------------------------------------------------------------------------

_NAMED_PATTERNS: list[tuple[str, str]] = [
    # Cloud
    ("AWS Access Key", r"\b((?:AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16})\b"),
    ("GCP API Key", r"\b(AIza[0-9A-Za-z_-]{35})\b"),
    # GitHub
    ("GitHub PAT (classic)", r"\b(ghp_[0-9a-zA-Z]{36})\b"),
    ("GitHub PAT (fine-grained)", r"\b(github_pat_[0-9a-zA-Z_]{82})\b"),
    ("GitHub OAuth", r"\b(gho_[0-9a-zA-Z]{36})\b"),
    # AI
    ("OpenAI API Key", r"\b(sk-(?:proj-)?[A-Za-z0-9]{20,160})\b"),
    ("Anthropic API Key", r"\b(sk-ant-api03-[a-zA-Z0-9_\-]{93}AA)\b"),
    ("HuggingFace Token", r"\b(hf_[a-z]{34})\b"),
    # CI/CD / packages
    ("GitLab PAT", r"\b(glpat-[0-9a-zA-Z_-]{20,})\b"),
    ("npm Token", r"\b(npm_[a-z0-9]{36})\b"),
    ("PyPI Token", r"\b(pypi-AgEIcHlwaS5vcmc[\w-]{50,})\b"),
    # Comm
    ("Slack Bot Token", r"\b(xoxb-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*)\b"),
    ("Slack Webhook", r"https://hooks\.slack\.com/services/[A-Za-z0-9+/]{43,56}"),
    ("Discord Webhook", r"https://discord\.com/api/webhooks/\d+/[\w-]+"),
    ("Telegram Bot Token", r"\b(\d{5,16}:A[a-zA-Z0-9_-]{34})\b"),
    # Payments
    ("Stripe Secret Key", r"\b(sk_(?:test|live)_[a-zA-Z0-9]{10,99})\b"),
    ("Stripe Publishable Key", r"\b(pk_(?:test|live)_[a-zA-Z0-9]{10,99})\b"),
    # Misc
    ("Twilio Account SID", r"\b(AC[a-z0-9]{32})\b"),
    ("Twilio API Key", r"\b(SK[0-9a-fA-F]{32})\b"),
    ("SendGrid Token", r"\b(SG\.[a-z0-9=_\-.]{66})\b"),
    ("Cloudflare API Key", r"\b([A-Fa-f0-9]{37})\b"),
    ("Heroku API Key", r"\b(HRKU-AA[0-9a-zA-Z_-]{58})\b"),
    # Generic JWT
    ("JWT Token", r"\b(eyJ[a-zA-Z0-9_\-]{10,}\.eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,})\b"),
]

# ---------------------------------------------------------------------------
# Generic secret patterns — higher false-positive rate, filtered by context
# ---------------------------------------------------------------------------

# Variable assignment with a secret-like key name
_GENERIC_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?:api[_\s-]?(?:key|secret|token)|secret[_\s-]?(?:key|token)?|
       access[_\s-]?(?:key|secret|token)|auth[_\s-]?token|
       private[_\s-]?key|password|passwd|credential)
    [\s'"().,\[\]{}]*[:=]\s*['"]([^'"]{8,})['"]
    """
)

# URL with embedded user:password
_URL_CREDS = re.compile(
    r"(?i)https?://[^:\s@]{1,64}:[^:\s@]{1,64}@[^\s'\")\]}>]+"
)

# PEM / private key blocks
_PRIVATE_KEY_BLOCK = re.compile(
    r"(?i)-----BEGIN (?:RSA |EC |DSA |ENCRYPTED |OPENSSH )?PRIVATE KEY(?: BLOCK)?-----"
)

# ---------------------------------------------------------------------------
# Dangerous file patterns that often contain secrets
# ---------------------------------------------------------------------------

_DANGEROUS_FILES = [
    ".env", ".env.*", ".envrc",
    "*.pem", "*.p12", "*.pfx",
    "*.key", "*.keystore", "*.jks", "*.p8",
    "*.credentials.json", "*credential*.json",
    "secrets.yml", "secrets.yaml", "secrets.json",
    "service-account*.json",
    "*.pubxml.user",
    ".netrc", "_netrc",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    "*.asc", "*.gpg", "*.sec",
]

# Secret-key-like words that are safe (false-positive allow-list)
_ALLOWLIST_VALUES = {
    "changeme", "placeholder", "todo", "xxxx", "xxxxxx",
    "your-key-here", "your-token-here", "example", "sample",
    "test", "demo", "fake", "dummy", "replace-me",
}

# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------

def shannon_entropy(data: str) -> float:
    if not data:
        return 0.0
    total = len(data)
    counts = Counter(data)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _char_class(ch: str) -> str:
    if ch in "01":
        return "binary"
    if ch in "0123456789":
        return "digit"
    if ch in "abcdef":
        return "lower_hex"
    if ch in "ABCDEF":
        return "upper_hex"
    if ch.islower():
        return "lower"
    if ch.isupper():
        return "upper"
    if ch in "+/=":
        return "b64_pad"
    if ch in "-_.":
        return "url_safe"
    return "symbol"


# Files excluded from entropy scanning (binary, generated, lock files)
_SKIP_ENTROPY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".svg",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".lock", ".woff", ".woff2", ".ttf", ".eot",
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
    ".mp3", ".mp4", ".mov", ".avi",
    ".min.js", ".min.css",
}
_SKIP_ENTROPY_NAMES = {"uv.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
                       "Cargo.lock", "Gemfile.lock", "poetry.lock"}


def is_likely_secret(token: str) -> bool:
    """Heuristic: high-entropy string inside a quoted literal."""
    if len(token) < 20:
        return False
    if token.lower() in _ALLOWLIST_VALUES:
        return False
    if token.startswith(("http://", "https://", "/", "./", "../", "data:", "file://")):
        return False
    # Looks like a hex hash (all hex chars, 32+ length)
    if re.match(r"^[0-9a-fA-F]{32,}$", token):
        return False
    # Skip strings that look like code or natural language
    if re.search(r"(?i)\b(def|class|import|return|self|pass|None|True|False)\b", token):
        return False

    h = shannon_entropy(token)
    classes = {_char_class(c) for c in token}

    # Natural language (has spaces) — requires much higher entropy
    if " " in token:
        return h >= 5.5

    # Hex with dashes (UUID-like) — skip
    if {"lower_hex", "upper_hex", "digit"} & classes and "-" in classes:
        return False
    # Base64-like (has padding chars)
    if "b64_pad" in classes and h >= 4.2 and len(token) >= 24:
        return True
    # Mixed-case alphanumeric (no spaces) with high entropy
    if {"lower", "upper"} <= classes and h >= 4.4 and len(token) >= 24:
        return True
    # Very long, very high-entropy strings (catch-all)
    if len(token) >= 40 and h >= 5.0:
        return True
    return False


def _should_skip_entropy(fpath: str) -> bool:
    p = Path(fpath)
    if p.suffix in _SKIP_ENTROPY_SUFFIXES:
        return True
    if p.name in _SKIP_ENTROPY_NAMES:
        return True
    return False


def _is_binary(path: Path) -> bool:
    """Quick check: does the file contain null bytes?"""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

class Finding(NamedTuple):
    severity: str  # HIGH / MEDIUM / LOW
    source: str    # file:line or commit hash
    kind: str      # description of what was found
    detail: str    # the matched text (truncated)


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True)


def git_tracked_files() -> list[str]:
    """Return list of all tracked files in the working tree."""
    try:
        return run(["git", "ls-files"]).splitlines()
    except subprocess.CalledProcessError:
        # Not a git repo — scan directory directly
        return [
            str(p.relative_to(ROOT))
            for p in ROOT.rglob("*")
            if p.is_file() and ".git" not in p.parts
        ]


def git_diff_history() -> str:
    """Return full unified diff of all commits, or empty string."""
    try:
        return run(["git", "log", "-p", "--all"])
    except subprocess.CalledProcessError:
        return ""


def check_dangerous_filenames() -> list[Finding]:
    """Check tracked files against dangerous filename patterns."""
    findings: list[Finding] = []
    tracked = git_tracked_files()
    for fpath in tracked:
        fname = Path(fpath).name
        for pattern in _DANGEROUS_FILES:
            if Path(fname).match(pattern):
                findings.append(Finding(
                    "HIGH", fpath,
                    f"Dangerous file: matches '{pattern}'",
                    fname,
                ))
                break
    return findings


def check_named_patterns(text: str, source: str) -> list[Finding]:
    """Scan text for known API key patterns."""
    findings: list[Finding] = []
    for line in text.splitlines():
        for name, pattern_str in _NAMED_PATTERNS:
            for m in re.finditer(pattern_str, line):
                value = m.group(0)
                if value.lower() in _ALLOWLIST_VALUES:
                    continue
                findings.append(Finding(
                    "HIGH", source, f"{name}: {value[:60]}", line.strip()[:200],
                ))
    return findings


def check_generic_patterns(text: str, source: str) -> list[Finding]:
    """Scan for generic secret patterns (assignments, URL creds, private keys)."""
    findings: list[Finding] = []

    for m in _GENERIC_ASSIGNMENT.finditer(text):
        value = m.group(1)
        if value.lower() in _ALLOWLIST_VALUES:
            continue
        findings.append(Finding(
            "MEDIUM", source, f"Generic secret assignment: {m.group(0)[:80]}", "",
        ))

    for m in _URL_CREDS.finditer(text):
        findings.append(Finding(
            "HIGH", source, "URL with embedded credentials", m.group(0),
        ))

    if _PRIVATE_KEY_BLOCK.search(text):
        # Find the line number
        for i, line in enumerate(text.splitlines(), 1):
            if _PRIVATE_KEY_BLOCK.search(line):
                findings.append(Finding(
                    "HIGH", f"{source}:{i}", "Private key block", line.strip(),
                ))
    return findings


def check_entropy(text: str, source: str) -> list[Finding]:
    """Find high-entropy strings inside quoted literals only."""
    findings: list[Finding] = []
    # Only examine quoted string literals — not arbitrary substrings of code
    for m in re.finditer(r"""['"]([^'"]{20,})['"]""", text):
        token = m.group(1)
        if is_likely_secret(token):
            findings.append(Finding(
                "LOW", source, f"High-entropy string ({shannon_entropy(token):.1f} bits)", token[:80],
            ))
    return findings


def check_git_history() -> list[Finding]:
    """Run all checks against the full git history."""
    findings: list[Finding] = []
    diff = git_diff_history()
    if not diff:
        return findings

    # Split by file to get source annotations
    findings += check_named_patterns(diff, "git-history")
    findings += check_generic_patterns(diff, "git-history")
    findings += check_entropy(diff, "git-history")
    return findings


def check_working_tree() -> list[Finding]:
    """Run all checks against the working tree."""
    findings: list[Finding] = []
    tracked = git_tracked_files()

    for fpath in tracked:
        full = ROOT / fpath
        if not full.exists() or full.is_dir():
            continue
        # Skip binary / large files
        if full.stat().st_size > 1_000_000:
            continue
        if _is_binary(full):
            continue
        try:
            text = full.read_text(errors="ignore")
        except Exception:
            continue

        findings += check_named_patterns(text, fpath)
        findings += check_generic_patterns(text, fpath)
        if not _should_skip_entropy(fpath):
            findings += check_entropy(text, fpath)

    findings += check_dangerous_filenames()
    return findings


def check_gitignore() -> list[Finding]:
    """Test that every dangerous file pattern is gitignored."""
    findings: list[Finding] = []
    for pattern in _DANGEROUS_FILES:
        try:
            subprocess.check_call(
                ["git", "check-ignore", "-q", pattern],
                cwd=ROOT, stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            findings.append(Finding(
                "LOW", ".gitignore",
                f"Pattern not gitignored: {pattern}", "",
            ))
    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Scan repo for accidentally committed secrets")
    p.add_argument("--full", action="store_true", help="Scan full git history (slow)")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    args = p.parse_args()

    findings = check_working_tree() + check_gitignore()
    if args.full:
        findings += check_git_history()

    # Deduplicate by (source, kind)
    seen: set[tuple[str, str]] = set()
    unique: list[Finding] = []
    for f in findings:
        key = (f.source, f.kind)
        if key not in seen:
            seen.add(key)
            unique.append(f)

    if args.json:
        print(json.dumps([f._asdict() for f in unique], indent=2))
    else:
        if not unique:
            print("No secrets found.")
            return
        for f in unique:
            tag = {"HIGH": "!!!", "MEDIUM": " ! ", "LOW": " - "}[f.severity]
            print(f"[{tag}] {f.source}: {f.kind}")
            if f.detail:
                print(f"       {f.detail}")

    high = sum(1 for f in unique if f.severity == "HIGH")
    print(f"\n{len(unique)} issue(s) ({high} high). Run with --full to scan git history.")
    if high:
        sys.exit(1)


if __name__ == "__main__":
    main()
