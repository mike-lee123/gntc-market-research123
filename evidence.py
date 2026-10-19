"""Evidence helpers: source dates, excerpts, report citations and 待驗證 marking."""
import re
from datetime import date
from urllib.parse import unquote

UNVERIFIED = '〔待驗證〕'
PROFILE_MARK = '〔公司背景〕'
EXCERPT_CHARS = 500
DATE_SCAN_CHARS = 8000

_FIGURE = re.compile(r'\d[\d,.]*\s*(?:%|％|個百分點|百分點|倍|兆|億|萬|美元|元|家|名|人次)')
_CITATION = re.compile(r'\]\(https?://|<https?://|\[\d+(?:\s*[,，]\s*\d+)*\]|〔公司背景〕')
_LINK = re.compile(r'\]\((https?://[^)\s]+)\)|<(https?://[^>\s]+)>')
_SENTENCE_END = re.compile(r'(?<=[。！？；!?])')
_WESTERN_DATE = re.compile(r'(?<!\d)(20\d\d)\s*[年/\-.]\s*(\d{1,2})(?:\s*[月/\-.]\s*(\d{1,2}))?')
_ROC_DATE = re.compile(r'(?<!\d)(?:民國\s*)?(1[01]\d)\s*年\s*(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日)?')
_REFERENCE_HEADING = re.compile(r'^#{1,6}\s*(參考資料|資料來源|References)')


def normalize_url(url):
    return unquote(url.strip()).rstrip('/').lower()


def _valid(year, month, day, today):
    try:
        found = date(year, month, day or 1)
    except ValueError:
        return None
    return found if found <= today else None


def estimate_date(url, text, today=None):
    """Latest plausible date in the URL or page text, as YYYY-MM[-DD]; None when nothing is found."""
    today = today or date.today()
    candidates = []
    for source in (unquote(url or ''), (text or '')[:DATE_SCAN_CHARS]):
        for y, m, d in _WESTERN_DATE.findall(source):
            found = _valid(int(y), int(m), int(d) if d else None, today)
            if found:
                candidates.append((found, bool(d)))
        for y, m, d in _ROC_DATE.findall(source):
            found = _valid(int(y) + 1911, int(m), int(d) if d else None, today)
            if found:
                candidates.append((found, bool(d)))
    if not candidates:
        return None
    found, has_day = max(candidates)
    return found.isoformat() if has_day else found.strftime('%Y-%m')


def excerpt(text, limit=EXCERPT_CHARS):
    text = re.sub(r'\s+', ' ', text or '').strip()
    return text if len(text) <= limit else text[:limit].rstrip() + '…'


def citing_passages(report, url, limit=3):
    """Report lines that link to the URL, without Markdown list/heading markers."""
    target = normalize_url(url)
    passages = []
    for line in (report or '').splitlines():
        links = [a or b for a, b in _LINK.findall(line)]
        if any(normalize_url(link) == target for link in links):
            passages.append(re.sub(r'^\s*(?:[-*+]|\d+\.|#+|>)\s*', '', line).strip())
            if len(passages) == limit:
                break
    return passages


def enrich_sources(sources, report, today=None):
    """Add estimated date, excerpt and report citations; drops the raw page text."""
    enriched = []
    for source in sources:
        raw = source.get('raw_content') or ''
        passages = citing_passages(report, source['url'])
        enriched.append({'url': source['url'], 'title': source.get('title') or source['url'],
                         'date': estimate_date(source['url'], raw, today), 'excerpt': excerpt(raw),
                         'cited': bool(passages), 'citations': passages})
    return enriched


def _mark_sentences(line):
    count = 0
    parts = []
    for sentence in _SENTENCE_END.split(line):
        if sentence and _FIGURE.search(sentence) and UNVERIFIED not in sentence:
            stripped = sentence.rstrip()
            sentence = stripped + UNVERIFIED + sentence[len(stripped):]
            count += 1
        parts.append(sentence)
    return ''.join(parts), count


def _mark_table_row(line):
    body = line.rstrip()
    if not body.endswith('|') or re.fullmatch(r'[\s|:\-]+', body):
        return line, 0
    return body[:-1].rstrip() + ' ' + UNVERIFIED + ' |', 1


def mark_unverified(report, skip_sections=()):
    """Mark figures in lines without any citation as 待驗證. Returns (report, count).

    skip_sections: titles of `## ` sections whose figures come from a trusted source (e.g. the official registry).
    """
    lines, total, in_code, in_references, in_skipped = [], 0, False, False, False
    for line in (report or '').splitlines():
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code = not in_code
        elif stripped.startswith('#'):
            in_references = bool(_REFERENCE_HEADING.match(stripped))
            if re.match(r'^##\s', stripped):
                in_skipped = any(title in stripped for title in skip_sections)
        elif in_skipped:
            pass
        elif not (in_code or in_references) and _FIGURE.search(line) and not _CITATION.search(line):
            if stripped.startswith('|'):
                if UNVERIFIED not in line:
                    line, count = _mark_table_row(line)
                    total += count
            else:
                line, count = _mark_sentences(line)
                total += count
        lines.append(line)
    total += sum(line.count(UNVERIFIED) for line in (report or '').splitlines())  # model-marked items
    return '\n'.join(lines), total
