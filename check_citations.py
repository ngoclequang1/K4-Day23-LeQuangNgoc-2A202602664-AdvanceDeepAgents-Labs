"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_CITATION = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")
_REF_LINE = re.compile(r"^\s*\[(\d+)\]\s+(.+)$")
_URL = re.compile(r"https?://[^\s<>()]+")


def _numbers(group):
    result = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            start, end = map(int, span.groups())
            if end < start or end - start > 200:
                result.extend((start, end))
            else:
                result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result


def _citations(body):
    cited = set()
    for index, segment in enumerate(_CODE.split(body)):
        if index % 2:  # fenced and inline code
            continue
        for match in _CITATION.finditer(segment):
            cited.update(_numbers(match.group(1)))
    return cited


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    if not isinstance(sources, list):
        return ["sources.json must be a JSON list"]
    if not sources:
        return ["no sources in sources.json"]

    problems = []
    by_number = {}
    seen_urls = {}
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            problems.append(f"source entry {index + 1} is not an object")
            continue
        number = source.get("n")
        url = source.get("url")
        if not isinstance(number, int) or isinstance(number, bool):
            problems.append(f"source entry {index + 1} has non-integer n={number!r}")
        elif number in by_number:
            problems.append(f"source number [{number}] appears more than once")
        else:
            by_number[number] = source
        if not isinstance(url, str) or not re.match(r"^https?://", url):
            problems.append(f"source [{number}] has invalid url")
        elif url in seen_urls:
            problems.append(f"duplicate url in sources [{seen_urls[url]}] and [{number}]")
        else:
            seen_urls[url] = number

    headings = list(_REF_HEADING.finditer(report_text))
    if not headings:
        problems.append("missing ## References heading")
        body, references = report_text, ""
    else:
        if len(headings) > 1:
            problems.append("more than one ## References heading")
        heading = headings[-1]
        body, references = report_text[:heading.start()], report_text[heading.end():]

    cited = _citations(body)
    for number in sorted(cited - set(by_number)):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(set(by_number) - cited):
        problems.append(f"source [{number}] never cited")

    reference_lines = {}
    for line in references.splitlines():
        match = _REF_LINE.match(line)
        if not match:
            continue
        number = int(match.group(1))
        reference_lines.setdefault(number, []).append(match.group(2))

    for number in sorted(reference_lines):
        if number not in by_number:
            problems.append(f"reference [{number}] missing from sources.json")
        if len(reference_lines[number]) > 1:
            problems.append(f"reference [{number}] appears more than once")
    for number, source in sorted(by_number.items()):
        lines = reference_lines.get(number, [])
        if not lines:
            problems.append(f"source [{number}] has no reference line")
            continue
        if len(lines) != 1:
            continue
        urls = _URL.findall(lines[0])
        if len(urls) != 1:
            problems.append(f"reference [{number}] must contain exactly one URL")
        elif urls[0].rstrip(".,;") != source.get("url"):
            problems.append(f"reference [{number}] URL does not match sources.json")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
