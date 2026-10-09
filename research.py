"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator
MAX_REPAIR_ROUNDS = 2


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    value = re.sub(r"[^\w]+", "-", str(topic).strip().lower(), flags=re.UNICODE).strip("-_")
    value = value[:60].rstrip("-_")
    return value or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f'Produce a rigorous English survey report about: "{topic.strip()}". '
        "Follow the complete required workflow in your system instructions: plan first, delegate at least three "
        "independent researcher tasks (in parallel where possible), use at least three source families in the final "
        "cited report, finalize citations, validate until OK, and spot-check representative claims."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_counts = Counter()
    input_tokens = 0
    output_tokens = 0
    for message in messages or []:
        if isinstance(message, dict):
            calls = message.get("tool_calls") or message.get("additional_kwargs", {}).get("tool_calls") or []
            usage = message.get("usage_metadata") or {}
        else:
            calls = getattr(message, "tool_calls", None) or (
                getattr(message, "additional_kwargs", {}) or {}
            ).get("tool_calls", [])
            usage = getattr(message, "usage_metadata", None) or {}
        for call in calls:
            if isinstance(call, dict):
                name = call.get("name") or (call.get("function") or {}).get("name")
            else:
                name = getattr(call, "name", None)
            if name:
                tool_counts[name] += 1
        input_tokens += int(usage.get("input_tokens", usage.get("input", 0)) or 0)
        output_tokens += int(usage.get("output_tokens", usage.get("output", 0)) or 0)
    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": tool_counts.get("task", 0),
        "tool_calls": dict(sorted(tool_counts.items())),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("sandbox did not produce a non-empty report.md")
    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError("sandbox did not produce sources.json")
    try:
        report_text = report_bytes.decode("utf-8")
        sources = json.loads(sources_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"sandbox output is invalid: {exc}") from exc
    if not isinstance(sources, list) or not sources or not all(isinstance(item, dict) for item in sources):
        raise RuntimeError("sources.json must be a non-empty JSON list of objects")

    name = slugify(topic)
    report_path = Path(reports_dir) / f"{name}.md"
    sources_path = Path(reports_dir) / f"{name}.sources.json"
    meta_path = Path(reports_dir) / f"{name}.meta.json"
    meta = {
        "topic": topic.strip(),
        **summarize(messages, elapsed, model_name),
        "n_sources": len(sources),
        "source_families": sorted({
            str(item.get("source")) for item in sources if item.get("source")
        }),
    }

    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    sources_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


def finalize_and_validate(backend):
    """Run the deterministic in-sandbox citation gate and return (ok, diagnostic)."""
    finalized = backend.execute(f"python3 {FINALIZER_PATH}")
    if finalized.exit_code != 0:
        return False, f"FINALIZER FAILED:\n{finalized.output}"
    validated = backend.execute(f"python3 {VALIDATOR_PATH}")
    if validated.exit_code != 0 or "OK:" not in validated.output:
        return False, f"VALIDATOR FAILED:\n{validated.output}"
    return True, validated.output.strip()


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    topic = (topic or "").strip()
    if not topic:
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    start = time.monotonic()
    try:
        model = make_model()
        model_name = os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL") or type(model).__name__
        with open_sandbox() as backend:
            created = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if created.exit_code != 0:
                raise RuntimeError(f"cannot create sandbox workspace: {created.output}")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })
            agent = build_lead_agent(backend, model)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000},
            )
            messages = result.get("messages", []) if isinstance(result, dict) else []
            ok, diagnostic = finalize_and_validate(backend)
            for _ in range(MAX_REPAIR_ROUNDS):
                if ok:
                    break
                repair = (
                    "The required in-sandbox citation gate failed. Read the current notes, report, and sources.json; "
                    "repair the REPORT BODY and source registry using only retrieved evidence. Do not hand-write "
                    "References. Run the finalizer and validator until OK. Diagnostic from the deterministic gate:\n"
                    f"{diagnostic}"
                )
                result = agent.invoke(
                    {"messages": [*messages, {"role": "user", "content": repair}]},
                    config={"recursion_limit": 1000},
                )
                messages = result.get("messages", messages) if isinstance(result, dict) else messages
                ok, diagnostic = finalize_and_validate(backend)
            if not ok:
                raise RuntimeError(f"citation gate still failing after {MAX_REPAIR_ROUNDS} repairs: {diagnostic}")
            path = save_outputs(
                backend, topic, messages, time.monotonic() - start, model_name
            )
    except Exception as exc:  # cleanup is guaranteed by open_sandbox's context manager
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Saved report to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
