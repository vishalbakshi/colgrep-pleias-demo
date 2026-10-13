#!/usr/bin/env python3
"""A local page for asking questions about a repository.

colgrep --agent finds the code that answers a question, Pleias-RAG reads that code and writes
an answer, and every run is saved as a JSON trace for later review.

Usage: python3 server.py /path/to/repo [--port 8000] [--pleias http://127.0.0.1:8081]

Uses only the Python standard library.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Pleias-RAG reads 4,096 tokens in total, and its reasoning and answer use the same budget.
CONTEXT = 4096
ANSWER_TOKENS = 1536  # kept free for the reasoning trace and the answer
# Each location goes to Pleias with 3 lines around it and at most 40 lines, as colgrep prints it.
CONTEXT_LINES = 3
MAX_LINES = 40
# Asked bare, Pleias sometimes judged the sources irrelevant. Framing the question as one about
# the sources helped in multivector-rag-demo (llm-worker.js).
QUERY_PREFIX = "According to the sources: "
# Greedy decoding with no repetition penalty, the defaults of Pleias's own library.
SAMPLING = {"temperature": 0, "repeat_penalty": 1.0}
TRACE_VERSION = 1

RUN_LOCK = threading.Lock()  # one question at a time: both models share this Mac


class ClientGone(Exception):
    """The browser closed the connection (the page was reloaded or closed) during a run."""


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slug(text, limit=48):
    words = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return words[:limit].rstrip("-") or "question"


def git_info(repo):
    """The repository's commit, so a trace records which version of the code was searched."""

    def git(*args):
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        commit = git("rev-parse", "HEAD")
        if commit is None:
            return None
        return {"commit": commit, "uncommitted_changes": bool(git("status", "--porcelain"))}
    except OSError:  # no git on PATH
        return None


def trace_path(folder, query):
    """traces/2026-10-12/153012-where-are-retries-configured.json, numbered if it exists."""
    now = datetime.now()
    day = folder / now.strftime("%Y-%m-%d")
    stem = f"{now.strftime('%H%M%S')}-{slug(query)}"
    path, n = day / f"{stem}.json", 2
    while path.exists():
        path, n = day / f"{stem}-{n}.json", n + 1
    return path


def list_traces(folder):
    """Every saved trace, newest first, with enough to label it in the page's list."""
    found = []
    for path in sorted(folder.glob("*/*.json"), reverse=True):
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        found.append({
            "path": str(path.relative_to(folder)),
            "query": trace.get("query"),
            "status": trace.get("status"),
            "started_at": trace.get("started_at"),
        })
    return found


def save_trace(path, trace):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# colgrep ------------------------------------------------------------------------------------


def run_colgrep(command_name, repo, query, record):
    """Run colgrep --agent --json and return its output. Fills `record` as it goes, so the
    trace keeps colgrep's stderr even when the run fails."""
    with tempfile.TemporaryDirectory() as tmp:
        transcript_file = Path(tmp) / "transcript.json"
        # COLGREP_AGENT_TRACE makes colgrep save the agent's whole conversation: its prompt,
        # every tool call, and every tool result.
        env = dict(os.environ, COLGREP_AGENT_TRACE=str(transcript_file))
        # -y builds the index without asking when the repository has none yet. "--" keeps a
        # question that starts with "-" from being read as an option.
        command = [command_name, "--agent", "--json", "-y", "--", query, str(repo)]
        record["command"] = command
        started = time.time()
        try:
            result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError(
                f"Could not find {command_name!r}. Install colgrep, or pass its path with --colgrep."
            )
        record["seconds"] = round(time.time() - started, 2)
        record["exit_code"] = result.returncode
        record["stderr"] = result.stderr
        if transcript_file.exists():
            record["transcript"] = json.loads(transcript_file.read_text(encoding="utf-8"))

    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()[-5:]
        raise RuntimeError(f"colgrep exited with code {result.returncode}: " + " ".join(tail))
    try:
        output = json.loads(result.stdout[result.stdout.index("{"):])
    except ValueError:
        raise RuntimeError("colgrep did not print JSON. Its output is saved in the trace.")
    record["output"] = output
    return output


def snippet(repo, location):
    """The located lines with a little context, at most MAX_LINES, as plain lines."""
    path = Path(location["file"])
    start, end = location["start_line"], location["end_line"]
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    first = max(1, start - CONTEXT_LINES)
    last = min(len(lines), end + CONTEXT_LINES, first + MAX_LINES - 1)
    rel = os.path.relpath(path, repo)
    return {
        "location": f"{rel}:{start}-{end}",
        "path": rel,
        "start_line": start,
        "end_line": end,
        "first_line": first,
        "last_line": last,
        "lines": lines[first - 1 : last],
    }


# Pleias ------------------------------------------------------------------------------------


def source_text(source):
    header = f"{source['path']} (lines {source['first_line']} to {source['last_line']})"
    return header + "\n" + "\n".join(source["lines"])


def build_prompt(query, sources):
    """The layout Pleias's own library builds (pleias_rag_interface/RAGWithCitations.py)."""
    text = f"<|query_start|>{QUERY_PREFIX}{query}<|query_end|>\n"
    for i, source in enumerate(sources, 1):
        text += f"<|source_start|><|source_id|>{i} {source_text(source)}<|source_end|>\n"
    return text + "<|language_start|>\n"


class Pleias:
    """A llama-server running a Pleias-RAG GGUF, started with --special."""

    def __init__(self, url):
        self.url = url.rstrip("/")

    def request(self, path, body=None, timeout=60):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.url + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"The Pleias server answered {error.code} for {path}: {error.read()[:300]!r}")
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Could not reach the Pleias server at {self.url} ({error.reason}). "
                "Start llama-server with the Pleias model on that port (see the README)."
            )

    def props(self):
        with self.request("/props") as res:
            return json.load(res)

    def count_tokens(self, prompt):
        # The same tokenization /completion uses: special tokens parsed, BOS added.
        body = {"content": prompt, "add_special": True, "parse_special": True}
        with self.request("/tokenize", body) as res:
            return len(json.load(res)["tokens"])


def fit_sources(pleias, query, snippets, budget):
    """Use as many snippets as fit in `budget` prompt tokens, in colgrep's order. If even the
    first one doesn't fit alone, cut lines from its end until it does."""
    used = [dict(s, lines=list(s["lines"])) for s in snippets if s["lines"]]
    while True:
        prompt = build_prompt(query, used)
        tokens = pleias.count_tokens(prompt)
        if tokens <= budget or not used:
            return used, prompt, tokens
        if len(used) > 1:
            used.pop()
        elif len(used[0]["lines"]) > 1:
            used[0]["lines"].pop()
            used[0]["last_line"] -= 1
        else:
            return used, prompt, tokens


# One run ------------------------------------------------------------------------------------


def run(app, query, emit):
    """Answer one question, sending events to the page with `emit`, and save the trace."""
    started = time.time()
    path = trace_path(app.traces, query)
    trace = {
        "trace_version": TRACE_VERSION,
        "id": path.stem,
        "started_at": timestamp(),
        "finished_at": None,
        "seconds": None,
        "status": "running",
        "error": None,
        "query": query,
        "repo": {"path": str(app.repo), "git": git_info(app.repo)},
        "colgrep": {},
        "pleias": None,
        "review": {"rating": None, "notes": ""},  # for you to fill in when you review runs
    }
    final = None
    try:
        output = run_colgrep(app.colgrep, app.repo, query, trace["colgrep"])
        snippets = [snippet(app.repo, loc) for loc in output.get("locations", [])]
        trace["colgrep"]["snippets"] = snippets
        emit({
            "type": "colgrep",
            "output": output,
            "transcript": trace["colgrep"].get("transcript"),
            "locations": snippets,
            "seconds": trace["colgrep"]["seconds"],
        })
        if not any(s["lines"] for s in snippets):
            raise RuntimeError("colgrep returned no code locations, so Pleias has nothing to read.")

        pleias = Pleias(app.pleias)
        props = pleias.props()
        context = min(CONTEXT, props.get("default_generation_settings", {}).get("n_ctx") or CONTEXT)
        used, prompt, prompt_tokens = fit_sources(pleias, query, snippets, context - ANSWER_TOKENS)
        settings = {"n_predict": context - prompt_tokens, "cache_prompt": False, **SAMPLING}
        sent = {s["location"] for s in used}
        record = trace["pleias"] = {
            "server": pleias.url,
            "model_path": props.get("model_path"),
            "model_ftype": props.get("model_ftype"),  # the GGUF's weight type, e.g. F16 or Q8_0
            "build_info": props.get("build_info"),
            "context": context,
            "settings": settings,
            "query_prefix": QUERY_PREFIX,
            "sources": [
                {"id": i, "location": s["location"], "first_line": s["first_line"],
                 "last_line": s["last_line"], "text": source_text(s)}
                for i, s in enumerate(used, 1)
            ],
            # Located code that didn't fit in the prompt (unreadable files are left out).
            "dropped": [s["location"] for s in snippets if s["lines"] and s["location"] not in sent],
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "completion": "",
            "tokens_predicted": None,
            "stop_type": None,
            "truncated": None,
            "timings": None,
            "seconds": None,
        }
        emit({
            "type": "prompt",
            "sources": record["sources"],
            "dropped": record["dropped"],
            "tokens": prompt_tokens,
            "model": record["model_path"],
            "ftype": record["model_ftype"],
        })

        pieces, last, t = [], {}, time.time()
        try:
            body = {"prompt": prompt, "stream": True, **settings}
            with pleias.request("/completion", body, timeout=600) as res:
                for raw in res:  # server-sent events: "data: {...}" lines, blanks, ": ping"s
                    line = raw.decode("utf-8").strip()
                    if line.startswith("error:"):
                        raise RuntimeError(f"The Pleias server stopped: {line[len('error:'):].strip()}")
                    if not line.startswith("data: "):
                        continue
                    chunk = json.loads(line[len("data: "):])
                    if "error" in chunk:
                        raise RuntimeError(f"The Pleias server stopped: {chunk['error']}")
                    text = chunk.get("content") or ""
                    if text:
                        pieces.append(text)
                        emit({"type": "text", "text": text})
                    if chunk.get("stop"):
                        last = chunk
                        break
        finally:
            record["completion"] = "".join(pieces)
            record["seconds"] = round(time.time() - t, 2)
        for key in ("tokens_predicted", "stop_type", "truncated", "timings"):
            record[key] = last.get(key)

        trace["status"] = "complete"
        rate = (last.get("timings") or {}).get("predicted_per_second")
        final = {"type": "done", "tokens": last.get("tokens_predicted"), "seconds": record["seconds"], "rate": rate}
    except ClientGone:
        trace["status"] = "cancelled"
    except Exception as error:  # anything else: report it and keep the partial trace
        trace["status"] = "error"
        trace["error"] = str(error)
        final = {"type": "error", "message": str(error)}
    finally:
        trace["finished_at"] = timestamp()
        trace["seconds"] = round(time.time() - started, 2)
        save_trace(path, trace)
        print(f"Saved {path} ({trace['status']})")

    if final:
        final["trace"] = str(path.relative_to(HERE)) if HERE in path.parents else str(path)
        try:
            emit(final)
        except ClientGone:
            pass


# HTTP ---------------------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "colgrep-pleias-demo"

    def send_body(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, value):
        self.send_body(status, "application/json", json.dumps(value).encode())

    def do_GET(self):
        app = self.server.app
        if self.path in ("/", "/index.html"):
            self.send_body(200, "text/html; charset=utf-8", (HERE / "index.html").read_bytes())
        elif self.path == "/info":
            self.send_json(200, {"repo": str(app.repo), "pleias": app.pleias, "traces": str(app.traces)})
        elif self.path == "/traces":
            self.send_json(200, list_traces(app.traces))
        elif self.path.startswith("/trace?"):
            name = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("path", [""])[0]
            path = (app.traces / name).resolve()
            if path.suffix != ".json" or app.traces not in path.parents or not path.is_file():
                return self.send_json(404, {"error": "No such trace"})
            self.send_body(200, "application/json", path.read_bytes())
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/run":
            return self.send_json(404, {"error": "Not found"})
        length = int(self.headers.get("Content-Length") or 0)
        try:
            query = (json.loads(self.rfile.read(length) or b"{}").get("query") or "").strip()
        except ValueError:
            query = ""
        if not query:
            return self.send_json(400, {"error": "Type a question first."})
        if not RUN_LOCK.acquire(blocking=False):
            return self.send_json(409, {"error": "A question is still running. Wait for it to finish."})
        try:
            # No Content-Length: the response streams one JSON event per line until the run ends.
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            run(self.server.app, query, self.emit)
        finally:
            RUN_LOCK.release()

    def emit(self, event):
        try:
            self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            raise ClientGone()


def main():
    parser = argparse.ArgumentParser(description="Ask questions about a repository with colgrep --agent and Pleias-RAG.")
    parser.add_argument("repo", type=Path, help="the repository to ask about")
    parser.add_argument("--port", type=int, default=8000, help="port for this page (default: 8000)")
    parser.add_argument("--pleias", default="http://127.0.0.1:8081", help="the Pleias llama-server (default: %(default)s)")
    parser.add_argument("--colgrep", default="colgrep", help="the colgrep command or its path (default: colgrep)")
    parser.add_argument("--traces", type=Path, default=HERE / "traces", help="folder for saved traces (default: traces/ next to this file)")
    app = parser.parse_args()
    app.repo = app.repo.expanduser().resolve()
    app.traces = app.traces.expanduser().resolve()
    if not app.repo.is_dir():
        parser.error(f"{app.repo} is not a folder")

    server = ThreadingHTTPServer(("127.0.0.1", app.port), Handler)  # this Mac only
    server.app = app
    print(f"Open http://127.0.0.1:{app.port} to ask about {app.repo}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
