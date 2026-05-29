#!/usr/bin/env python3
"""
CoT Batch Generator — Production Version
• True parallel batching with ThreadPoolExecutor (MAX_WORKERS)
• Real SSE streaming (text/event-stream)
• Modern UI with live logs, stats, and safe syntax highlighting
• Robust error handling and queue-based thread communication
"""

import json
import http.server
import socketserver
import urllib.parse
import urllib.request
import os
import time
import threading
import queue
import uuid
import unicodedata
from datetime import datetime
from html import escape as html_escape
from concurrent.futures import ThreadPoolExecutor

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("⚠️  Install 'requests' for streaming: pip install requests")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "qwen/qwen3.5-9b"
PORT = 8080
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.jsonl")
MAX_WORKERS = 6
STREAM_BUFFER_SIZE = 1  # Send each token immediately

FILE_LOCK = threading.Lock()

PUNCT_MAP = {
    '\u2014': '--', '\u2013': '-', '\u2018': "'", '\u2019': "'",
    '\u201c': '"',  '\u201d': '"', '\u2026': '...', '\u00A0': ' ',
}
TRANSTBL = str.maketrans(PUNCT_MAP)

def to_ascii(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.translate(TRANSTBL)
    s = unicodedata.normalize('NFKD', s)
    return s.encode('ascii', 'ignore').decode('ascii')

def asciiize_entry(entry: dict) -> dict:
    new = json.loads(json.dumps(entry))
    for m in new.get('messages', []):
        if 'content' in m and m['content'] is not None:
            m['content'] = to_ascii(m['content'])
    return new

def is_valid_jsonl_entry(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    messages = entry.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return False
    expected_roles = ("system", "user", "assistant")
    for idx, role in enumerate(expected_roles):
        msg = messages[idx]
        if not isinstance(msg, dict) or msg.get("role") != role:
            return False
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            return False
    return True

# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT (full, as requested)
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a generator for the wop/XXXXXL-chain-of-thought dataset.

Given a user QUESTION, respond with ONLY the assistant reply content + thinking.
Make sure you put the think tags.
The reply must have this structure:

<think>
### Cold start
[Analyze the situation, state what you know/don't know]

### Hot start
[Work through the answer step by step, then conclude]
</think>

[Visible answer to the user]

Examples:

User: hi
Assistant:
<think>
### Cold start
I am stateless right now — completely cold-starting on this input. That matters. It means I cannot infer emotional state from history, cannot reference a prior topic, cannot assume familiarity. Every bit of signal I have is encoded in exactly those two characters and nothing else.
Ready for hot starts.

### Hot start
0 system 0 tools 1 message

The user said: 'hi'. That's it. No punctuation, no context, no prior turn, no system-level memory of who this person is or what they've been doing.
</think>

**Hello** there. 😃
How may I be helpful at the moment?

User: How many R in strawberry
Assistant:
<think>
### Cold start
The user requested me to count how many R letters are in the word strawberry. And I have no other memory access, no tool access and no system message to guide me. I am completely stateless and isolated.

### Hot start
0 system 0 tools 1 message
Let's think.
We can split the word into characters.
And verify each one.
Sounds like a good plan, let's do this.
S
T
R  I found one
A
W
B
E
R  I found two
R  I found three
Y
I found three letters R in strawberry.
Okay, but there is a problem.
What problem?
The user provided an uppercase letter and did not specify case sensitiveness.
True. We should account for that.
What can we do about this?
I think I can just mention it.
Sounds fine. Go ahead.
Okay. Proceeding. Because now I know what I have to do.
Early finish.
</think>

**Great question to stress-test!**

There are **3** letter r's in the word *strawberry*:

> st**r**awbe**rr**y

---

**🧐 A nuance worth noting is case sensitivity.**

You wrote capital **R** in your question.

| Target | Count |
|--------|-------|
| lowercase **r** | **3** |
| uppercase **R** | **0** |

The word *strawberry* contains no uppercase **R** at all.

IMPORTANT: Output ONLY the assistant reply content + thinking. No JSON wrapper, no explanations, no extra text."""

# ─────────────────────────────────────────────────────────────
# DATA HANDLING
# ─────────────────────────────────────────────────────────────
def build_entry(question: str, assistant_content: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Enable thinking features: INTUITION, COLD START, HOT START"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def load_entries(valid_only: bool = False) -> list:
    if not os.path.exists(DATA_FILE):
        return []
    entries = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if not valid_only or is_valid_jsonl_entry(entry):
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
    return entries

def save_entry(entry: dict) -> None:
    with FILE_LOCK:
        with open(DATA_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()

# ─────────────────────────────────────────────────────────────
# STREAMING MODEL CALL (Worker Thread)
# ─────────────────────────────────────────────────────────────
def call_model_streaming(question: str, event_queue: queue.Queue, run_id: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"User: {question}\nAssistant:"}
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
        "stream": True
    }

    start_time = time.time()
    full_content = ""
    char_count = 0

    def send_event(event_type: str, data: dict):
        event_queue.put({
            "run_id": run_id,
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            **data
        })

    try:
        send_event("start", {"question": question})

        if HAS_REQUESTS:
            resp = requests.post(API_URL, json=payload, stream=True, timeout=300)
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                text = line.strip()
                if text.startswith("data: "):
                    data_str = text[6:].strip()
                else:
                    data_str = text
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except (json.JSONDecodeError, TypeError):
                    continue

                try:
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if not content:
                        message = chunk.get("choices", [{}])[0].get("message", {})
                        content = message.get("content", "") or chunk.get("text", "")
                except Exception:
                    content = ""

                if content:
                    full_content += content
                    char_count = len(full_content)
                    elapsed = time.time() - start_time
                    cps = round(char_count / elapsed, 1) if elapsed > 0 else 0
                    send_event("token", {
                        "content": content,
                        "cumulative_chars": char_count,
                        "cps": cps,
                        "elapsed": round(elapsed, 2)
                    })
        else:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(API_URL, data=req_data, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                full_content = result["choices"][0]["message"]["content"].strip()
                send_event("token", {
                    "content": full_content,
                    "cumulative_chars": len(full_content),
                    "cps": 0,
                    "elapsed": 0
                })

        entry = build_entry(question, full_content)
        save_entry(entry)
        duration = time.time() - start_time

        send_event("complete", {
            "duration": round(duration, 2),
            "total_chars": char_count,
            "assistant_content": full_content,
            "entry": entry,
            "success": True
        })
        return {"ok": True, "question": question, "entry": entry, "duration": round(duration, 2)}

    except Exception as e:
        send_event("error", {"error": str(e), "success": False, "question": question})
        return {"ok": False, "question": question, "error": str(e)}

# ─────────────────────────────────────────────────────────────
# HTML UI (use raw string so backslashes are preserved)
# ─────────────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>CoT Generator Pro</title>
<style>
:root{
  --bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;
  --accent:#58a6ff;--success:#238636;--error:#da3633;--warning:#d29922;
  --muted:#8b949e;--code:#6e7681
}
*{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  background:var(--bg);color:var(--text);line-height:1.5;
  padding:16px;max-width:1400px;margin:0 auto
}
header{
  display:flex;justify-content:space-between;align-items:center;
  padding:12px 0;border-bottom:1px solid var(--border);margin-bottom:20px
}
header h1{font-size:1.4rem;color:var(--accent)}
.status{
  display:flex;gap:8px;align-items:center;font-size:13px;color:var(--muted)
}
.status-dot{width:10px;height:10px;border-radius:50%;background:var(--error)}
.status-dot.ok{background:var(--success)}
.status-dot.loading{background:var(--warning);animation:pulse 1s infinite}
@keyframes pulse{50%{opacity:.5}}

.card{
  background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:16px;margin-bottom:16px
}
.card-header{
  display:flex;justify-content:space-between;align-items:center;
  margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border)
}
.card-title{font-weight:600;color:var(--accent);font-size:1rem}

textarea{
  width:100%;background:#0d1117;border:1px solid var(--border);
  color:var(--text);border-radius:6px;padding:12px;font-family:monospace;
  font-size:13px;resize:vertical;min-height:100px
}
textarea:focus{outline:2px solid var(--accent);border-color:transparent}

.btn{
  background:#21262d;border:1px solid var(--border);color:var(--text);
  padding:8px 16px;border-radius:6px;cursor:pointer;font-size:13px;
  transition:background.15s
}
.btn:hover{background:#30363d}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn-primary{background:var(--success);border:none;color:#fff;font-weight:600}
.btn-primary:hover{background:#2ea043}
.btn-sm{padding:4px 10px;font-size:12px}

.grid{display:grid;gap:16px}
.grid-2{grid-template-columns:1fr 1fr}
.grid-3{grid-template-columns:repeat(3,1fr)}
@media(max-width:900px){.grid-2,.grid-3{grid-template-columns:1fr}}

/* Live Stream Panel */
.stream-panel{
  display:flex;flex-direction:column;height:400px;border:1px solid var(--border);
  border-radius:8px;overflow:hidden
}
.stream-header{
  display:flex;justify-content:space-between;gap:12px;padding:10px 14px;
  background:#0d1117;border-bottom:1px solid var(--border);font-size:13px
}
.stream-head-main{min-width:0;display:flex;gap:8px;align-items:center;flex:1}
.stream-question{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.stream-status{
  color:var(--success);font-size:11px;border:1px solid var(--border);
  border-radius:999px;padding:1px 8px;white-space:nowrap;background:#161b22
}
.stream-toolbar{display:flex;gap:8px;align-items:center}
.run-strip{
  display:flex;gap:6px;align-items:center;padding:8px 10px;
  background:#161b22;border-bottom:1px solid var(--border);overflow-x:auto
}
.run-pill{
  border:1px solid var(--border);background:#21262d;color:var(--muted);
  border-radius:999px;padding:3px 9px;font-size:11px;white-space:nowrap;cursor:pointer
}
.run-pill.active{border-color:var(--accent);color:var(--accent);background:#0d1117}
.stream-content{
  flex:1;padding:14px;overflow-y:auto;font-family:monospace;font-size:13px;
  white-space:pre-wrap;background:#0d1117
}
.stream-content.empty{color:var(--muted);font-style:italic}
.stream-footer{
  display:flex;gap:10px;align-items:center;padding:8px 10px;
  background:#0d1117;border-top:1px solid var(--border);font-size:11px;color:var(--muted)
}
.stream-progress{flex:1;margin:0;height:5px}
.think{color:var(--warning);font-weight:700}
.content{color:var(--text)}
.stats{display:flex;gap:12px;font-size:12px;color:var(--muted)}
.stat{display:flex;gap:4px}
.stat-val{color:var(--accent);font-weight:600}

/* Results List */
.result-item{
  border:1px solid var(--border);border-radius:6px;padding:12px;
  margin-bottom:8px;background:#0d1117
}
.result-item.ok{border-left:3px solid var(--success)}
.result-item.err{border-left:3px solid var(--error)}
.result-q{font-weight:600;color:var(--accent);margin-bottom:6px;font-size:13px}
.result-preview{
  color:var(--code);font-size:12px;white-space:pre-wrap;
  max-height:80px;overflow:hidden;text-overflow:ellipsis
}
.result-meta{
  display:flex;gap:12px;margin-top:8px;font-size:11px;color:var(--muted)
}

/* Logs Panel */
.logs-panel{max-height:200px;overflow-y:auto;font-family:monospace;font-size:12px}
.log-entry{padding:4px 0;border-bottom:1px dashed var(--border);display:flex;gap:8px}
.log-time{color:var(--muted);min-width:80px}
.log-info{color:var(--accent)}.log-warn{color:var(--warning)}.log-err{color:var(--error)}

/* Progress */
.progress-bar{
  height:6px;background:#21262d;border-radius:3px;overflow:hidden;margin:8px 0
}
.progress-fill{
  height:100%;background:var(--accent);transition:width.2s ease
}

/* Utilities */
.flex{display:flex}.gap-8{gap:8px}.gap-12{gap:12px}.items-center{align-items:center}
.justify-between{justify-content:space-between}.mt-8{margin-top:8px}.text-sm{font-size:13px}
.text-xs{font-size:11px}.muted{color:var(--muted)}.success{color:var(--success)}.error{color:var(--error)}
.copy-btn{margin-left:auto;padding:2px 8px;font-size:11px}
</style>
</head>
<body>

<header>
  <h1>🧠 CoT Generator Pro</h1>
  <div class="status">
    <span class="status-dot" id="apiStatus"></span>
    <span id="apiStatusText">Checking API...</span>
  </div>
</header>

<div class="grid grid-2">
  <!-- Input Panel -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">📝 Input Questions</span>
      <button class="btn btn-sm" onclick="clearInput()">Clear</button>
    </div>
    <textarea id="questions" placeholder="Enter questions (one per line)...">What is the meaning of life?
Explain quantum entanglement simply
Write a haiku about debugging
How does a microwave work?</textarea>
    <div class="flex gap-8 items-center mt-8">
      <button class="btn btn-primary" id="generateBtn" onclick="startBatch()">▶ Generate Batch</button>
      <span class="text-sm muted" id="batchInfo">0 questions</span>
    </div>
  </div>

  <!-- Stats Panel -->
  <div class="card">
    <div class="card-header"><span class="card-title">📊 Live Stats</span></div>
    <div class="grid grid-3">
      <div class="stat"><span class="muted">Queued:</span><span class="stat-val" id="statQueued">0</span></div>
      <div class="stat"><span class="muted">Processing:</span><span class="stat-val" id="statActive">0</span></div>
      <div class="stat"><span class="muted">Completed:</span><span class="stat-val" id="statDone">0</span></div>
      <div class="stat"><span class="muted">Avg Speed:</span><span class="stat-val" id="statSpeed">–</span></div>
      <div class="stat"><span class="muted">Success Rate:</span><span class="stat-val" id="statSuccess">–</span></div>
      <div class="stat"><span class="muted">Total Time:</span><span class="stat-val" id="statTime">–</span></div>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
    <div class="text-xs muted" id="progressText">0 / 0 completed</div>
    <div class="text-xs muted" id="etaLine">ETA: <span id="statETA">–</span> • Finish: <span id="statFinish">–</span></div>
  </div>
</div>

<!-- Live Stream Panel -->
<div class="card">
  <div class="card-header">
    <span class="card-title">🔴 Live Stream</span>
    <div class="stats">
      <div class="stat"><span class="muted">Speed:</span><span class="stat-val" id="liveSpeed">–</span></div>
      <div class="stat"><span class="muted">Chars:</span><span class="stat-val" id="liveChars">0</span></div>
      <div class="stat"><span class="muted">Time:</span><span class="stat-val" id="liveTime">0.0s</span></div>
    </div>
  </div>
  <div class="stream-panel">
    <div class="stream-header">
      <div class="stream-head-main">
        <span id="streamQuestion" class="stream-question muted">Waiting...</span>
        <span id="streamStatus" class="stream-status">Idle</span>
      </div>
      <div class="stream-toolbar">
        <button class="btn btn-sm copy-btn" onclick="copyStream()">📋 Copy</button>
      </div>
    </div>
    <div class="run-strip" id="runStrip"><span class="muted text-xs">No active streams</span></div>
    <div class="stream-content empty" id="streamContent">Start generating to see live output...</div>
    <div class="stream-footer">
      <div class="progress-bar stream-progress"><div class="progress-fill" id="streamProgressFill" style="width:0%"></div></div>
      <span id="liveEstimate">Run ETA: –</span>
    </div>
  </div>
</div>

<!-- Results + Logs -->
<div class="grid grid-2">
  <div class="card">
    <div class="card-header">
      <span class="card-title">✅ Completed Entries</span>
      <div class="flex gap-8">
        <button class="btn btn-sm" onclick="refreshEntries()">↻ Refresh</button>
        <button class="btn btn-sm" onclick="copyJSONL()">📋 Copy JSONL</button>
        <button class="btn btn-sm" onclick="downloadJSONL()">⬇ Download</button>
      </div>
    </div>
    <div id="resultsList" style="max-height:300px;overflow-y:auto">
      <p class="muted text-sm">No entries yet.</p>
    </div>
  </div>
  
  <div class="card">
    <div class="card-header">
      <span class="card-title">🪵 Activity Log</span>
      <button class="btn btn-sm" onclick="clearLogs()">Clear</button>
    </div>
    <div class="logs-panel" id="logsPanel"></div>
  </div>
</div>

<script>
// ─────────────────────────────────────────────────────────────
// STATE & UTILS
// ─────────────────────────────────────────────────────────────
const state = {
  running: false,
  batchTotal: 0,
  results: [],
  logs: [],
  history: { durations: [], chars: [], cps: [] },
  stats: { queued:0, active:0, done:0, failed:0, totalChars:0, totalTime:0 }
};

const runMeta = {};
const SERVER_MAX_WORKERS = 6;
const DEFAULT_AVG_SEC = 12;
const HISTORY_LIMIT = 40;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const now = () => new Date().toLocaleTimeString();
const clamp = (v, min, max) => Math.min(max, Math.max(min, v));

function log(msg, type='info'){
  const entry = { time: now(), msg, type };
  state.logs.unshift(entry);
  if(state.logs.length > 200) state.logs.pop();
  renderLogs();
}

function renderLogs(){
  const panel = $('logsPanel');
  if(state.logs.length === 0){
    panel.innerHTML = '<p class="muted text-sm">No logs yet.</p>';
    return;
  }
  panel.innerHTML = state.logs.map(l => 
    `<div class="log-entry"><span class="log-time">${l.time}</span><span class="log-${l.type}">${esc(l.msg)}</span></div>`
  ).join('');
  panel.scrollTop = 0;
}

function updateStats(){
  $('statQueued').textContent = state.stats.queued;
  $('statActive').textContent = state.stats.active;
  $('statDone').textContent = state.stats.done;
  $('statSuccess').textContent = state.stats.done + state.stats.failed > 0 
    ? Math.round(state.stats.done/(state.stats.done+state.stats.failed)*100)+'%' : '–';
  $('statTime').textContent = state.stats.totalTime > 0 ? state.stats.totalTime.toFixed(1)+'s' : '–';
  $('statSpeed').textContent = state.stats.totalChars > 0 && state.stats.totalTime > 0
    ? Math.round(state.stats.totalChars/state.stats.totalTime)+' c/s' : '–';
  
  const total = state.batchTotal || (state.stats.queued + state.stats.active + state.stats.done + state.stats.failed);
  const done = state.stats.done + state.stats.failed;
  $('progressFill').style.width = total ? (done/total*100)+'%' : '0%';
  $('progressText').textContent = `${done} / ${total} completed`;
}

function formatSeconds(sec) {
  if (!isFinite(sec) || sec <= 0) return '0s';
  const s = Math.round(sec);
  if (s < 60) return s + 's';
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}m ${r}s`;
}

function positiveNumbers(values) {
  return (values || []).filter(v => Number.isFinite(v) && v > 0);
}

function pushHistory(bucket, value) {
  if (!Number.isFinite(value) || value <= 0) return;
  bucket.push(value);
  if (bucket.length > HISTORY_LIMIT) bucket.splice(0, bucket.length - HISTORY_LIMIT);
}

function percentile(values, p) {
  const arr = positiveNumbers(values).sort((a,b)=>a-b);
  if (!arr.length) return null;
  const idx = (arr.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return arr[lo];
  return arr[lo] + (arr[hi] - arr[lo]) * (idx - lo);
}

function trimmedMean(values, trim=0.15) {
  const arr = positiveNumbers(values).sort((a,b)=>a-b);
  if (!arr.length) return null;
  if (arr.length < 5) return arr.reduce((a,b)=>a+b, 0) / arr.length;
  const cut = Math.min(Math.floor(arr.length * trim), Math.floor((arr.length - 1) / 2));
  const trimmed = arr.slice(cut, arr.length - cut);
  return trimmed.reduce((a,b)=>a+b, 0) / trimmed.length;
}

function getDurationEstimate() {
  const hist = positiveNumbers(state.history.durations);
  if (!hist.length) return DEFAULT_AVG_SEC;
  const avg = hist.reduce((a,b)=>a+b, 0) / hist.length;
  if (hist.length < 3) {
    const defaultWeight = Math.max(0, 3 - hist.length);
    return ((avg * hist.length) + (DEFAULT_AVG_SEC * defaultWeight)) / (hist.length + defaultWeight);
  }
  const trimmed = trimmedMean(hist) || avg;
  const median = percentile(hist, 0.5) || trimmed;
  const p75 = percentile(hist, 0.75) || median;
  return Math.max(1, trimmed * 0.50 + median * 0.30 + p75 * 0.20);
}

function getCharEstimate() {
  const hist = positiveNumbers(state.history.chars);
  if (!hist.length) return null;
  const avg = hist.reduce((a,b)=>a+b, 0) / hist.length;
  if (hist.length < 3) return avg;
  return Math.max(1,
    (trimmedMean(hist) || avg) * 0.45 +
    (percentile(hist, 0.5) || avg) * 0.25 +
    (percentile(hist, 0.75) || avg) * 0.30
  );
}

function getCpsEstimate() {
  const hist = positiveNumbers(state.history.cps);
  if (!hist.length) return null;
  return trimmedMean(hist) || hist.reduce((a,b)=>a+b, 0) / hist.length;
}

function activeRunList(runMetaMap=runMeta) {
  return Object.entries(runMetaMap)
    .filter(([, meta]) => meta && !meta.done)
    .sort((a,b)=>(a[1].start || 0) - (b[1].start || 0));
}

function getRunElapsed(meta) {
  const liveElapsed = (Date.now()/1000) - (meta.start || (Date.now()/1000));
  return Math.max(0, meta.lastElapsed || 0, liveElapsed);
}

function estimateActiveRun(meta, durationEstimate=getDurationEstimate(), charEstimate=getCharEstimate()) {
  const elapsed = getRunElapsed(meta);
  const chars = Math.max(0, meta.chars || 0);
  const cps = meta.cps || (chars > 0 && elapsed > 0 ? chars / elapsed : 0);
  const timeRemaining = Math.max(0, durationEstimate - elapsed);
  let charRemaining = null;
  let targetChars = null;

  if (charEstimate && chars > 0 && cps > 0.1) {
    targetChars = Math.max(charEstimate, chars * 1.04);
    charRemaining = Math.max(0, targetChars - chars) / cps;
  }

  let remaining = timeRemaining;
  if (charRemaining !== null) {
    const liveWeight = clamp(chars / Math.max(targetChars || chars, 1), 0.35, 0.82);
    remaining = charRemaining * liveWeight + timeRemaining * (1 - liveWeight);
  }

  if (elapsed > durationEstimate && charRemaining !== null) remaining = Math.max(1.5, charRemaining);
  if (elapsed > durationEstimate && charRemaining === null) remaining = Math.max(2, durationEstimate * 0.15);

  const lastTokenAge = (Date.now()/1000) - (meta.lastTokenAt || meta.start || (Date.now()/1000));
  if (chars > 0 && lastTokenAge < 3 && remaining < 1.5) remaining = 1.5;

  const progressTarget = targetChars || Math.max(durationEstimate, elapsed + remaining);
  const progress = targetChars
    ? clamp(chars / progressTarget, 0, remaining <= 0.5 ? 1 : 0.98)
    : clamp(elapsed / progressTarget, 0, remaining <= 0.5 ? 1 : 0.98);

  return { remaining, progress, elapsed, cps };
}

function estimateQueuedDuration(durationEstimate) {
  const historicalCps = getCpsEstimate();
  const liveCpsValues = activeRunList()
    .map(([, meta]) => meta.cps || 0)
    .filter(v => Number.isFinite(v) && v > 0);
  if (!historicalCps || !liveCpsValues.length) return durationEstimate;
  const liveCps = liveCpsValues.reduce((a,b)=>a+b, 0) / liveCpsValues.length;
  const paceScale = clamp(historicalCps / Math.max(liveCps, 0.1), 0.75, 2.5);
  return durationEstimate * paceScale;
}

function estimateRemainingTime(durationEstimate, runMetaMap, queuedCount, concurrency) {
  const M = Math.max(1, Math.floor(concurrency || 1));
  const activeRemaining = activeRunList(runMetaMap)
    .map(([, meta]) => estimateActiveRun(meta, durationEstimate).remaining)
    .sort((a,b)=>a-b);
  const slots = activeRemaining.slice(0, M);
  while (slots.length < M) slots.push(0);
  const queuedDur = estimateQueuedDuration(durationEstimate);
  for (let i = 0; i < queuedCount; i++) {
    let minIdx = 0;
    for (let j = 1; j < slots.length; j++) {
      if (slots[j] < slots[minIdx]) minIdx = j;
    }
    slots[minIdx] += queuedDur;
  }
  return Math.max(...slots);
}

function updateEstimator() {
  const queued = state.stats.queued;
  const active = state.stats.active;
  const remainingCount = queued + active;
  if (remainingCount === 0) {
    $('statETA').textContent = '0s';
    $('statFinish').textContent = '–';
    return;
  }
  const avg = getDurationEstimate();
  const concurrency = SERVER_MAX_WORKERS || Math.max(1, state.stats.active || 1);
  const remainingSec = estimateRemainingTime(avg, runMeta, queued, concurrency);
  $('statETA').textContent = '≈ ' + formatSeconds(remainingSec);
  try {
    $('statFinish').textContent = new Date(Date.now() + remainingSec*1000).toLocaleTimeString();
  } catch(e) {
    $('statFinish').textContent = '–';
  }
}

// ─────────────────────────────────────────────────────────────
// STREAM HANDLING (per-run buffers, safer highlighting)
// ─────────────────────────────────────────────────────────────
const runBuffers = {};          // run_id -> full accumulated string
const finishedRuns = {};        // run_id -> final status/meta for the selected stream
let currentRunId = null;

function highlightThinkTags(escapedText){
  return escapedText
    .replace(/&lt;think&gt;/g, '<span class="think">&lt;think&gt;</span>')
    .replace(/&lt;\/think&gt;/g, '<span class="think">&lt;/think&gt;</span>');
}

function selectRun(run_id) {
  if (!run_id || !(run_id in runBuffers)) return;
  currentRunId = run_id;
  renderRunStrip();
  renderLiveStream();
}

function renderRunStrip() {
  const strip = $('runStrip');
  const active = activeRunList();
  if (!active.length) {
    strip.innerHTML = '<span class="muted text-xs">No active streams</span>';
    return;
  }
  if (!currentRunId || (!(currentRunId in runMeta) && !finishedRuns[currentRunId])) {
    currentRunId = active[active.length - 1][0];
  }
  strip.innerHTML = active.map(([rid, meta]) => {
    const chars = meta.chars || 0;
    const speed = meta.cps ? `${Math.round(meta.cps)} c/s` : 'warming up';
    const label = esc((meta.question || rid).slice(0, 42));
    const title = esc(meta.question || rid).replace(/"/g, '&quot;');
    return `<button class="run-pill ${rid===currentRunId?'active':''}" onclick="selectRun('${rid}')" title="${title}">${label} • ${chars} chars • ${speed}</button>`;
  }).join('');
}

function renderLiveStream() {
  const active = activeRunList();
  if (!currentRunId && active.length) currentRunId = active[active.length - 1][0];

  const meta = currentRunId ? runMeta[currentRunId] : null;
  const finished = currentRunId ? finishedRuns[currentRunId] : null;
  const text = currentRunId ? (runBuffers[currentRunId] || '') : '';

  if (!currentRunId && !text) {
    $('streamQuestion').textContent = 'Waiting...';
    $('streamStatus').textContent = 'Idle';
    $('streamContent').classList.add('empty');
    $('streamContent').textContent = 'Start generating to see live output...';
    $('liveSpeed').textContent = '–';
    $('liveChars').textContent = '0';
    $('liveTime').textContent = '0.0s';
    $('liveEstimate').textContent = 'Run ETA: –';
    $('streamProgressFill').style.width = '0%';
    return;
  }

  $('streamQuestion').textContent = meta?.question || finished?.question || ('Run ' + currentRunId);
  $('streamStatus').textContent = meta ? 'Streaming' : (finished?.ok === false ? 'Error' : 'Complete');

  if (text) {
    $('streamContent').classList.remove('empty');
    $('streamContent').innerHTML = highlightThinkTags(esc(text));
  } else {
    $('streamContent').classList.add('empty');
    $('streamContent').textContent = 'Waiting for first token...';
  }
  $('streamContent').scrollTop = $('streamContent').scrollHeight;

  if (meta) {
    const live = estimateActiveRun(meta);
    $('liveSpeed').textContent = meta.cps ? Math.round(meta.cps) + ' c/s' : '–';
    $('liveChars').textContent = meta.chars || 0;
    $('liveTime').textContent = live.elapsed.toFixed(1) + 's';
    $('liveEstimate').textContent = `Run ETA: ≈ ${formatSeconds(live.remaining)} • ${Math.round(live.progress * 100)}%`;
    $('streamProgressFill').style.width = (live.progress * 100) + '%';
  } else if (finished) {
    const cps = finished.duration && finished.total_chars ? Math.round(finished.total_chars / finished.duration) : 0;
    $('liveSpeed').textContent = cps ? cps + ' c/s' : '–';
    $('liveChars').textContent = finished.total_chars || text.length || 0;
    $('liveTime').textContent = finished.duration ? finished.duration + 's' : '–';
    $('liveEstimate').textContent = finished.ok === false ? 'Run failed' : 'Run complete';
    $('streamProgressFill').style.width = finished.ok === false ? '0%' : '100%';
  }
}

function handleStreamEvent(data){
  const run_id = data.run_id;
  if(data.type === 'start'){
    runBuffers[run_id] = '';
    currentRunId = run_id;
    runMeta[run_id] = {
      start: Date.now()/1000,
      lastElapsed: 0,
      done: false,
      chars: 0,
      cps: 0,
      question: data.question || ('Run ' + run_id),
      lastTokenAt: null
    };
    log(`Started: ${data.question || run_id} [${run_id}]`, 'info');
    state.stats.active++;
    state.stats.queued = Math.max(0, state.stats.queued - 1);
    renderRunStrip();
    renderLiveStream();
    updateStats();
    updateEstimator();
  }
  else if(data.type === 'token'){
    runBuffers[run_id] = (runBuffers[run_id] || '') + (data.content || '');
    if(runMeta[run_id]){
      const meta = runMeta[run_id];
      meta.lastElapsed = data.elapsed || ((Date.now()/1000) - meta.start);
      meta.chars = data.cumulative_chars || runBuffers[run_id].length || 0;
      const nextCps = data.cps || (meta.chars > 0 && meta.lastElapsed > 0 ? meta.chars / meta.lastElapsed : 0);
      if (nextCps > 0) meta.cps = meta.cps ? (meta.cps * 0.65 + nextCps * 0.35) : nextCps;
      meta.lastTokenAt = Date.now()/1000;
    }
    if(!currentRunId || !(currentRunId in runMeta)) currentRunId = run_id;
    renderRunStrip();
    if(run_id === currentRunId) renderLiveStream();
    updateEstimator();
  }
  else if(data.type === 'complete'){
    state.stats.done++;
    state.stats.active = Math.max(0, state.stats.active - 1);
    state.stats.totalChars += data.total_chars || 0;
    state.stats.totalTime += data.duration || 0;
    pushHistory(state.history.durations, data.duration || 0);
    pushHistory(state.history.chars, data.total_chars || 0);
    if(data.duration && data.total_chars) pushHistory(state.history.cps, data.total_chars / data.duration);
    finishedRuns[run_id] = {
      ok: true,
      question: data.entry?.messages?.[1]?.content || data.question || ('Run ' + run_id),
      duration: data.duration || 0,
      total_chars: data.total_chars || 0
    };
    if(runMeta[run_id]) delete runMeta[run_id];
    log(`Completed: ${data.duration || 0}s, ${data.total_chars || 0} chars [${run_id}]`, 'info');
    if(data.entry){
      state.results.unshift({
        ok: true,
        question: data.entry?.messages?.[1]?.content || data.question || ('Run ' + run_id),
        entry: data.entry,
        duration: data.duration || '–'
      });
      renderResults();
    } else {
      state.results.unshift({
        ok: true,
        question: data.question || ('Run ' + run_id),
        entry: null,
        duration: data.duration || '–'
      });
      renderResults();
    }
    if(run_id === currentRunId){
      const active = activeRunList();
      if(active.length) currentRunId = active[active.length - 1][0];
    }
    renderRunStrip();
    renderLiveStream();
    updateStats();
    updateEstimator();
  }
  else if(data.type === 'error'){
    state.stats.failed++;
    state.stats.active = Math.max(0, state.stats.active - 1);
    finishedRuns[run_id] = {
      ok: false,
      question: data.question || ('Run ' + run_id),
      error: data.error || 'unknown'
    };
    if(runMeta[run_id]) delete runMeta[run_id];
    log(`Error [${run_id}]: ${data.error}`, 'err');
    state.results.unshift({
      ok: false,
      question: data.question || ('Run ' + run_id),
      error: data.error
    });
    renderResults();
    if(run_id === currentRunId){
      const active = activeRunList();
      if(active.length) currentRunId = active[active.length - 1][0];
    }
    renderRunStrip();
    renderLiveStream();
    updateStats();
    updateEstimator();
  }
}

// ─────────────────────────────────────────────────────────────
// MAIN BATCH GENERATION
// ─────────────────────────────────────────────────────────────
async function startBatch(){
  const text = $('questions').value.trim();
  if(!text){ log('No questions entered', 'warn'); return; }
  if(state.running){ log('Already running', 'warn'); return; }
  
  const questions = text.split('\n').map(s=>s.trim()).filter(Boolean);
  if(questions.length === 0) return;
  
  state.running = true;
  state.batchTotal = questions.length;
  state.stats = { queued:questions.length, active:0, done:0, failed:0, totalChars:0, totalTime:0 };
  currentRunId = null;
  for (const rid in runMeta) delete runMeta[rid];
  for (const rid in finishedRuns) delete finishedRuns[rid];
  for (const rid in runBuffers) delete runBuffers[rid];
  renderRunStrip();
  renderLiveStream();
  $('generateBtn').disabled = true;
  $('generateBtn').textContent = 'Generating...';
  log(`Starting batch: ${questions.length} question${questions.length!==1?'s':''}`, 'info');
  updateStats();
  updateEstimator();
  
  try{
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ questions })
    });
    
    if(!res.ok){
      const err = await res.text();
      log(`Server error: ${err}`, 'err');
      return;
    }
    
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      buffer += decoder.decode(value, {stream:true});
      const parts = buffer.split(/\r?\n\r?\n/);
      buffer = parts.pop() || '';
      
      for(const block of parts){
        if(!block.trim()) continue;
        const lines = block.split(/\r?\n/);
        let dataLine = '';
        for(const l of lines){
          if(l.startsWith('data: ')) dataLine += l.slice(6);
          else if(l.startsWith('data:')) dataLine += l.slice(5);
        }
        if(!dataLine) continue;
        try{
          const event = JSON.parse(dataLine);
          handleStreamEvent(event);
        }catch(e){ console.warn('Parse error', e, dataLine); }
      }
    }
  }catch(e){
    log(`Stream error: ${e?.message || e}`, 'err');
  }finally{
    state.running = false;
    $('generateBtn').disabled = false;
    $('generateBtn').textContent = '▶ Generate Batch';
    state.stats.queued = 0;
    updateStats();
    updateEstimator();
    log('Batch finished', 'info');
  }
}

// ─────────────────────────────────────────────────────────────
// UI ACTIONS
// ─────────────────────────────────────────────────────────────
function clearInput(){ $('questions').value = ''; }
function clearLogs(){ state.logs = []; renderLogs(); }

function copyStream(){
  const text = currentRunId ? (runBuffers[currentRunId] || '') : $('streamContent').innerText.trim();
  if(!text){ log('No stream text to copy', 'warn'); return; }
  navigator.clipboard.writeText(text).then(()=> log('Copied stream to clipboard', 'info'));
}

function renderResults(){
  const list = $('resultsList');
  if(state.results.length === 0){
    list.innerHTML = '<p class="muted text-sm">No entries yet.</p>';
    return;
  }
  list.innerHTML = state.results.slice(0,50).map((r,i)=>{
    const preview = r.ok 
      ? (r.entry?.messages?.[2]?.content?.slice(0,150) || r.entry?.messages?.[2]?.content || '').replace(/\n/g,' ') + (r.entry ? '...' : '')
      : r.error;
    return `<div class="result-item ${r.ok?'ok':'err'}">
      <div class="result-q">#${i+1} ${r.ok?'✓':'✗'} ${esc(r.question || '')}</div>
      <div class="result-preview">${esc(preview)}</div>
      <div class="result-meta">
        ${r.ok ? `<span>⏱ ${esc(String(r.duration || '–'))}s</span><span>📝 ${r.entry?.messages?.[2]?.content?.length||0} chars</span>` 
               : `<span class="error">Error: ${esc(r.error || 'unknown')}</span>`}
      </div>
    </div>`;
  }).join('');
}

async function refreshEntries(){
  try{
    const res = await fetch('/api/entries');
    const entries = await res.json();
    state.results = entries.reverse().map(e => ({
      ok: true, question: e.messages[1].content, entry: e, duration: '–'
    }));
    renderResults();
    log(`Loaded ${entries.length} entries from file`, 'info');
  }catch(e){ log(`Refresh failed: ${e?.message || e}`, 'err'); }
}

async function copyJSONL(){
  try{
    const res = await fetch('/api/download?ascii=1&valid=1&bom=0');
    if(!res.ok) throw new Error('Fetch failed: ' + res.status);
    const raw = await res.text();
    const valid = buildValidJSONL(raw);
    if(valid.count === 0) throw new Error('No valid JSONL rows found');
    await navigator.clipboard.writeText(valid.text);
    const skippedMsg = valid.skipped ? `; skipped ${valid.skipped} invalid row${valid.skipped!==1?'s':''}` : '';
    log(`Copied ${valid.count} valid ASCII JSONL row${valid.count!==1?'s':''}${skippedMsg}`, 'info');
  }catch(e){
    log('Copy JSONL failed: ' + (e.message || e), 'err');
    alert('Copy failed: ' + (e.message || e));
  }
}

function isValidJSONLEntry(entry){
  if(!entry || typeof entry !== 'object' || !Array.isArray(entry.messages) || entry.messages.length < 3) return false;
  const roles = ['system', 'user', 'assistant'];
  for(let i=0; i<roles.length; i++){
    const msg = entry.messages[i];
    if(!msg || msg.role !== roles[i] || typeof msg.content !== 'string' || !msg.content.trim()) return false;
  }
  return true;
}

function buildValidJSONL(text){
  const rows = [];
  let skipped = 0;
  const lines = String(text || '').replace(/^\uFEFF/, '').split(/\r?\n/);
  for(const line of lines){
    const trimmed = line.trim();
    if(!trimmed) continue;
    try{
      const parsed = JSON.parse(trimmed);
      if(!isValidJSONLEntry(parsed)){ skipped++; continue; }
      rows.push(JSON.stringify(parsed));
    }catch(e){
      skipped++;
    }
  }
  return { text: rows.join('\n') + (rows.length ? '\n' : ''), count: rows.length, skipped };
}

function downloadJSONL(){
  fetch('/api/download?valid=1').then(r=>{
    const rows = r.headers.get('X-JSONL-Rows');
    return r.blob().then(b=>({b, rows}));
  }).then(({b, rows})=>{
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b);
    a.download = 'train.jsonl';
    a.click();
    log(`Downloaded train.jsonl${rows ? ` (${rows} valid rows)` : ''}`, 'info');
  }).catch(e=> log('Download failed: ' + (e.message || e), 'err'));
}

// ─────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────
async function checkAPI(){
  try{
    const res = await fetch('/api/health');
    const data = await res.json();
    const dot = $('apiStatus'), txt = $('apiStatusText');
    if(data.ok){
      dot.className = 'status-dot ok';
      txt.textContent = 'API Connected';
      log('API health check: OK', 'info');
    }else{
      dot.className = 'status-dot';
      txt.textContent = 'API Error: ' + data.error;
      log('API health check failed: ' + data.error, 'err');
    }
  }catch(e){
    $('apiStatus').className = 'status-dot';
    $('apiStatusText').textContent = 'API Unreachable';
    log('API unreachable: ' + (e?.message || e), 'err');
  }
}

$('questions').addEventListener('input', ()=>{
  const count = $('questions').value.split('\n').filter(s=>s.trim()).length;
  $('batchInfo').textContent = `${count} question${count!==1?'s':''}`;
});

checkAPI();
refreshEntries();
renderLogs();
log('UI initialized', 'info');
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────
# HTTP HANDLER
# ─────────────────────────────────────────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    
    def log_message(self, format, *args):
        pass  # Suppress default logging
    
    def _send_sse_headers(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache, no-transform')
        self.send_header('Connection', 'keep-alive')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()
    
    def _send_json(self, data, status=200, content_type='application/json'):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path.rstrip('/') or '/'
        
        if path == '/':
            body = HTML_PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        
        if path == '/api/entries':
            self._send_json(load_entries())
            return
        
        if path == '/api/download':
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            ascii_mode = qs.get('ascii', ['0'])[0].lower() in ('1', 'true', 'yes', 'on')
            valid_only = qs.get('valid', ['0'])[0].lower() in ('1', 'true', 'yes', 'on')
            include_bom = qs.get('bom', ['1'])[0].lower() not in ('0', 'false', 'no', 'off')

            entries = load_entries(valid_only=valid_only)
            if ascii_mode:
                out_entries = [asciiize_entry(e) for e in entries]
            else:
                out_entries = entries

            body_text = '\n'.join(json.dumps(e, ensure_ascii=False) for e in out_entries)
            body = (b'\xef\xbb\xbf' if include_bom else b'') + body_text.encode('utf-8')

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Disposition', 'attachment; filename=train.jsonl')
            self.send_header('X-JSONL-Rows', str(len(out_entries)))
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        
        if path == '/api/health':
            self._send_json(self._check_api_health())
            return
        
        self.send_error(404)
    
    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        
        if path == '/api/generate':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send_json({'error': 'Invalid JSON'}, 400)
                return
            
            questions = [q.strip() for q in data.get('questions', []) if q.strip()]
            if not questions:
                self._send_json({'error': 'No valid questions'}, 400)
                return
            
            self._send_sse_headers()
            
            event_queue = queue.Queue()
            futures = []
            total = len(questions)
            
            try:
                with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total or 1)) as executor:
                    for q in questions:
                        run_id = str(uuid.uuid4())[:8]
                        futures.append(executor.submit(call_model_streaming, q, event_queue, run_id))
                    
                    completed = 0
                    while completed < total:
                        try:
                            event = event_queue.get(timeout=1.0)
                        except queue.Empty:
                            if all(f.done() for f in futures):
                                # no events and workers done
                                break
                            continue
                        
                        try:
                            s = json.dumps(event, ensure_ascii=False)
                            sse_line = f"data: {s}\n\n"
                            self.wfile.write(sse_line.encode('utf-8'))
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            # client disconnected
                            for f in futures:
                                try:
                                    f.cancel()
                                except Exception:
                                    pass
                            return
                        
                        if event.get('type') in ('complete', 'error'):
                            completed += 1
                    
                    try:
                        self.wfile.write(b"data: {\"type\":\"batch_complete\"}\n\n")
                        self.wfile.flush()
                    except Exception:
                        pass
            except Exception as e:
                try:
                    err_event = {"type": "error", "error": str(e)}
                    sse_line = f"data: {json.dumps(err_event, ensure_ascii=False)}\n\n"
                    self.wfile.write(sse_line.encode('utf-8'))
                    self.wfile.flush()
                except Exception:
                    pass
            return
        
        self.send_error(404)
    def _check_api_health(self):
        try:
            payload = json.dumps({
                "model": MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1
            }).encode('utf-8')
            req = urllib.request.Request(API_URL, data=payload, method='POST')
            req.add_header('Content-Type', 'application/json')
            with urllib.request.urlopen(req, timeout=5) as resp:
                return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"🚀 CoT Generator Pro")
    print(f"   UI:   http://localhost:{PORT}")
    print(f"   API:  {API_URL} → {MODEL}")
    print(f"   Data: {os.path.abspath(DATA_FILE)}")
    print(f"   Workers: {MAX_WORKERS}")
    if not HAS_REQUESTS:
        print("⚠️  Install 'requests' for streaming: pip install requests")
    
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    with http.server.ThreadingHTTPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
