const API_URL = ""; // change if your backend runs elsewhere

const $ = (sel) => document.querySelector(sel);

// One session per tab. sessionStorage is wiped when the tab closes.
const sid = sessionStorage.getItem("sid") || crypto.randomUUID();
sessionStorage.setItem("sid", sid);

const state = {
  mode: "general",
  files: [],
  busy: false,
  uploading: false,
  history: { general: [], document: [] }, // the backend keeps a separate memory per mode
};

const el = {
  messages: $("#messages"), input: $("#input"), send: $("#sendBtn"), composer: $("#composer"),
  fileInput: $("#fileInput"), drop: $("#dropzone"), dropTitle: $("#dropTitle"), dropSub: $("#dropSub"),
  fileList: $("#fileList"), clear: $("#clearBtn"), hint: $("#modeHint"),
  title: $("#chatTitle"), switch: $("#modeSwitch"),
};

const COPY = {
  general: {
    title: "General chat",
    hint: "Ask anything. Answers come from the model, not from your files.",
    placeholder: "Message the assistant",
    empty: ["Ask me |anything", "This mode doesn’t use your documents. Switch to Documents to ask about your files."],
  },
  document: {
    title: "Document chat",
    hint: "Answers come only from your uploaded files, with the sources listed.",
    placeholder: "Ask about your documents",
    empty: ["Ask about your |documents", "Questions unrelated to your files will be declined."],
    emptyNoFiles: ["Add a |document| to begin", "Upload a PDF, DOCX, TXT or MD file on the left, then ask questions about it."],
  },
};

/* ---------- Rendering ---------- */
function buildMessage(m) {
  const div = document.createElement("div");
  div.className = `msg ${m.role}` + (m.error ? " error" : "");
  let text = m.text;
  let sources = null;

  if (m.role === "assistant") {
    const match = text.match(/\n\nSources: (.+)$/s);
    if (match) {
      text = text.slice(0, match.index);
      sources = match[1].split(",").map((s) => s.trim()).filter(Boolean);
    }
  }
  div.append(document.createTextNode(text));

  if (sources) {
    const row = document.createElement("div");
    row.className = "sources";
    sources.forEach((s) => {
      const chip = document.createElement("span");
      chip.textContent = s;
      row.append(chip);
    });
    div.append(row);
  }
  return div;
}

const ROBOT = `<svg viewBox="0 0 260 190" class="hero-art" aria-hidden="true">
<ellipse cx="130" cy="172" rx="95" ry="12" fill="var(--accent-soft)"/>
<rect x="52" y="62" width="40" height="40" rx="11" fill="#8b5cf6" transform="rotate(-10 72 82)"/>
<text x="72" y="91" font-size="26" font-weight="700" fill="#fff" text-anchor="middle">?</text>
<rect x="172" y="30" width="38" height="38" rx="10" fill="#22c07a" transform="rotate(8 191 49)"/>
<rect x="182" y="40" width="18" height="18" rx="3" fill="#fff" opacity=".9"/>
<rect x="196" y="86" width="34" height="34" rx="10" fill="#fbbf24" transform="rotate(12 213 103)"/>
<line x1="130" y1="22" x2="130" y2="40" stroke="var(--accent)" stroke-width="4"/>
<circle cx="130" cy="18" r="6" fill="var(--accent)"/>
<rect x="88" y="38" width="84" height="66" rx="30" fill="#fff" stroke="var(--line)" stroke-width="3"/>
<rect x="98" y="50" width="64" height="42" rx="20" fill="#14213d"/>
<path d="M112 72q6-8 12 0M136 72q6-8 12 0M120 82q10 8 20 0" stroke="#7dd3fc" stroke-width="4" fill="none" stroke-linecap="round"/>
<rect x="100" y="106" width="60" height="50" rx="24" fill="#fff" stroke="var(--line)" stroke-width="3"/>
<rect x="150" y="110" width="34" height="44" rx="6" fill="var(--accent)" transform="rotate(8 167 132)"/>
<path d="M156 124h20M156 134h20M156 144h12" stroke="#fff" stroke-width="3" stroke-linecap="round" transform="rotate(8 167 132)"/>
</svg>`;

const CHIPS = [["✨", "Tell me about AI"], ["💡", "Explain LangGraph"], ["📊", "Get coding help"], ["📖", "Learn something new"]];

function emptyState() {
  const c = COPY[state.mode];
  const [title, body] = state.mode === "document" && !state.files.length ? c.emptyNoFiles : c.empty;
  const [pre, hl = "", post = ""] = title.split("|");
  const div = document.createElement("div");
  div.className = "empty";
  div.innerHTML = ROBOT + "<h3></h3><p></p>";
  const em = document.createElement("em");
  em.textContent = hl;
  div.querySelector("h3").append(pre, em, post);
  div.querySelector("p").textContent = body;
  if (state.mode === "general") {
    const row = document.createElement("div");
    row.className = "chips";
    CHIPS.forEach(([icon, label]) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = `${icon} ${label}`;
      b.addEventListener("click", () => { if (!state.busy) send(label); });
      row.append(b);
    });
    div.append(row);
  }
  return div;
}

function render() {
  el.messages.replaceChildren();
  const list = state.history[state.mode];
  if (!list.length) el.messages.append(emptyState());
  else list.forEach((m) => el.messages.append(buildMessage(m)));
  scrollDown();
}

function scrollDown() {
  el.messages.scrollTop = el.messages.scrollHeight;
}

function addMessage(mode, role, text, error = false) {
  const m = { role, text, error };
  state.history[mode].push(m);
  if (mode !== state.mode) return;
  el.messages.querySelector(".empty")?.remove();
  el.messages.append(buildMessage(m));
  scrollDown();
}

function showTyping() {
  el.messages.querySelector(".empty")?.remove();
  const t = document.createElement("div");
  t.className = "typing";
  t.innerHTML = "<i></i><i></i><i></i>";
  el.messages.append(t);
  scrollDown();
  return t;
}

function updateUI() {
  const c = COPY[state.mode];
  document.body.dataset.mode = state.mode;
  el.switch.dataset.mode = state.mode;
  el.switch.querySelectorAll("button").forEach((b) =>
    b.setAttribute("aria-pressed", String(b.dataset.mode === state.mode)));
  el.title.textContent = c.title;
  el.hint.textContent = c.hint;

  const locked = state.mode === "document" && !state.files.length;
  el.input.disabled = locked || state.busy;
  el.send.disabled = locked || state.busy;
  el.input.placeholder = locked ? "Add a document first" : c.placeholder;

  el.clear.hidden = !state.files.length;
  el.fileList.replaceChildren(...state.files.map((f) => {
    const li = document.createElement("li");
    const name = document.createElement("span");
    const meta = document.createElement("span");
    name.className = "name"; name.textContent = f.file; name.title = f.file;
    meta.className = "meta"; meta.textContent = `${f.chunks} passages`;
    li.append(name, meta);
    return li;
  }));

  el.dropTitle.textContent = state.uploading ? "Indexing your files…" : "Add documents";
  el.dropSub.textContent = state.uploading
    ? "This can take a moment on the first upload."
    : "Drop files or click to browse. PDF, DOCX, TXT, MD.";
  el.drop.classList.toggle("busy", state.uploading);
}

function setMode(mode) {
  state.mode = mode;
  updateUI();
  render();
  if (!el.input.disabled) el.input.focus();
}

/* ---------- API helpers ---------- */
function errText(data) {
  const d = data && data.detail;
  if (typeof d === "string") return d;
  if (d && d.error) return d.error;
  if (Array.isArray(d)) return "The request was invalid. Check your message and try again.";
  return "Something went wrong. Try again.";
}

function networkText(e) {
  return e instanceof TypeError
    ? "Can’t reach the server. Check that the backend is running."
    : e.message;
}

/* ---------- Upload ---------- */
async function uploadFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;

  const fd = new FormData();
  fd.append("session_id", sid);
  files.forEach((f) => fd.append("files", f));

  state.uploading = true;
  updateUI();
  setMode("document"); // so the result is visible

  try {
    const res = await fetch(`${API_URL}/build_index`, { method: "POST", body: fd });
    const data = await res.json().catch(() => ({}));
    const info = res.ok ? data : (data.detail && typeof data.detail === "object" ? data.detail : {});

    const indexed = info.indexed || [];
    const rejected = info.rejected || [];
    state.files.push(...indexed);
    rejected.forEach((r) => addMessage("document", "notice", `Couldn’t index ${r.file}: ${r.reason}`, true));

    if (indexed.length) {
      const n = indexed.length;
      addMessage("document", "notice", `${n} ${n === 1 ? "file" : "files"} ready. Ask a question.`);
    } else if (!rejected.length) {
      addMessage("document", "notice", errText(data), true);
    }
  } catch (e) {
    addMessage("document", "notice", networkText(e), true);
  } finally {
    state.uploading = false;
    el.fileInput.value = "";
    updateUI();
    render();
  }
}

/* ---------- Chat ---------- */
async function send(text) {
  const mode = state.mode;
  addMessage(mode, "user", text);
  state.busy = true;
  updateUI();
  const typing = showTyping();

  try {
    const res = await fetch(`${API_URL}${mode === "general" ? "/chat" : "/docs_chat"}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid, message: text }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(errText(data));
    addMessage(mode, "assistant", data.answer);
  } catch (e) {
    addMessage(mode, "notice", networkText(e), true);
  } finally {
    typing.remove();
    state.busy = false;
    updateUI();
    el.input.focus();
  }
}

async function clearDocuments() {
  if (!confirm("Remove all documents and clear both chats?")) return;
  await fetch(`${API_URL}/session/end`, { method: "POST", body: sid }).catch(() => {});
  state.files = [];
  state.history = { general: [], document: [] };
  setMode("general");
}

/* ---------- Events ---------- */
el.switch.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-mode]");
  if (btn && btn.dataset.mode !== state.mode) setMode(btn.dataset.mode);
});

el.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = el.input.value.trim();
  if (!text || state.busy) return;
  el.input.value = "";
  el.input.style.height = "auto";
  send(text);
});

el.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    el.composer.requestSubmit();
  }
});

el.input.addEventListener("input", () => {
  el.input.style.height = "auto";
  el.input.style.height = Math.min(el.input.scrollHeight, 160) + "px";
});

el.fileInput.addEventListener("change", () => uploadFiles(el.fileInput.files));
el.clear.addEventListener("click", clearDocuments);

["dragenter", "dragover"].forEach((ev) =>
  el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) =>
  el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.remove("over"); }));
el.drop.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));

// Keep the session alive while the tab is open, and clear it when the tab closes.
setInterval(() => {
  fetch(`${API_URL}/session/ping`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sid }),
  }).catch(() => {});
}, 60000);

window.addEventListener("pagehide", () => navigator.sendBeacon(`${API_URL}/session/end`, sid));

// Theme, sidebar and attach controls
const root = document.documentElement;
root.dataset.theme = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
$("#themeBtn").addEventListener("click", () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
});
document.querySelectorAll(".js-toggle").forEach((b) =>
  b.addEventListener("click", () => $("#app").classList.toggle("collapsed")));
$("#addBtn").addEventListener("click", () => el.fileInput.click());
$("#attachBtn").addEventListener("click", () => el.fileInput.click());

updateUI();
render();