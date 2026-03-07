/* ═══════════════════════════════════════════════════════
   Three.js  –  Particle Network Background
   ═══════════════════════════════════════════════════════ */
(function initBackground() {
  const canvas = document.getElementById("bg-canvas");
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.z = 300;

  const COUNT = 220;
  const posArr = new Float32Array(COUNT * 3);
  const velArr = [];
  for (let i = 0; i < COUNT; i++) {
    posArr[i * 3]     = (Math.random() - 0.5) * 600;
    posArr[i * 3 + 1] = (Math.random() - 0.5) * 600;
    posArr[i * 3 + 2] = (Math.random() - 0.5) * 300;
    velArr.push((Math.random() - 0.5) * 0.15, (Math.random() - 0.5) * 0.15, (Math.random() - 0.5) * 0.05);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
  const mat = new THREE.PointsMaterial({ color: 0xa5b4fc, size: 2, transparent: true, opacity: 0.35 });
  const points = new THREE.Points(geom, mat);
  scene.add(points);

  const lineMat = new THREE.LineBasicMaterial({ color: 0xc7d2fe, transparent: true, opacity: 0.18 });
  let lineGeom = new THREE.BufferGeometry();
  const lines = new THREE.LineSegments(lineGeom, lineMat);
  scene.add(lines);

  function updateLines() {
    const pos = points.geometry.attributes.position.array;
    const segs = [];
    const DIST = 80;
    for (let i = 0; i < COUNT; i++) {
      for (let j = i + 1; j < COUNT; j++) {
        const dx = pos[i * 3] - pos[j * 3];
        const dy = pos[i * 3 + 1] - pos[j * 3 + 1];
        const dz = pos[i * 3 + 2] - pos[j * 3 + 2];
        if (dx * dx + dy * dy + dz * dz < DIST * DIST) {
          segs.push(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
          segs.push(pos[j * 3], pos[j * 3 + 1], pos[j * 3 + 2]);
        }
      }
    }
    lines.geometry.dispose();
    lines.geometry = new THREE.BufferGeometry();
    if (segs.length) lines.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(segs), 3));
  }

  let frame = 0;
  function animate() {
    requestAnimationFrame(animate);
    const pos = points.geometry.attributes.position.array;
    for (let i = 0; i < COUNT; i++) {
      pos[i * 3]     += velArr[i * 3];
      pos[i * 3 + 1] += velArr[i * 3 + 1];
      pos[i * 3 + 2] += velArr[i * 3 + 2];
      if (Math.abs(pos[i * 3]) > 300) velArr[i * 3] *= -1;
      if (Math.abs(pos[i * 3 + 1]) > 300) velArr[i * 3 + 1] *= -1;
      if (Math.abs(pos[i * 3 + 2]) > 150) velArr[i * 3 + 2] *= -1;
    }
    points.geometry.attributes.position.needsUpdate = true;
    if (frame++ % 3 === 0) updateLines();
    points.rotation.y += 0.0002;
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
})();

/* ═══════════════════════════════════════════════════════
   Auth State
   ═══════════════════════════════════════════════════════ */
const TOKEN_KEY = "pdf_assistant_token";
const auth = {
  token: localStorage.getItem(TOKEN_KEY),
  user: null,
};

function authHeaders() {
  return auth.token ? { Authorization: `Bearer ${auth.token}` } : {};
}

async function authFetch(url, options = {}) {
  options.headers = { ...options.headers, ...authHeaders() };
  const res = await fetch(url, options);
  if (res.status === 401) {
    doLogout();
    throw new Error("Session expired. Please sign in again.");
  }
  return res;
}

function doLogin(token, user) {
  auth.token = token;
  auth.user = user;
  localStorage.setItem(TOKEN_KEY, token);
  showPage("app");
  updateUserUI();
  initApp();
}

function doLogout() {
  auth.token = null;
  auth.user = null;
  localStorage.removeItem(TOKEN_KEY);
  state.chats = [];
  state.activeChatId = null;
  showPage("landing");
}

async function checkAuth() {
  if (!auth.token) {
    showPage("landing");
    return;
  }
  try {
    const res = await fetch("/api/auth/me", {
      headers: { Authorization: `Bearer ${auth.token}` },
    });
    if (res.ok) {
      const data = await res.json();
      auth.user = data.user;
      showPage("app");
      updateUserUI();
      initApp();
    } else {
      doLogout();
    }
  } catch {
    doLogout();
  }
}

/* ═══════════════════════════════════════════════════════
   Page & Auth UI Management
   ═══════════════════════════════════════════════════════ */
const elLanding = document.getElementById("page-landing");
const elApp     = document.getElementById("page-app");
const elAuthOvl = document.getElementById("auth-overlay");

function showPage(name) {
  elLanding.style.display = name === "landing" ? "" : "none";
  elApp.style.display     = name === "app" ? "flex" : "none";
  hideAuth();
}

function showAuth(mode) {
  elAuthOvl.classList.add("show");
  switchAuth(mode);
}

function hideAuth() {
  elAuthOvl.classList.remove("show");
  document.getElementById("signin-error").className = "auth-error";
  document.getElementById("signup-error").className = "auth-error";
  document.getElementById("signin-error").textContent = "";
  document.getElementById("signup-error").textContent = "";
}

function switchAuth(mode) {
  document.getElementById("form-signin").style.display = mode === "signin" ? "" : "none";
  document.getElementById("form-signup").style.display = mode === "signup" ? "" : "none";
  document.getElementById("signin-error").className = "auth-error";
  document.getElementById("signup-error").className = "auth-error";
}

function updateUserUI() {
  if (!auth.user) return;
  document.getElementById("sidebar-user-name").textContent = auth.user.name;
  document.getElementById("sidebar-user-email").textContent = auth.user.email;
  document.getElementById("user-avatar").textContent = auth.user.name.charAt(0).toUpperCase();
}

/* ── Auth Event Listeners ──────────────────────────────── */
// Open auth modal
document.getElementById("nav-signin").addEventListener("click", () => showAuth("signin"));
document.getElementById("nav-signup").addEventListener("click", () => showAuth("signup"));
document.getElementById("hero-signin").addEventListener("click", () => showAuth("signin"));
document.getElementById("hero-signup").addEventListener("click", () => showAuth("signup"));

// Close auth modal
document.getElementById("auth-close").addEventListener("click", hideAuth);
elAuthOvl.addEventListener("click", (e) => {
  if (e.target === elAuthOvl) hideAuth();
});

// Switch between forms
document.getElementById("switch-to-signup").addEventListener("click", (e) => { e.preventDefault(); switchAuth("signup"); });
document.getElementById("switch-to-signin").addEventListener("click", (e) => { e.preventDefault(); switchAuth("signin"); });

// Sign In
document.getElementById("signin-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const email = document.getElementById("signin-email").value;
  const password = document.getElementById("signin-password").value;
  const errEl = document.getElementById("signin-error");
  const btn = document.getElementById("signin-submit");

  btn.classList.add("btn-loading");
  btn.textContent = "Signing in…";
  errEl.className = "auth-error";
  errEl.textContent = "";

  try {
    const res = await fetch("/api/auth/signin", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) {
      errEl.textContent = data.detail || "Sign in failed";
      errEl.className = "auth-error show";
      return;
    }
    doLogin(data.token, data.user);
  } catch (err) {
    errEl.textContent = "Network error. Please try again.";
    errEl.className = "auth-error show";
  } finally {
    btn.classList.remove("btn-loading");
    btn.textContent = "Sign In";
  }
});

// Sign Up
document.getElementById("signup-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = document.getElementById("signup-name").value;
  const email = document.getElementById("signup-email").value;
  const password = document.getElementById("signup-password").value;
  const errEl = document.getElementById("signup-error");
  const btn = document.getElementById("signup-submit");

  btn.classList.add("btn-loading");
  btn.textContent = "Creating account…";
  errEl.className = "auth-error";
  errEl.textContent = "";

  try {
    const res = await fetch("/api/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password }),
    });
    const data = await res.json();
    if (!res.ok) {
      errEl.textContent = data.detail || "Sign up failed";
      errEl.className = "auth-error show";
      return;
    }
    doLogin(data.token, data.user);
  } catch (err) {
    errEl.textContent = "Network error. Please try again.";
    errEl.className = "auth-error show";
  } finally {
    btn.classList.remove("btn-loading");
    btn.textContent = "Create Account";
  }
});

// Sign Out
document.getElementById("btn-signout").addEventListener("click", () => {
  if (confirm("Sign out?")) doLogout();
});

/* ═══════════════════════════════════════════════════════
   App State & DOM Refs
   ═══════════════════════════════════════════════════════ */
const $ = (sel) => document.querySelector(sel);
const state = { chats: [], activeChatId: null, streaming: false };

const elChatList   = $("#chat-list");
const elWelcome    = $("#view-welcome");
const elChat       = $("#view-chat");
const elMessages   = $("#messages");
const elMsgInput   = $("#msg-input");
const elBtnSend    = $("#btn-send");
const elBtnNew     = $("#btn-new-chat");
const elDropZone   = $("#drop-zone");
const elFileInput  = $("#file-input");
const elUploadProg = $("#upload-progress");
const elChatTitle  = $("#chat-title");
const elPdfBadge   = $("#chat-pdf-name");

/* ═══════════════════════════════════════════════════════
   Helpers
   ═══════════════════════════════════════════════════════ */
function showView(name) {
  elWelcome.classList.toggle("active", name === "welcome");
  elChat.classList.toggle("active", name === "chat");
}

function timeAgo(iso) {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function renderMarkdown(text) {
  if (typeof marked !== "undefined") {
    marked.setOptions({ breaks: true, gfm: true });
    return marked.parse(text);
  }
  return text.replace(/\n/g, "<br>");
}

function scrollToBottom() {
  requestAnimationFrame(() => { elMessages.scrollTop = elMessages.scrollHeight; });
}

/* ═══════════════════════════════════════════════════════
   Sidebar – Render Chat List
   ═══════════════════════════════════════════════════════ */
function renderChatList() {
  elChatList.innerHTML = "";
  if (state.chats.length === 0) {
    elChatList.innerHTML = `<div style="padding:24px;text-align:center;color:var(--text3);font-size:12px">
      No chats yet.<br>Upload a PDF to start.</div>`;
    return;
  }
  state.chats.forEach((c) => {
    const div = document.createElement("div");
    div.className = "chat-item" + (c.chat_id === state.activeChatId ? " active" : "");
    div.innerHTML = `
      <div class="chat-item-title">${esc(c.title)}</div>
      <div class="chat-item-pdf">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
        </svg>
        ${esc(c.pdf_name)}
      </div>
      <div class="chat-item-date">${timeAgo(c.created_at)}</div>
      <button class="delete-btn" data-id="${c.chat_id}" title="Delete chat">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="3 6 5 6 21 6"/><path d="M19 6l-2 14H7L5 6"/>
          <path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/>
        </svg>
      </button>`;
    div.addEventListener("click", (e) => {
      if (e.target.closest(".delete-btn")) return;
      openChat(c.chat_id);
    });
    div.querySelector(".delete-btn").addEventListener("click", () => deleteChat(c.chat_id));
    elChatList.appendChild(div);
  });
}

function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

/* ═══════════════════════════════════════════════════════
   API Calls (with auth)
   ═══════════════════════════════════════════════════════ */
async function fetchChats() {
  try {
    const res = await authFetch("/api/chats");
    const data = await res.json();
    state.chats = data.chats;
    renderChatList();
  } catch { /* handled by authFetch */ }
}

async function uploadPDF(file) {
  elUploadProg.classList.remove("hidden");
  elDropZone.style.display = "none";
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await authFetch("/api/upload", { method: "POST", body: form });
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch { throw new Error(text.slice(0, 200) || "Server error"); }
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    await fetchChats();
    openChat(data.chat_id);
  } catch (err) {
    alert("Upload failed: " + err.message);
    elDropZone.style.display = "";
  } finally {
    elUploadProg.classList.add("hidden");
  }
}

async function openChat(chatId) {
  state.activeChatId = chatId;
  const chat = state.chats.find((c) => c.chat_id === chatId);
  if (!chat) return;

  elChatTitle.textContent = chat.title;
  elPdfBadge.textContent = "\uD83D\uDCC4 " + chat.pdf_name;
  elMessages.innerHTML = "";
  showView("chat");
  renderChatList();

  try {
    const res = await authFetch(`/api/chats/${chatId}/messages`);
    const data = await res.json();
    data.messages.forEach((m) => appendMessage(m.role, m.content));
    scrollToBottom();
  } catch { /* first message */ }
}

async function deleteChat(chatId) {
  if (!confirm("Delete this chat?")) return;
  await authFetch(`/api/chats/${chatId}`, { method: "DELETE" });
  if (state.activeChatId === chatId) {
    state.activeChatId = null;
    showView("welcome");
    elDropZone.style.display = "";
  }
  await fetchChats();
}

/* ═══════════════════════════════════════════════════════
   Chat – Send & Stream
   ═══════════════════════════════════════════════════════ */
async function sendMessage() {
  const text = elMsgInput.value.trim();
  if (!text || !state.activeChatId || state.streaming) return;

  state.streaming = true;
  elBtnSend.classList.add("is-empty");
  elMsgInput.value = "";
  updateSendBtn();

  appendMessage("user", text);
  scrollToBottom();

  const assistDiv = appendMessage("assistant", "");
  const typingHtml = `<div class="typing-dots"><span></span><span></span><span></span></div>`;
  assistDiv.innerHTML = typingHtml;
  scrollToBottom();

  let fullText = "";
  try {
    const res = await authFetch(`/api/chat/${state.activeChatId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6);
        try {
          const json = JSON.parse(payload);
          if (json.done) break;
          if (json.error) { fullText += `\n\n\u26A0\uFE0F Error: ${json.error}`; break; }
          if (json.content) {
            fullText += json.content;
            assistDiv.innerHTML = renderMarkdown(fullText);
            scrollToBottom();
          }
        } catch { /* skip bad json */ }
      }
    }

    if (!fullText) assistDiv.innerHTML = `<span style="color:var(--text3)">No response received.</span>`;
  } catch (err) {
    assistDiv.innerHTML = `<span style="color:#f87171">Error: ${esc(err.message)}</span>`;
  } finally {
    state.streaming = false;
    updateSendBtn();
    elMsgInput.focus();
  }
}

function appendMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = content ? renderMarkdown(content) : "";
  elMessages.appendChild(div);
  return div;
}

/* ═══════════════════════════════════════════════════════
   App Event Listeners
   ═══════════════════════════════════════════════════════ */

// Send
elBtnSend.addEventListener("click", () => {
  if (!elBtnSend.classList.contains("is-empty")) sendMessage();
});
elMsgInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

function updateSendBtn() {
  const hasText = elMsgInput.value.trim().length > 0 && !state.streaming;
  elBtnSend.classList.toggle("is-empty", !hasText);
}
elMsgInput.addEventListener("input", updateSendBtn);

// Attach / upload new PDF from chat view
const elBtnAttach = $("#btn-attach");
const elChatFileInput = $("#chat-file-input");
if (elBtnAttach && elChatFileInput) {
  elBtnAttach.addEventListener("click", () => elChatFileInput.click());
  elChatFileInput.addEventListener("change", () => {
    if (elChatFileInput.files[0]) {
      uploadPDF(elChatFileInput.files[0]);
      elChatFileInput.value = "";
    }
  });
}

// New chat
elBtnNew.addEventListener("click", () => {
  state.activeChatId = null;
  showView("welcome");
  elDropZone.style.display = "";
  renderChatList();
});

// File upload (welcome view)
elDropZone.addEventListener("click", () => elFileInput.click());
elFileInput.addEventListener("change", () => {
  if (elFileInput.files[0]) uploadPDF(elFileInput.files[0]);
});
elDropZone.addEventListener("dragover", (e) => { e.preventDefault(); elDropZone.classList.add("dragover"); });
elDropZone.addEventListener("dragleave", () => elDropZone.classList.remove("dragover"));
elDropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  elDropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file && file.name.toLowerCase().endsWith(".pdf")) uploadPDF(file);
  else alert("Please drop a PDF file.");
});

/* ═══════════════════════════════════════════════════════
   Init
   ═══════════════════════════════════════════════════════ */
function initApp() {
  fetchChats();
  showView("welcome");
}

// Check authentication on page load
checkAuth();
