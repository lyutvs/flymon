// FlyMon Live: renders viewer events (flymon/live/events.py, version 1) for one fly at a time.
// Unknown event types and fields are ignored, so the brain can add data without touching this file.
"use strict";

const params = new URLSearchParams(location.search);
const COACH_KIND = { attack: "공격", support: "보조기", switch: "교체", default: "기본" };
const EFFECT = { super: "효과 굉장함", neutral: "보통", resisted: "효과 별로", immune: "효과 없음", unknown: "상성 알 수 없음" };

const view = { flies: new Map(), fly: params.get("fly"), follow: true, turn: null, ignored: 0 };
// NOT `$`: the Showdown renderer on this page uses jQuery's global `$`, and a global const would shadow it.
const el = (id) => document.getElementById(id);

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmt(v) {
  return typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : String(v);
}

// ---- state --------------------------------------------------------------------------------------
function battleOf(ev) {
  let fly = view.flies.get(ev.fly);
  if (!fly) {
    fly = { battles: new Map(), last: null };
    view.flies.set(ev.fly, fly);
    if (!view.fly) view.fly = ev.fly;
  }
  let b = fly.battles.get(ev.battle_tag);
  if (!b) {
    b = { tag: ev.battle_tag, turns: new Map(), maxTurn: 0, end: null, lines: [], lastLineTurn: 0 };
    fly.battles.set(ev.battle_tag, b);
    fly.last = ev.battle_tag;
  }
  return b;
}

function turnOf(b, turn) {
  let t = b.turns.get(turn);
  if (!t) {
    t = { decisions: [], outcomes: [], traces: [] };
    b.turns.set(turn, t);
  }
  b.maxTurn = Math.max(b.maxTurn, turn);
  return t;
}

function ingest(ev) {
  if (!ev || ev.v !== 1 || typeof ev.fly !== "string" || typeof ev.battle_tag !== "string") {
    view.ignored += 1;
    return;
  }
  const kinds = { decision: "decisions", outcome: "outcomes", trace: "traces" };
  if (ev.type === "battle_start") {
    battleOf(ev);
    if (ev.fly === view.fly) view.follow = true;
  } else if (ev.type === "battle_end") {
    battleOf(ev).end = ev;
  } else if (ev.type === "protocol") {
    const b = battleOf(ev);
    b.lines.push(...ev.lines);
    for (const line of ev.lines) {
      const m = /^\|turn\|(\d+)/.exec(line);
      if (m) b.lastLineTurn = Number(m[1]);
    }
  } else if (kinds[ev.type]) {
    turnOf(battleOf(ev), ev.turn)[kinds[ev.type]].push(ev);
  } else {
    view.ignored += 1;
  }
}

function currentBattle() {
  const fly = view.flies.get(view.fly);
  return fly && fly.last ? fly.battles.get(fly.last) : null;
}

// ---- the official battle renderer (replay-embed.js) -------------------------------------------
const renderer = {
  tag: null,
  fed: 0,
  ok: null,
  available() {
    const b = window.Replays && window.Replays.battle;
    return !!(b && typeof b.add === "function" && typeof b.instantAdd === "function" && typeof Replays.init === "function");
  },
  show(battle) {
    if (!this.available()) {
      this.ok = false;
      return false;
    }
    this.ok = true;
    if (this.tag !== battle.tag) {
      try { Replays.battle.destroy(); } catch (e) { /* older renderer: the new init replaces it */ }
      document.querySelector("script.battle-log-data").textContent = "";
      Replays.init();
      this.tag = battle.tag;
      this.fed = 0;
      for (const line of battle.lines) Replays.battle.instantAdd(line);
      this.fed = battle.lines.length;
      Replays.battle.play();
      this.catchUp(battle);                                                 // start at the live turn
      return true;
    }
    for (let i = this.fed; i < battle.lines.length; i++) Replays.battle.add(battle.lines[i]);
    this.fed = battle.lines.length;
    if (Replays.battle.paused) Replays.battle.play();
    this.catchUp(battle);
    return true;
  },
  catchUp(battle) {
    // animation is slower than play: if it falls behind, jump to the newest turn instead of drifting
    if (battle.lastLineTurn && Replays.battle.turn < battle.lastLineTurn - 1) {
      try { Replays.battle.seekTurn(battle.lastLineTurn); } catch (e) { /* older renderer: let it animate */ }
    }
  }
};

// ---- panels -------------------------------------------------------------------------------------
function barsHTML(title, values, labels) {
  const lo = Math.min(0, ...values), hi = Math.max(0, ...values), span = hi - lo || 1;
  const zero = ((0 - lo) / span) * 100;
  const lines = values.map((v, i) => {
    const w = (Math.abs(v) / span) * 100, left = v >= 0 ? zero : zero - w;
    return `<div class="line"><span>${esc(labels[i])}</span><span class="track">` +
      `<span class="fill${v < 0 ? " neg" : ""}" style="left:${left}%;width:${w}%"></span></span><span>${fmt(v)}</span></div>`;
  });
  return `<div class="bars"><div class="title">${esc(title)}</div>${lines.join("")}</div>`;
}

function detailHTML(detail, cands, depth = 0) {
  return Object.entries(detail).map(([k, v]) => {
    if (Array.isArray(v) && v.length > 0 && v.length === cands.length && v.every((x) => typeof x === "number")) {
      return barsHTML(k, v, cands);
    }
    if (v !== null && typeof v === "object" && !Array.isArray(v) && depth < 2) {
      return `<div class="sub"><div class="subtitle">${esc(k)}</div>${detailHTML(v, cands, depth + 1)}</div>`;
    }
    const text = v === null || ["number", "string", "boolean"].includes(typeof v) ? fmt(v) : JSON.stringify(v);
    return `<div class="row"><span class="k">${esc(k)}</span><span>${esc(text)}</span></div>`;
  }).join("");
}

function decisionHTML(d) {
  const who = d.decider === "fly" ? '<span class="badge fly">초파리</span>' : '<span class="badge coach">코치</span>';
  const cands = d.candidates.map((c) => `<li class="${c === d.chosen ? "chosen" : ""}">${esc(c)}${c === d.chosen ? " ✔" : ""}</li>`);
  const chosen = d.candidates.includes(d.chosen) || d.chosen === null ? "" : `<div class="muted">선택: ${esc(d.chosen)}</div>`;
  const detail = d.detail && typeof d.detail === "object" ? `<div class="detail">${detailHTML(d.detail, d.candidates)}</div>` : "";
  return `<div class="decision">${who} 코치 주문: ${esc(COACH_KIND[d.coach_kind] || d.coach_kind)}` +
    (cands.length ? `<ul class="cands">${cands.join("")}</ul>` : "") + chosen + detail + "</div>";
}

function outcomeHTML(ev) {
  const o = ev.outcome || {};
  const flag = (label, on) => `<span>${label} ${on ? "✔" : "✗"}</span>`;
  return `<div><strong>${esc(o.move_id || "?")}</strong> → 직접 피해 ${Math.round((o.dealt_frac || 0) * 100)}%` +
    ` · ${esc(EFFECT[o.effectiveness] || o.effectiveness)}</div>` +
    `<div class="flags">${flag("기절", o.target_fainted_by_me)}${flag("빗나감", o.missed)}` +
    `${flag("행동 불가", o.no_action)}${flag("불확실", o.uncertain)}</div>` +
    (o.notes && o.notes.length ? `<div class="muted">${esc(o.notes.join(" · "))}</div>` : "");
}

function sparkHTML(points, t0) {
  const xs = points.map((p) => p[0] - t0), ys = points.map((p) => Number(p[1]));
  const xmax = Math.max(...xs) || 1, ymin = Math.min(...ys), yspan = Math.max(...ys) - ymin || 1;
  const pts = xs.map((x, i) => `${((x / xmax) * 100).toFixed(1)},${(26 - ((ys[i] - ymin) / yspan) * 24).toFixed(1)}`);
  return `<svg viewBox="0 0 100 28" preserveAspectRatio="none"><polyline points="${pts.join(" ")}"/></svg>`;
}

function barSvgHTML(values) {
  const vmax = Math.max(...values, 0) || 1, w = 100 / values.length;
  const rects = values.map((v, i) => {
    const h = (Math.max(v, 0) / vmax) * 26;
    return `<rect x="${(i * w + w * 0.1).toFixed(1)}" y="${(27 - h).toFixed(1)}" width="${(w * 0.8).toFixed(1)}" height="${h.toFixed(1)}"/>`;
  });
  return `<svg viewBox="0 0 100 28" preserveAspectRatio="none">${rects.join("")}</svg>`;
}

function traceHTML(tr, cands) {
  const all = tr.series.flatMap((s) => s.points.map((p) => p[0]));
  const t0 = all.length ? Math.min(...all) : 0;                  // the tap clock runs on: rebase to 0
  const slot = tr.slot === null || tr.slot === undefined ? "" : ` · 후보 ${tr.slot}${cands[tr.slot] ? " " + cands[tr.slot] : ""}`;
  const body = tr.series.map((s) => {
    if (!s.points.length) return "";
    const last = s.points[s.points.length - 1];
    if (s.kind === "scalar") {
      return `<div class="series"><span>${esc(s.path)}</span>${sparkHTML(s.points, t0)}<span>${fmt(Number(last[1]))}</span></div>`;
    }
    if (s.kind === "bars") {
      return `<div class="series"><span>${esc(s.path)}</span>${barSvgHTML(last[1].map(Number))}<span>${last[1].length}개</span></div>`;
    }
    if (s.kind === "text") {
      return `<div class="texts">${s.points.map((p) => `${Math.round(p[0] - t0)} ms: ${esc(p[1])}`).join("<br>")}</div>`;
    }
    return "";
  }).join("");
  return `<div class="group"><div class="title">${esc(tr.phase)}${esc(slot)}</div>${body}</div>`;
}

function chipHTML(turn, t, selected) {
  const fly = t.decisions.some((d) => d.decider === "fly");
  const o = t.outcomes.length ? t.outcomes[t.outcomes.length - 1].outcome || {} : null;
  let mark = "";
  if (o) {
    if (o.missed) mark = " ✗빗나감";
    else if (o.no_action) mark = " 행동불가";
    else if (o.uncertain) mark = " ?";
    else if (o.target_fainted_by_me) mark = " 기절";
    else if (o.dealt_frac > 0) mark = ` ✔${Math.round(o.dealt_frac * 100)}%`;
  }
  return `<button type="button" class="chip${fly ? " fly" : ""}${selected ? " selected" : ""}" data-turn="${turn}">` +
    `${turn} ${fly ? "초파리" : "코치"}${mark}</button>`;
}

// ---- render -------------------------------------------------------------------------------------
function render() {
  const sel = el("fly");
  const names = [...view.flies.keys()].sort();
  if (sel.options.length !== names.length || [...sel.options].some((o, i) => o.value !== names[i])) {
    sel.innerHTML = names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  }
  sel.value = view.fly || "";
  el("ignored").hidden = view.ignored === 0;
  el("ignored").textContent = `무시한 이벤트 ${view.ignored}`;

  const b = currentBattle();
  el("battle").textContent = b ? b.tag + (b.end ? ` (끝남: ${b.end.won === true ? "승" : b.end.won === false ? "패" : "무"})` : "") : "배틀 없음";
  const note = el("battle-note");
  if (!b) {
    note.textContent = "배틀 없음";
  } else if (renderer.show(b)) {
    note.textContent = `${b.tag} · 프로토콜 ${b.lines.length}줄`;
  } else {
    note.textContent = "배틀 화면을 불러오지 못했습니다 (인터넷 연결 또는 Showdown 렌더러 확인). 결정·결과 패널은 그대로 동작합니다.";
  }

  const turns = b ? [...b.turns.keys()].sort((x, y) => x - y) : [];
  const turn = view.follow ? (turns.length ? turns[turns.length - 1] : null) : view.turn;
  const t = b && turn !== null ? b.turns.get(turn) : null;
  el("turn").textContent = turn === null ? "" : `턴 ${turn}`;
  el("follow").hidden = view.follow;

  el("decision").innerHTML = t && t.decisions.length ? t.decisions.map(decisionHTML).join("") : "아직 결정 없음";
  el("outcome").innerHTML = t && t.outcomes.length ? t.outcomes.map(outcomeHTML).join("") : "결과 대기 중";
  // trace slots are the fly's candidates: a forced switch adds a coach decision with none
  const flyDecisions = t ? t.decisions.filter((d) => d.decider === "fly") : [];
  const cands = flyDecisions.length ? flyDecisions[flyDecisions.length - 1].candidates : [];
  el("brain").innerHTML = t && t.traces.length ?
    `<div class="groups">${t.traces.map((tr) => traceHTML(tr, cands)).join("")}</div>` : "뇌 미연결";
  el("timeline").innerHTML = turns.map((n) => chipHTML(n, b.turns.get(n), n === turn)).join("");
}

// ---- wiring -------------------------------------------------------------------------------------
el("fly").addEventListener("change", (e) => { view.fly = e.target.value; view.follow = true; render(); });
el("follow").addEventListener("click", () => { view.follow = true; render(); });
el("timeline").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  view.follow = false;
  view.turn = Number(chip.dataset.turn);
  render();
});

const stream = new EventSource("/stream");
stream.addEventListener("state", (e) => {
  view.flies.clear();
  view.ignored = 0;
  for (const ev of JSON.parse(e.data).events) ingest(ev);
  render();
});
stream.onmessage = (e) => { ingest(JSON.parse(e.data)); render(); };
stream.onopen = () => { el("conn").className = "conn on"; el("conn").textContent = "연결됨"; };
stream.onerror = () => { el("conn").className = "conn off"; el("conn").textContent = "재연결 중"; };
render();
