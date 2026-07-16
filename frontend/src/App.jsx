import { useState, useEffect, useCallback } from "react";
import {
  AlertTriangle, CheckCircle2, FilePlus2, Search, Loader2,
  ChevronRight, Database, GitBranch, Layers, Activity, Clock,
  ExternalLink, Copy, Check, Terminal, ServerCog
} from "lucide-react";

/* ───────────────────────────────────────────────────────────────────────────
   CONFIG
─────────────────────────────────────────────────────────────────────────── */
const API_BASE = "http://localhost:8000";

const CATEGORY_COLOR = {
  product: "#22D3C8",
  infrastructure: "#F5A623",
  automation: "#A78BFA",
};

const ACTION_META = {
  DUPLICATE: {
    label: "Duplicate",
    color: "#FB5B5B",
    bg: "rgba(251,91,91,0.08)",
    border: "rgba(251,91,91,0.35)",
    Icon: AlertTriangle,
    blurb: "Matches an open ticket. Link to it instead of filing a new one.",
  },
  WORKAROUND_AVAILABLE: {
    label: "Workaround available",
    color: "#22D3C8",
    bg: "rgba(34,211,200,0.08)",
    border: "rgba(34,211,200,0.35)",
    Icon: CheckCircle2,
    blurb: "Matches a resolved ticket. A fix already exists.",
  },
  NEW_ISSUE: {
    label: "New issue",
    color: "#F5A623",
    bg: "rgba(245,166,35,0.08)",
    border: "rgba(245,166,35,0.35)",
    Icon: FilePlus2,
    blurb: "No matching ticket found above the similarity threshold.",
  },
};

/* ───────────────────────────────────────────────────────────────────────────
   MOCK FALLBACK — used only if the FastAPI backend isn't reachable, so the
   dashboard is still inspectable standalone. Mirrors the real API shapes.
─────────────────────────────────────────────────────────────────────────── */
const MOCK_BUILDS = [
  { build_id: "BUILD-5002", job_name: "powerstore-product-regression", failure_category: "product", failure_summary: "Snapshot creation fails with ENOSPC on StoragePool POOL86DX2F", environment: "vmware-integration-env", timestamp: "2026-05-08T09:54:46Z" },
  { build_id: "BUILD-5008", job_name: "powerstore-product-regression", failure_category: "product", failure_summary: "iSCSI session drops intermittently during heavy I/O", environment: "iscsi-regression-lab", timestamp: "2026-06-02T11:12:09Z" },
  { build_id: "BUILD-5005", job_name: "powerstore-infrastructure-regression", failure_category: "infrastructure", failure_summary: "Lab test node san-testbed-qa-03 went offline during nightly regression", environment: "san-testbed-qa-03", timestamp: "2026-05-28T03:40:02Z" },
  { build_id: "BUILD-5009", job_name: "powerstore-automation-regression", failure_category: "automation", failure_summary: "Jenkins pipeline YAML syntax error breaks release/4.1 branch builds", environment: "powerstore-lab-02", timestamp: "2026-06-10T08:02:51Z" },
];

const MOCK_STATS = {
  total_jenkins_builds: 150,
  total_jira_tickets: 300,
  failure_category_distribution: { product: 57, infrastructure: 53, automation: 40 },
  ticket_status_distribution: { Open: 33, "In Progress": 26, "Under Investigation": 29, Reopened: 30, Resolved: 41, Fixed: 33, Closed: 32, "Won't Fix": 41, Duplicate: 35 },
};

function buildMockVerdict(build) {
  const actions = ["DUPLICATE", "WORKAROUND_AVAILABLE", "NEW_ISSUE"];
  let hash = 0;
  for (let i = 0; i < build.build_id.length; i++) hash = (hash * 31 + build.build_id.charCodeAt(i)) >>> 0;
  const action = actions[hash % 3];
  const hasMatch = action !== "NEW_ISSUE";
  return {
    build_id: build.build_id,
    failure_category: build.failure_category,
    bug_fields: {
      bug_title: build.failure_summary,
      component: build.job_name.split("-")[1] ?? "Unknown",
      error_message: build.failure_summary,
      symptoms: "Observed during validation; pipeline marked FAILURE",
      category: build.failure_category,
    },
    top_match_ticket_id: hasMatch ? "PSTR-1242" : null,
    top_match_score: hasMatch ? 0.81 : 0.41,
    is_duplicate: hasMatch,
    confidence: hasMatch ? 0.87 : 0.4,
    reason: hasMatch
      ? "Component and error pattern match the retrieved ticket."
      : "No sufficiently similar ticket found above threshold.",
    action,
    workaround_steps: action === "WORKAROUND_AVAILABLE"
      ? ["Run pool reclamation task manually on the affected pool.", "Retry the snapshot or volume operation.", "Confirm space accounting is consistent before re-enabling automation."]
      : [],
    matched_ticket: hasMatch ? {
      ticket_id: "PSTR-1242",
      summary: "Snapshot creation fails with ENOSPC on StoragePool",
      status: action === "DUPLICATE" ? "Open" : "Resolved",
      component: build.job_name.split("-")[1] ?? "Unknown",
      priority: "Medium",
      assignee: "david.okonkwo",
    } : null,
    elapsed_seconds: 0.8,
    _mock: true,
  };
}

/* ───────────────────────────────────────────────────────────────────────────
   SMALL PRIMITIVES
─────────────────────────────────────────────────────────────────────────── */

function CategoryDot({ category }) {
  return (
    <span
      style={{ background: CATEGORY_COLOR[category] ?? "#888", boxShadow: `0 0 8px ${CATEGORY_COLOR[category] ?? "#888"}66` }}
      className="inline-block w-1.5 h-1.5 rounded-full mr-1.5 align-middle"
    />
  );
}

function ScoreBar({ score }) {
  const pct = Math.max(0, Math.min(1, score)) * 100;
  const color = score >= 0.82 ? "#22D3C8" : score >= 0.6 ? "#F5A623" : "#5A6478";
  return (
    <div className="w-full h-1.5 rounded-full bg-[#1B2333] overflow-hidden">
      <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

function CopyableId({ id }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => { navigator.clipboard?.writeText(id); setCopied(true); setTimeout(() => setCopied(false), 1200); }}
      className="inline-flex items-center gap-1 font-mono text-[12px] px-1.5 py-0.5 rounded border border-[#2A3344] bg-[#10151F] text-[#C7CEDB] hover:border-[#3D4B66] transition-colors"
      title="Copy ID"
    >
      {id}
      {copied ? <Check size={11} className="text-[#22D3C8]" /> : <Copy size={11} className="text-[#5A6478]" />}
    </button>
  );
}

/* ───────────────────────────────────────────────────────────────────────────
   HEADER STAT CARD
─────────────────────────────────────────────────────────────────────────── */

function StatCard({ icon: Icon, label, value, sub }) {
  return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-lg border border-[#1F2738] bg-[#0E1320]">
      <div className="w-9 h-9 rounded-md bg-[#161D2E] flex items-center justify-center shrink-0">
        <Icon size={16} className="text-[#6FE3D8]" />
      </div>
      <div className="min-w-0">
        <div className="text-[20px] leading-none font-semibold text-[#EAEEF6] font-mono">{value}</div>
        <div className="text-[11px] text-[#6B7689] mt-1 truncate">{label}{sub ? <span className="text-[#4A5468]"> · {sub}</span> : null}</div>
      </div>
    </div>
  );
}

/* ───────────────────────────────────────────────────────────────────────────
   BUILD PICKER (left rail)
─────────────────────────────────────────────────────────────────────────── */

function BuildRow({ build, active, onClick }) {
  const color = CATEGORY_COLOR[build.failure_category] ?? "#888";
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2.5 rounded-md border transition-all group ${
        active ? "bg-[#141C2C] border-[#2C3A52]" : "bg-transparent border-transparent hover:bg-[#0F1422] hover:border-[#1B2333]"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[12px] text-[#9AA5B8]">{build.build_id}</span>
        <span style={{ color }} className="text-[10px] uppercase tracking-wider font-medium">
          <CategoryDot category={build.failure_category} />{build.failure_category}
        </span>
      </div>
      <div className="text-[12.5px] text-[#D3D9E5] mt-1 leading-snug line-clamp-2">{build.failure_summary}</div>
      <div className="text-[10.5px] text-[#535D72] mt-1 font-mono">{build.environment}</div>
    </button>
  );
}

/* ───────────────────────────────────────────────────────────────────────────
   VERDICT CARD — the signature element of this dashboard
─────────────────────────────────────────────────────────────────────────── */

function VerdictCard({ verdict, loading }) {
  if (loading) {
    return (
      <div className="rounded-xl border border-[#1F2738] bg-[#0E1320] p-10 flex flex-col items-center justify-center gap-3 min-h-[420px]">
        <Loader2 size={28} className="animate-spin text-[#3D4B66]" />
        <div className="text-[13px] text-[#5A6478] font-mono">running 4-prompt reasoning chain…</div>
      </div>
    );
  }

  if (!verdict) {
    return (
      <div className="rounded-xl border border-dashed border-[#222B3D] bg-[#0B0F17] p-10 flex flex-col items-center justify-center gap-3 min-h-[420px] text-center">
        <Terminal size={26} className="text-[#2C3650]" />
        <div className="text-[13px] text-[#5A6478] max-w-xs">
          Select a Jenkins build from the left, or paste a raw console log, then run triage.
        </div>
      </div>
    );
  }

  const meta = ACTION_META[verdict.action];
  const Icon = meta.Icon;

  return (
    <div className="rounded-xl border bg-[#0E1320] overflow-hidden" style={{ borderColor: meta.border }}>
      {/* Verdict header strip */}
      <div className="px-6 py-5 flex items-center gap-4" style={{ background: meta.bg }}>
        <div className="w-11 h-11 rounded-full flex items-center justify-center shrink-0" style={{ background: `${meta.color}1A`, border: `1px solid ${meta.color}55` }}>
          <Icon size={20} style={{ color: meta.color }} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-[17px] font-semibold" style={{ color: meta.color }}>{meta.label}</div>
          <div className="text-[12px] text-[#9AA5B8] mt-0.5">{meta.blurb}</div>
        </div>
        {verdict._mock && (
          <span className="text-[10px] uppercase tracking-wider px-2 py-1 rounded border border-[#3D4B66] text-[#7C8AA3] font-mono shrink-0">mock data</span>
        )}
      </div>

      <div className="p-6 space-y-5">
        {/* Build / bug fields */}
        <div>
          <div className="text-[10.5px] uppercase tracking-wider text-[#5A6478] mb-2">Extracted bug description</div>
          <div className="rounded-lg border border-[#1B2333] bg-[#0A0E16] p-4 space-y-2">
            <Row label="Title" value={verdict.bug_fields.bug_title} mono={false} />
            <Row label="Component" value={verdict.bug_fields.component} />
            <Row label="Category">
              <span style={{ color: CATEGORY_COLOR[verdict.failure_category] }} className="text-[12.5px] font-medium">
                <CategoryDot category={verdict.failure_category} />{verdict.failure_category}
              </span>
            </Row>
            <Row label="Error">
              <code className="text-[11.5px] text-[#D3D9E5] font-mono break-all">{verdict.bug_fields.error_message}</code>
            </Row>
          </div>
        </div>

        {/* Similarity + matched ticket */}
        {verdict.top_match_ticket_id ? (
          <div>
            <div className="text-[10.5px] uppercase tracking-wider text-[#5A6478] mb-2">Matched Jira ticket</div>
            <div className="rounded-lg border border-[#1B2333] bg-[#0A0E16] p-4 space-y-3">
              <div className="flex items-center justify-between">
                <CopyableId id={verdict.top_match_ticket_id} />
                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-[#5A6478]">similarity</span>
                  <span className="font-mono text-[13px] text-[#EAEEF6]">{verdict.top_match_score.toFixed(3)}</span>
                </div>
              </div>
              <ScoreBar score={verdict.top_match_score} />
              {verdict.matched_ticket && (
                <div className="pt-1 flex items-center justify-between text-[12px]">
                  <span className="text-[#9AA5B8] truncate pr-3">{verdict.matched_ticket.summary}</span>
                  <span className="shrink-0 px-2 py-0.5 rounded-full text-[10.5px] font-medium border border-[#2A3344] text-[#C7CEDB]">
                    {verdict.matched_ticket.status}
                  </span>
                </div>
              )}
              <div className="text-[11.5px] text-[#6B7689] pt-1 border-t border-[#161D2E]">
                {verdict.reason} <span className="text-[#3D4B66]">·</span> confidence {(verdict.confidence * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        ) : (
          <div className="rounded-lg border border-[#1B2333] bg-[#0A0E16] p-4 text-[12px] text-[#6B7689]">
            No candidate ticket scored above the duplicate threshold (0.82). {verdict.reason}
          </div>
        )}

        {/* Workaround steps */}
        {verdict.workaround_steps?.length > 0 && (
          <div>
            <div className="text-[10.5px] uppercase tracking-wider text-[#5A6478] mb-2">Workaround steps</div>
            <ol className="space-y-2">
              {verdict.workaround_steps.map((step, i) => (
                <li key={i} className="flex gap-3 text-[12.5px] text-[#D3D9E5]">
                  <span className="shrink-0 w-5 h-5 rounded-full bg-[#15291F] border border-[#1F4A39] text-[#22D3C8] text-[10.5px] font-mono flex items-center justify-center mt-0.5">{i + 1}</span>
                  <span className="leading-snug pt-0.5">{step}</span>
                </li>
              ))}
            </ol>
          </div>
        )}

        {/* Footer meta */}
        <div className="flex items-center justify-between pt-3 border-t border-[#161D2E] text-[11px] text-[#4A5468] font-mono">
          <span className="flex items-center gap-1.5"><Clock size={11} />{verdict.elapsed_seconds}s</span>
          <span>build {verdict.build_id}</span>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value, mono = true, children }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <span className="text-[11px] text-[#5A6478] shrink-0 pt-0.5">{label}</span>
      <span className={`text-right text-[12.5px] text-[#D3D9E5] ${mono ? "font-mono" : ""}`}>{children ?? value}</span>
    </div>
  );
}

/* ───────────────────────────────────────────────────────────────────────────
   CATEGORY DISTRIBUTION BAR (small viz, header)
─────────────────────────────────────────────────────────────────────────── */

function CategoryDistribution({ dist }) {
  const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
  return (
    <div className="rounded-lg border border-[#1F2738] bg-[#0E1320] px-4 py-3">
      <div className="text-[11px] text-[#6B7689] mb-2">Failure mix · {total} builds</div>
      <div className="flex h-2 rounded-full overflow-hidden mb-2">
        {Object.entries(dist).map(([cat, count]) => (
          <div key={cat} style={{ width: `${(count / total) * 100}%`, background: CATEGORY_COLOR[cat] }} />
        ))}
      </div>
      <div className="flex gap-4">
        {Object.entries(dist).map(([cat, count]) => (
          <div key={cat} className="text-[11px] text-[#9AA5B8] flex items-center">
            <CategoryDot category={cat} />{cat} <span className="text-[#4A5468] ml-1">{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ───────────────────────────────────────────────────────────────────────────
   MAIN APP
─────────────────────────────────────────────────────────────────────────── */

export default function App() {
  const [builds, setBuilds] = useState([]);
  const [stats, setStats] = useState(null);
  const [selectedBuild, setSelectedBuild] = useState(null);
  const [verdict, setVerdict] = useState(null);
  const [loading, setLoading] = useState(false);
  const [usingMock, setUsingMock] = useState(false);
  const [pastedLog, setPastedLog] = useState("");
  const [mode, setMode] = useState("builds"); // "builds" | "paste"
  const [apiError, setApiError] = useState(null);

  const loadFromApi = useCallback(async () => {
    try {
      const [bRes, sRes] = await Promise.all([
        fetch(`${API_BASE}/builds?limit=30`),
        fetch(`${API_BASE}/stats`),
      ]);
      if (!bRes.ok || !sRes.ok) throw new Error("API not reachable");
      setBuilds(await bRes.json());
      setStats(await sRes.json());
      setUsingMock(false);
      setApiError(null);
    } catch (e) {
      setBuilds(MOCK_BUILDS);
      setStats(MOCK_STATS);
      setUsingMock(true);
      setApiError(e.message);
    }
  }, []);

  useEffect(() => { loadFromApi(); }, [loadFromApi]);

  const runTriage = async (build) => {
    setSelectedBuild(build);
    setVerdict(null);
    setLoading(true);
    try {
      if (usingMock) {
        await new Promise((r) => setTimeout(r, 650));
        setVerdict(buildMockVerdict(build));
      } else {
        const res = await fetch(`${API_BASE}/triage`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ build_id: build.build_id }),
        });
        if (!res.ok) throw new Error(await res.text());
        setVerdict(await res.json());
      }
    } catch (e) {
      setVerdict({ _error: e.message });
    } finally {
      setLoading(false);
    }
  };

  const runTriageOnPaste = async () => {
    if (!pastedLog.trim()) return;
    setSelectedBuild({ build_id: "ad-hoc", failure_summary: pastedLog.slice(0, 80) });
    setVerdict(null);
    setLoading(true);
    try {
      if (usingMock) {
        await new Promise((r) => setTimeout(r, 650));
        setVerdict(buildMockVerdict({ build_id: "AD-HOC", job_name: "manual-paste", failure_category: "product", failure_summary: pastedLog.slice(0, 80) }));
      } else {
        const res = await fetch(`${API_BASE}/triage`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ raw_log_text: pastedLog }),
        });
        if (!res.ok) throw new Error(await res.text());
        setVerdict(await res.json());
      }
    } catch (e) {
      setVerdict({ _error: e.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#070A10] text-[#D3D9E5]" style={{ fontFamily: "'Inter', system-ui, sans-serif" }}>
      {/* Top bar */}
      <div className="border-b border-[#161D2E] px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-md bg-gradient-to-br from-[#22D3C8] to-[#1B8E84] flex items-center justify-center">
            <Layers size={16} className="text-[#04201D]" />
          </div>
          <div>
            <div className="text-[14.5px] font-semibold text-[#EAEEF6] leading-none">Bug Triage Console</div>
            <div className="text-[11px] text-[#5A6478] mt-0.5">RAG-based duplicate detection · Dell PowerStore SAN</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {usingMock && (
            <span className="flex items-center gap-1.5 text-[11px] text-[#F5A623] border border-[#3A2E12] bg-[#1A1408] px-2.5 py-1 rounded-full">
              <ServerCog size={11} /> API offline — showing mock data
            </span>
          )}
          <span className="flex items-center gap-1.5 text-[11px] text-[#22D3C8] border border-[#123330] bg-[#08130F] px-2.5 py-1 rounded-full">
            <Activity size={11} /> Qwen3-8B · BGE-large · Qdrant
          </span>
        </div>
      </div>

      {/* Stats row */}
      {stats && (
        <div className="px-6 py-4 grid grid-cols-2 lg:grid-cols-4 gap-3 border-b border-[#161D2E]">
          <StatCard icon={GitBranch} label="Jenkins builds indexed" value={stats.total_jenkins_builds} />
          <StatCard icon={Database} label="Jira tickets in Qdrant" value={stats.total_jira_tickets} />
          <div className="col-span-2 lg:col-span-2">
            <CategoryDistribution dist={stats.failure_category_distribution} />
          </div>
        </div>
      )}

      {/* Main layout */}
      <div className="grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-0">
        {/* Left rail */}
        <div className="border-r border-[#161D2E] p-4 lg:h-[calc(100vh-178px)] lg:overflow-y-auto">
          <div className="flex gap-1 mb-3 p-1 rounded-lg bg-[#0E1320] border border-[#1B2333]">
            <button
              onClick={() => setMode("builds")}
              className={`flex-1 text-[12px] py-1.5 rounded-md transition-colors ${mode === "builds" ? "bg-[#1B2333] text-[#EAEEF6]" : "text-[#6B7689]"}`}
            >
              Recent builds
            </button>
            <button
              onClick={() => setMode("paste")}
              className={`flex-1 text-[12px] py-1.5 rounded-md transition-colors ${mode === "paste" ? "bg-[#1B2333] text-[#EAEEF6]" : "text-[#6B7689]"}`}
            >
              Paste log
            </button>
          </div>

          {mode === "builds" ? (
            <div className="space-y-1.5">
              {builds.map((b) => (
                <BuildRow key={b.build_id} build={b} active={selectedBuild?.build_id === b.build_id} onClick={() => runTriage(b)} />
              ))}
            </div>
          ) : (
            <div className="space-y-3">
              <textarea
                value={pastedLog}
                onChange={(e) => setPastedLog(e.target.value)}
                placeholder="Paste a raw Jenkins console log here…"
                className="w-full h-48 rounded-md bg-[#0A0E16] border border-[#1B2333] text-[12px] font-mono text-[#D3D9E5] p-3 resize-none focus:outline-none focus:border-[#3D4B66] placeholder:text-[#3D4760]"
              />
              <button
                onClick={runTriageOnPaste}
                disabled={!pastedLog.trim() || loading}
                className="w-full flex items-center justify-center gap-2 rounded-md bg-[#22D3C8] text-[#04201D] text-[12.5px] font-medium py-2.5 hover:bg-[#3EE2D8] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <Search size={14} /> Run triage
              </button>
            </div>
          )}
        </div>

        {/* Right — verdict */}
        <div className="p-6 max-w-2xl">
          <VerdictCard verdict={verdict?._error ? null : verdict} loading={loading} />
          {verdict?._error && (
            <div className="mt-4 rounded-lg border border-[#3A1A1A] bg-[#160B0B] p-4 text-[12.5px] text-[#FB5B5B]">
              Triage failed: {verdict._error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
