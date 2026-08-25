"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  createManagedFeedSocket,
  fetchPresets,
  fetchReport,
  fetchExportAll,
  fetchSystemInfo,
  downloadJson,
  acknowledgeAlert,
  clearAlerts,
  snapshotUrl,
  fetchVideos,
  uploadVideo,
  deleteVideo,
  type ManagedSocket,
  type SourceKind,
  type VideoMeta,
  type PlaybackProgress,
  type FeedFrame,
  type DetectionObject,
  type ReportResult,
  type SystemInfo,
  type ThreatAlert,
  type ThreatStats,
  type Severity,
} from "@/lib/api";

/* ── helpers ─────────────────────────────────────────────────────────── */

function fmtTime(s: number) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return [h, m, sec].map((v) => String(v).padStart(2, "0")).join(":");
}

function fmtBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1048576).toFixed(1)} MB`;
}

/* ── Icon components ─────────────────────────────────────────────────── */

function GearIcon({ className = "w-5 h-5" }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={1.5}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 010 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 010-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28z"
      />
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
      />
    </svg>
  );
}

function Spinner({ className = "w-5 h-5" }: { className?: string }) {
  return (
    <svg
      className={`animate-spin ${className}`}
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

/* ── Markdown prose wrapper ──────────────────────────────────────────── */

function MdReport({ text }: { text: string }) {
  return (
    <div
      className="prose prose-invert prose-xs max-w-none
      prose-headings:text-cyan-400 prose-headings:text-xs prose-headings:font-bold prose-headings:mb-1 prose-headings:mt-2
      prose-p:text-[11px] prose-p:text-slate-300 prose-p:leading-relaxed prose-p:mb-1
      prose-li:text-[11px] prose-li:text-slate-300 prose-li:leading-relaxed
      prose-strong:text-slate-200 prose-ul:my-1 prose-ol:my-1"
    >
      <ReactMarkdown>{text}</ReactMarkdown>
    </div>
  );
}

/* ── Generic expandable card ─────────────────────────────────────────── */

/* ── Threat alert presentation ───────────────────────────────────────── */

const SEVERITY_STYLE: Record<
  Severity,
  { chip: string; border: string; dot: string; label: string }
> = {
  critical: {
    chip: "bg-red-500/20 text-red-300 border-red-500/40",
    border: "border-red-500/50",
    dot: "bg-red-500",
    label: "CRITICAL",
  },
  // Severity colours are semantic, not brand — they deliberately survive the
  // cyan rebrand. `high` stays orange (the old accent colour) precisely because
  // the chrome no longer competes with it, and `low` moved off sky-500, which
  // is now too close to the cyan brand to read as a distinct severity.
  high: {
    chip: "bg-orange-500/20 text-orange-300 border-orange-500/40",
    border: "border-orange-500/40",
    dot: "bg-orange-500",
    label: "HIGH",
  },
  medium: {
    chip: "bg-yellow-500/20 text-yellow-300 border-yellow-500/40",
    border: "border-yellow-500/30",
    dot: "bg-yellow-500",
    label: "MEDIUM",
  },
  low: {
    chip: "bg-blue-500/20 text-blue-300 border-blue-500/40",
    border: "border-blue-500/30",
    dot: "bg-blue-500",
    label: "LOW",
  },
  info: {
    chip: "bg-slate-500/20 text-slate-300 border-slate-500/40",
    border: "border-slate-600/40",
    dot: "bg-slate-500",
    label: "INFO",
  },
};

function AlertCard({
  alert,
  onAck,
}: {
  alert: ThreatAlert;
  onAck: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const s = SEVERITY_STYLE[alert.severity] ?? SEVERITY_STYLE.info;
  const time = new Date(alert.timestamp).toLocaleTimeString();

  return (
    <div
      className={`rounded-xl border bg-slate-800/50 overflow-hidden ${s.border} ${
        alert.acknowledged ? "opacity-50" : ""
      }`}
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-slate-800/80 transition-colors"
      >
        <span
          className={`w-2 h-2 rounded-full shrink-0 ${s.dot} ${
            alert.severity === "critical" && !alert.acknowledged
              ? "animate-pulse"
              : ""
          }`}
        />
        <span
          className={`text-[9px] font-semibold px-1.5 py-0.5 rounded border shrink-0 ${s.chip}`}
        >
          {s.label}
        </span>
        <span className="text-[11px] text-slate-200 font-medium truncate flex-1">
          {alert.label}
        </span>
        <span className="text-[10px] text-slate-500 font-mono shrink-0">
          {time}
        </span>
        <span className="text-slate-500 text-xs shrink-0">
          {open ? "▾" : "▸"}
        </span>
      </button>

      {open && (
        <div className="px-3 pb-3 border-t border-slate-700/30 space-y-2">
          <p className="text-[11px] text-slate-300 pt-2">{alert.detail}</p>

          {alert.snapshot && (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img
              src={snapshotUrl(alert.snapshot)}
              alt={`Evidence for ${alert.label}`}
              className="rounded-lg border border-slate-700/50 w-full"
            />
          )}

          <div className="flex flex-wrap gap-3 text-[9px] text-slate-600 font-mono">
            <span>score {alert.score}</span>
            <span>rule {alert.rule}</span>
            <span>{alert.category}</span>
            <span>id {alert.id}</span>
          </div>

          <p className="text-[9px] text-slate-600 italic">
            Automated signal for human review — not a determination that an
            offence occurred.
          </p>

          {!alert.acknowledged && (
            <button
              onClick={() => onAck(alert.id)}
              className="text-[10px] px-2 py-1 rounded-lg bg-slate-700/60 hover:bg-slate-700 text-slate-300 transition-colors"
            >
              Acknowledge
            </button>
          )}
        </div>
      )}
    </div>
  );
}

interface ExpandableCardProps {
  timestamp: string;
  preview: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
  meta?: React.ReactNode;
}

function ExpandableCard({
  timestamp,
  preview,
  defaultOpen,
  children,
  meta,
}: ExpandableCardProps) {
  const [open, setOpen] = useState(defaultOpen ?? false);
  const time = new Date(timestamp).toLocaleTimeString();
  return (
    <div className="bg-slate-800/50 border border-slate-700/40 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-slate-800/80 transition-colors"
      >
        <div className="min-w-0 flex-1">
          <span className="text-[10px] text-slate-500 font-mono mr-2">
            {time}
          </span>
          <span className="text-[11px] text-slate-400 truncate">
            {preview.slice(0, 80)}
            {preview.length > 80 ? "…" : ""}
          </span>
        </div>
        <span className="text-slate-500 text-xs ml-2 shrink-0">
          {open ? "▾" : "▸"}
        </span>
      </button>
      {open && (
        <div className="px-3 pb-3 border-t border-slate-700/30">
          {children}
          {meta && (
            <div className="flex gap-3 mt-2 text-[9px] text-slate-600 font-mono">
              {meta}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Main page ───────────────────────────────────────────────────────── */

export default function Home() {
  /* Feed state */
  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [aiFrame, setAiFrame] = useState<string | null>(null);
  const [rawFrame, setRawFrame] = useState<string | null>(null);
  const [caption, setCaption] = useState("");
  const [captions, setCaptions] = useState<
    { text: string; timestamp: string }[]
  >([]);
  const [captionSearch, setCaptionSearch] = useState("");
  const [detectionInfo, setDetectionInfo] = useState("");
  const [log, setLog] = useState<string[]>([]);
  const [logSearch, setLogSearch] = useState("");
  const [objectCounts, setObjectCounts] = useState<Record<string, number>>({});
  const [currentAction, setCurrentAction] = useState<
    { label: string; confidence: number }[] | null
  >(null);
  const [poseInfo, setPoseInfo] = useState("");
  const wsRef = useRef<ManagedSocket | null>(null);
  const [wsError, setWsError] = useState<string | null>(null);

  /* Start dropdown */
  const [startOpen, setStartOpen] = useState(false);
  const [quickUrl, setQuickUrl] = useState("");

  /* Global search */
  const [globalSearch, setGlobalSearch] = useState("");
  const [globalSearchOpen, setGlobalSearchOpen] = useState(false);

  /* Bandwidth */
  const [bandwidth, setBandwidth] = useState(0);
  const bwAccum = useRef(0);
  const bwTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  /* Timer */
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* Settings panel */
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [source, setSource] = useState<SourceKind>("local");
  const [presets, setPresets] = useState<Record<string, string>>({});
  const [selectedPreset, setSelectedPreset] = useState("");
  const [streamUrl, setStreamUrl] = useState("");

  /* Video files */
  const [videos, setVideos] = useState<VideoMeta[]>([]);
  const [selectedVideo, setSelectedVideo] = useState("");
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [loopVideo, setLoopVideo] = useState(false);
  const [progress, setProgress] = useState<PlaybackProgress | null>(null);
  const [playbackDone, setPlaybackDone] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [conf, setConf] = useState(0.45);
  const [iou, setIou] = useState(0.45);
  const [vlmInterval, setVlmInterval] = useState(5);
  const [enableDet, setEnableDet] = useState(true);
  const [enableVlm, setEnableVlm] = useState(true);
  const [enablePose, setEnablePose] = useState(true);
  const [enableThreat, setEnableThreat] = useState(true);

  /* Threat alerts */
  const [alerts, setAlerts] = useState<ThreatAlert[]>([]);
  const [threatStats, setThreatStats] = useState<ThreatStats | null>(null);
  const [alertFilter, setAlertFilter] = useState<Severity | "all">("all");
  /* Collapsed by default: an alert strip that grows as incidents arrive steals
     height from the video, and the video resizing mid-incident is worse than
     having to click to read the list. Expansion is user-driven only. */
  const [alertsExpanded, setAlertsExpanded] = useState(false);

  /* Action log */
  const [actionLog, setActionLog] = useState<
    { actions: { label: string; confidence: number }[]; timestamp: string }[]
  >([]);
  const [actionSearch, setActionSearch] = useState("");

  const [autoReportInterval, setAutoReportInterval] = useState(60);

  /* Report */
  const [report, setReport] = useState<ReportResult | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reports, setReports] = useState<ReportResult[]>([]);
  const [reportSearch, setReportSearch] = useState("");
  const autoReportRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* System info */
  const [sysInfo, setSysInfo] = useState<SystemInfo | null>(null);

  /* Tab */
  const [tab, setTab] = useState<"feed" | "export">("feed");
  const [exportData, setExportData] = useState<object | null>(null);

  /* ── Init ──────────────────────────────────────────────────────── */
  useEffect(() => {
    fetchPresets()
      .then(setPresets)
      .catch(() => {});
    fetchSystemInfo()
      .then(setSysInfo)
      .catch(() => {});
    fetchVideos()
      .then((r) => {
        setVideos(r.videos);
        setSelectedVideo((cur) => cur || (r.videos.length ? r.videos[0].path : ""));
      })
      .catch(() => {});
  }, []);

  /* ── Video files ───────────────────────────────────────────────── */
  const handleUpload = useCallback(
    async (file: File) => {
      setUploadError(null);
      setUploadPct(0);
      try {
        const meta = await uploadVideo(file, setUploadPct);
        setVideos((prev) => [...prev, meta]);
        setSelectedVideo(meta.path);
        setSource("file");
      } catch (e) {
        setUploadError(e instanceof Error ? e.message : String(e));
      } finally {
        setUploadPct(null);
      }
    },
    [],
  );

  const handleDeleteVideo = useCallback(
    async (v: VideoMeta) => {
      try {
        await deleteVideo(v.name);
        setVideos((prev) => prev.filter((x) => x.path !== v.path));
        setSelectedVideo((cur) => (cur === v.path ? "" : cur));
      } catch {
        /* ignore */
      }
    },
    [],
  );

  /* ── Bandwidth meter ───────────────────────────────────────────── */
  useEffect(() => {
    bwTimer.current = setInterval(() => {
      setBandwidth(bwAccum.current);
      bwAccum.current = 0;
    }, 1000);
    return () => {
      if (bwTimer.current) clearInterval(bwTimer.current);
    };
  }, []);

  /* ── Auto-report timer ─────────────────────────────────────────── */
  const triggerReport = useCallback(async () => {
    try {
      const r = await fetchReport();
      setReport(r);
      setReports((prev) => [...prev, r].slice(-20));
    } catch (e) {
      const err: ReportResult = {
        timestamp: new Date().toISOString(),
        model: "error",
        report: String(e),
        detection_count: 0,
        caption_count: 0,
      };
      setReport(err);
      setReports((prev) => [...prev, err].slice(-20));
    }
  }, []);

  useEffect(() => {
    if (autoReportRef.current) clearInterval(autoReportRef.current);
    if (connected && autoReportInterval > 0) {
      autoReportRef.current = setInterval(() => {
        triggerReport();
      }, autoReportInterval * 1000);
    }
    return () => {
      if (autoReportRef.current) clearInterval(autoReportRef.current);
    };
  }, [connected, autoReportInterval, triggerReport]);

  /* ── Start / Stop feed ─────────────────────────────────────────── */
  const startFeed = useCallback(() => {
    wsRef.current?.close();
    setConnecting(true);
    setWsError(null);
    setProgress(null);
    setPlaybackDone(false);

    const managed = createManagedFeedSocket(
      {
        source,
        url: source === "file" ? selectedVideo : streamUrl,
        loop: loopVideo,
        conf,
        iou,
        vlm_interval: vlmInterval,
        enable_det: enableDet,
        enable_vlm: enableVlm,
        enable_pose: enablePose,
        enable_threat: enableThreat,
      },
      {
        onOpen() {
          setConnecting(false);
          setConnected(true);
          setWsError(null);
          setElapsed(0);
          timerRef.current = setInterval(() => setElapsed((p) => p + 1), 1000);
        },
        onClose() {
          setConnecting(false);
          setConnected(false);
          if (timerRef.current) clearInterval(timerRef.current);
        },
        onError(msg) {
          setWsError(msg);
        },
        onMessage(data) {
          if (data.progress) setProgress(data.progress);
          if (data.event === "eof") {
            setPlaybackDone(true);
            // A file run ends before the auto-report interval can fire (the
            // socket closes, which clears the timer), so generate the report
            // here — finishing the clip is the natural trigger.
            setReportLoading(true);
            triggerReport().finally(() => setReportLoading(false));
            return;
          }
          if (data.frame_bytes) bwAccum.current += data.frame_bytes;
          if (data.ai_frame)
            setAiFrame(`data:image/jpeg;base64,${data.ai_frame}`);
          if (data.raw_frame)
            setRawFrame(`data:image/jpeg;base64,${data.raw_frame}`);
          if (data.caption) {
            const cap = data.caption;
            setCaption(cap);
            setCaptions((prev) =>
              [...prev, { text: cap, timestamp: data.timestamp }].slice(-50),
            );
          }
          if (data.detection) {
            const d = data.detection;
            const objs = d.objects
              .map(
                (o: DetectionObject) =>
                  `${o.class} (${Math.round(o.confidence * 100)}%)`,
              )
              .join(", ");
            setDetectionInfo(`${d.count} obj | ${d.time_ms}ms | ${d.fps} FPS`);
            setLog((prev) =>
              [
                ...prev,
                `[${new Date(data.timestamp).toLocaleTimeString()}] ${objs || "(clear)"}`,
              ].slice(-50),
            );
          }
          if (data.object_counts) {
            setObjectCounts(data.object_counts);
          }
          if (data.pose) {
            const p = data.pose;
            setPoseInfo(
              `${p.count} person${p.count !== 1 ? "s" : ""} | ${p.time_ms}ms`,
            );
          }
          if (data.action) {
            setCurrentAction(data.action.actions);
            setActionLog((prev) =>
              [
                ...prev,
                { actions: data.action!.actions, timestamp: data.timestamp },
              ].slice(-50),
            );
          }
          if (data.alerts?.length) {
            setAlerts((prev) => [...data.alerts!, ...prev].slice(0, 100));
          }
          if (data.threat) setThreatStats(data.threat);
        },
      },
    );

    wsRef.current = managed;
  }, [
    source,
    streamUrl,
    selectedVideo,
    loopVideo,
    triggerReport,
    conf,
    iou,
    vlmInterval,
    enableDet,
    enableVlm,
    enablePose,
    enableThreat,
  ]);

  /* ── Alert actions ─────────────────────────────────────────────── */
  const handleAckAlert = useCallback(async (id: string) => {
    setAlerts((prev) =>
      prev.map((a) => (a.id === id ? { ...a, acknowledged: true } : a)),
    );
    try {
      await acknowledgeAlert(id);
    } catch {
      /* optimistic update stands; backend retry not worth blocking the UI */
    }
  }, []);

  const handleClearAlerts = useCallback(async () => {
    setAlerts([]);
    setThreatStats(null);
    try {
      await clearAlerts();
    } catch {
      /* ok */
    }
  }, []);

  const visibleAlerts =
    alertFilter === "all"
      ? alerts
      : alerts.filter((a) => a.severity === alertFilter);

  const unackedCritical = alerts.filter(
    (a) => !a.acknowledged && (a.severity === "critical" || a.severity === "high"),
  ).length;

  const stopFeed = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setConnecting(false);
    setConnected(false);
    setWsError(null);
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  useEffect(
    () => () => {
      wsRef.current?.close();
      if (timerRef.current) clearInterval(timerRef.current);
    },
    [],
  );

  /* ── Manual report ─────────────────────────────────────────────── */
  const handleReport = async () => {
    setReportLoading(true);
    await triggerReport();
    setReportLoading(false);
  };

  /* ── Export ────────────────────────────────────────────────────── */
  const handleExport = async () => {
    try {
      const d = await fetchExportAll();
      setExportData(d);
      downloadJson(d, `sentinel-export-${Date.now()}.json`);
    } catch {
      /* ok */
    }
  };

  /* ── bandwidth color ───────────────────────────────────────────── */
  const bwColor =
    bandwidth === 0
      ? "text-slate-500"
      : bandwidth < 500_000
        ? "text-green-400"
        : bandwidth < 2_000_000
          ? "text-yellow-400"
          : "text-red-400";

  /* ── status label ──────────────────────────────────────────────── */
  const statusLabel = connecting ? "Connecting" : connected ? "Live" : "Idle";
  const statusClass = connecting
    ? "bg-yellow-500/20 text-yellow-400 ring-1 ring-yellow-500/30"
    : connected
      ? "bg-green-500/20 text-green-400 ring-1 ring-green-500/30"
      : "bg-slate-800 text-slate-500";

  /* ── Render ────────────────────────────────────────────────────── */
  return (
    <div className="h-screen bg-slate-950 text-slate-100 flex flex-col overflow-hidden">
      {/* Hidden file picker — mounted once at the root so any control
          (settings panel, start dropdown) can trigger it regardless of
          which panels happen to be open. */}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*,.mp4,.mov,.avi,.mkv,.webm,.m4v,.mpg,.mpeg,.wmv,.flv"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleUpload(f);
          e.target.value = "";
        }}
      />

      {/* ═══ Header ═══════════════════════════════════════════════ */}
      <header className="shrink-0 border-b border-slate-800/60 bg-slate-900/80 backdrop-blur sticky top-0 z-30 px-3 sm:px-5 py-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <h1 className="text-xl sm:text-2xl font-black tracking-tight text-cyan-400 shrink-0">
            SENTINEL
          </h1>
          <div className="hidden sm:flex items-center gap-0.5 ml-2">
            {(["feed", "export"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-3 py-1.5 rounded-full text-xs font-semibold transition-colors ${tab === t ? "bg-slate-700/80 text-cyan-400" : "text-slate-500 hover:text-slate-300"}`}
              >
                {t === "feed" ? "Live Feed" : "Export"}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-1.5 shrink-0 text-xs">
          <div
            className={`hidden md:flex items-center gap-1 font-mono px-2 py-1 rounded-full bg-slate-800/80 border border-slate-700/50 ${bwColor}`}
          >
            <span>↕</span> {fmtBytes(bandwidth)}/s
          </div>
          {sysInfo?.device && (
            <span className="hidden lg:inline px-2 py-1 rounded-full bg-slate-800/80 border border-slate-700/50 text-slate-400 font-mono uppercase">
              {sysInfo.device} · {sysInfo.machine}
            </span>
          )}
          {sysInfo?.local_ip && (
            <span className="hidden xl:inline px-2 py-1 rounded-full bg-slate-800/80 border border-slate-700/50 text-blue-400 font-mono text-[10px]">
              {sysInfo.local_ip}
            </span>
          )}
          <div className="font-mono tabular-nums px-2 py-1 rounded-full bg-slate-800/80 border border-slate-700/50">
            <span className={connected ? "text-green-400" : "text-slate-500"}>
              {fmtTime(elapsed)}
            </span>
          </div>
          <span
            className={`px-2 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider flex items-center gap-1 ${statusClass}`}
          >
            {connecting && <Spinner className="w-3 h-3" />}
            {statusLabel}
          </span>
          <button
            onClick={() => setSettingsOpen(!settingsOpen)}
            className={`p-1.5 rounded-full transition-colors ${settingsOpen ? "bg-cyan-600 text-white" : "bg-slate-800 text-slate-400 hover:text-slate-200"}`}
          >
            <GearIcon className="w-4 h-4" />
          </button>
          <button
            onClick={() => setGlobalSearchOpen(!globalSearchOpen)}
            className={`p-1.5 rounded-full transition-colors ${globalSearchOpen ? "bg-cyan-600 text-white" : "bg-slate-800 text-slate-400 hover:text-slate-200"}`}
          >
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z"
              />
            </svg>
          </button>
        </div>
      </header>

      {/* ═══ Global search bar ═══════════════════════════════════ */}
      {globalSearchOpen && (
        <div className="shrink-0 px-3 py-2 bg-slate-900/90 backdrop-blur border-b border-slate-800/40">
          <div className="flex items-center gap-2 max-w-2xl mx-auto">
            <svg
              className="w-4 h-4 text-slate-500 shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z"
              />
            </svg>
            <input
              type="text"
              value={globalSearch}
              onChange={(e) => {
                const v = e.target.value;
                setGlobalSearch(v);
                setCaptionSearch(v);
                setLogSearch(v);
                setReportSearch(v);
              }}
              placeholder="Search all captions, detections, and reports…"
              autoFocus
              className="flex-1 bg-transparent text-slate-200 text-sm placeholder-slate-600 focus:outline-none"
            />
            {globalSearch && (
              <button
                onClick={() => {
                  setGlobalSearch("");
                  setCaptionSearch("");
                  setLogSearch("");
                  setReportSearch("");
                }}
                className="text-slate-500 hover:text-slate-300 text-xs"
              >
                Clear
              </button>
            )}
          </div>
        </div>
      )}

      {/* ═══ Mobile tab bar ══════════════════════════════════════ */}
      <div className="sm:hidden flex border-b border-slate-800/40 bg-slate-900/60">
        {(["feed", "export"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 py-2 text-xs font-semibold text-center ${tab === t ? "text-cyan-400 border-b-2 border-cyan-500" : "text-slate-500"}`}
          >
            {t === "feed" ? "Live Feed" : "Export"}
          </button>
        ))}
      </div>

      {/* ═══ WebSocket error banner ═════════════════════════════ */}
      {wsError && (
        <div className="shrink-0 px-4 py-2 bg-red-900/60 border-b border-red-700/40 flex items-center justify-between gap-2">
          <span className="text-red-300 text-xs font-medium">{wsError}</span>
          <button
            onClick={() => setWsError(null)}
            className="text-red-400 hover:text-red-200 text-xs font-bold"
          >
            &times;
          </button>
        </div>
      )}

      {/* ═══ Settings slide-out panel ════════════════════════════ */}
      {settingsOpen && (
        <div className="absolute top-[52px] right-2 z-40 w-80 max-h-[calc(100vh-60px)] overflow-y-auto bg-slate-900/95 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-2xl p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-200">Settings</h3>
            <button
              onClick={() => setSettingsOpen(false)}
              className="text-slate-500 hover:text-slate-300 text-lg leading-none"
            >
              &times;
            </button>
          </div>

          {/* Source */}
          <div className="space-y-2">
            <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              Source
            </h4>
            <div className="flex gap-2">
              {(["local", "ip", "file"] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setSource(s)}
                  className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${source === s ? "bg-cyan-600 text-white" : "bg-slate-800 text-slate-400 hover:bg-slate-700"}`}
                >
                  {s === "local"
                    ? "Local Camera"
                    : s === "ip"
                      ? "IP / Stream"
                      : "Video File"}
                </button>
              ))}
            </div>
            {source === "file" && (
              <div className="space-y-2">
                {/* Drop zone / picker */}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    const f = e.dataTransfer.files?.[0];
                    if (f) handleUpload(f);
                  }}
                  className="w-full cursor-pointer rounded-xl border border-dashed border-slate-700 hover:border-cyan-500/60 bg-slate-800/40 px-3 py-4 text-center transition-colors"
                >
                  <p className="text-xs text-slate-300">
                    Click to choose a video, or drop one here
                  </p>
                  <p className="text-[10px] text-slate-500 mt-0.5">
                    mp4, mov, avi, mkv, webm
                  </p>
                </div>

                {uploadPct !== null && (
                  <div className="space-y-1">
                    <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
                      <div
                        className="h-full bg-cyan-500 transition-all"
                        style={{ width: `${uploadPct}%` }}
                      />
                    </div>
                    <p className="text-[10px] text-slate-500">
                      Uploading… {Math.round(uploadPct)}%
                    </p>
                  </div>
                )}

                {uploadError && (
                  <p className="text-[10px] text-red-400">{uploadError}</p>
                )}

                {/* Library */}
                {videos.length > 0 && (
                  <div className="space-y-1 max-h-40 overflow-y-auto">
                    {videos.map((v) => (
                      <div
                        key={v.path}
                        onClick={() => setSelectedVideo(v.path)}
                        className={`group flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-lg cursor-pointer text-[11px] transition-colors ${
                          selectedVideo === v.path
                            ? "bg-cyan-600/20 border border-cyan-500/40 text-cyan-200"
                            : "bg-slate-800/60 border border-transparent text-slate-400 hover:bg-slate-700/60"
                        }`}
                      >
                        <span className="truncate flex-1" title={v.name}>
                          {v.name}
                        </span>
                        <span className="text-[9px] text-slate-500 shrink-0">
                          {v.duration_s ? `${Math.round(v.duration_s)}s` : ""}
                          {v.frames ? ` · ${v.frames}f` : ""}
                        </span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteVideo(v);
                          }}
                          className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-red-400 transition-opacity shrink-0"
                          title="Delete"
                        >
                          ✕
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                {/* Or point at a path already on disk */}
                <input
                  type="text"
                  value={selectedVideo}
                  onChange={(e) => setSelectedVideo(e.target.value)}
                  placeholder="…or paste an absolute path: /Users/you/Movies/clip.mp4"
                  className="w-full bg-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2 border border-slate-700/50 focus:outline-none focus:ring-1 focus:ring-cyan-500/50"
                />

                <label className="flex items-center gap-1.5 cursor-pointer text-xs text-slate-400">
                  <input
                    type="checkbox"
                    checked={loopVideo}
                    onChange={(e) => {
                      setLoopVideo(e.target.checked);
                      wsRef.current?.send({ loop: e.target.checked });
                    }}
                    className="accent-cyan-500 w-3.5 h-3.5"
                  />
                  Loop when finished
                </label>
              </div>
            )}
            {source === "ip" && (
              <div className="space-y-2">
                <div className="flex flex-wrap gap-1">
                  {Object.keys(presets).map((k) => (
                    <button
                      key={k}
                      onClick={() => {
                        setSelectedPreset(k);
                        setStreamUrl(presets[k]);
                      }}
                      className={`px-2 py-1 rounded-full text-[10px] font-medium transition-colors ${selectedPreset === k ? "bg-cyan-600 text-white" : "bg-slate-800 text-slate-400 hover:bg-slate-700 border border-slate-700/50"}`}
                    >
                      {k.replace(/ \(.*\)/, "")}
                    </button>
                  ))}
                </div>
                <input
                  type="text"
                  value={streamUrl}
                  onChange={(e) => setStreamUrl(e.target.value)}
                  placeholder="YouTube / RTSP / MJPEG URL"
                  className="w-full bg-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2 border border-slate-700/50 focus:outline-none focus:ring-1 focus:ring-cyan-500/50"
                />
              </div>
            )}
          </div>

          {/* Model toggles */}
          <div className="space-y-2">
            <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              Models
            </h4>
            <div className="flex flex-wrap gap-x-4 gap-y-2 text-xs">
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableDet}
                  onChange={(e) => setEnableDet(e.target.checked)}
                  className="accent-cyan-500 w-3.5 h-3.5"
                />{" "}
                Detection
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableVlm}
                  onChange={(e) => setEnableVlm(e.target.checked)}
                  className="accent-cyan-500 w-3.5 h-3.5"
                />{" "}
                VLM
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enablePose}
                  onChange={(e) => setEnablePose(e.target.checked)}
                  className="accent-purple-500 w-3.5 h-3.5"
                />{" "}
                Pose + Action
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableThreat}
                  onChange={(e) => setEnableThreat(e.target.checked)}
                  className="accent-red-500 w-3.5 h-3.5"
                />{" "}
                Threat alerts
              </label>
            </div>
          </div>

          {/* Sliders */}
          <div className="space-y-2 text-xs text-slate-400">
            <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              Detection
            </h4>
            <label className="block">
              <span className="flex justify-between">
                <span>Confidence</span>
                <span className="text-cyan-400 font-mono">
                  {conf.toFixed(2)}
                </span>
              </span>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={conf}
                onChange={(e) => setConf(+e.target.value)}
                className="w-full accent-cyan-500 mt-1"
              />
            </label>
            <label className="block">
              <span className="flex justify-between">
                <span>IoU</span>
                <span className="text-cyan-400 font-mono">
                  {iou.toFixed(2)}
                </span>
              </span>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={iou}
                onChange={(e) => setIou(+e.target.value)}
                className="w-full accent-cyan-500 mt-1"
              />
            </label>
            <label className="block">
              <span className="flex justify-between">
                <span>VLM Interval</span>
                <span className="text-cyan-400 font-mono">
                  {vlmInterval}s
                </span>
              </span>
              <input
                type="range"
                min="3"
                max="15"
                step="1"
                value={vlmInterval}
                onChange={(e) => setVlmInterval(+e.target.value)}
                className="w-full accent-cyan-500 mt-1"
              />
            </label>
          </div>

          {/* Auto-report */}
          <div className="space-y-2 text-xs text-slate-400">
            <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              Auto Report
            </h4>
            <label className="block">
              <span className="flex justify-between">
                <span>Interval</span>
                <span className="text-cyan-400 font-mono">
                  {autoReportInterval === 0 ? "Off" : `${autoReportInterval}s`}
                </span>
              </span>
              <input
                type="range"
                min="0"
                max="300"
                step="15"
                value={autoReportInterval}
                onChange={(e) => setAutoReportInterval(+e.target.value)}
                className="w-full accent-cyan-500 mt-1"
              />
            </label>
            <p className="text-[10px] text-slate-600">
              0 = manual only. Reports use DeepSeek-R1 via Ollama.
            </p>
          </div>

          {/* System */}
          {sysInfo && (
            <div className="space-y-1 bg-slate-800/40 rounded-xl p-2.5 border border-slate-700/30">
              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                System
              </h4>
              <div className="text-[11px] text-slate-400 space-y-0.5 font-mono">
                <p>
                  {sysInfo.platform} {sysInfo.machine} · Python {sysInfo.python}
                </p>
                <p>
                  Device:{" "}
                  <span className="text-cyan-400">{sysInfo.device}</span>
                </p>
                {sysInfo.local_ip && (
                  <p>
                    Local IP:{" "}
                    <span className="text-blue-400">{sysInfo.local_ip}</span>
                  </p>
                )}
                <p>Detector: {sysInfo.models?.detector ?? "–"}</p>
                <p>
                  Pose:{" "}
                  {(sysInfo.models as Record<string, string>)?.pose ?? "–"}
                </p>
                <p>
                  Action:{" "}
                  {(sysInfo.models as Record<string, string>)?.action ?? "–"}
                </p>
                <p>VLM: {sysInfo.models?.vlm ?? "–"}</p>
                <p>LLM: {sysInfo.models?.llm ?? "–"}</p>
                <p>
                  Logs: {sysInfo.log_counts?.detections ?? 0} det /{" "}
                  {sysInfo.log_counts?.captions ?? 0} cap /{" "}
                  {sysInfo.log_counts?.reports ?? 0} rpt
                </p>
              </div>
            </div>
          )}

          {/* Start/Stop */}
          <div className="flex gap-2 pt-1">
            <button
              onClick={() => {
                startFeed();
                setSettingsOpen(false);
              }}
              disabled={connecting}
              className="flex-1 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-sm font-bold py-2 rounded-xl transition-colors flex items-center justify-center gap-1.5"
            >
              {connecting ? (
                <>
                  <Spinner className="w-4 h-4" /> Connecting…
                </>
              ) : (
                "▶ Start"
              )}
            </button>
            <button
              onClick={stopFeed}
              className="flex-1 bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm font-bold py-2 rounded-xl transition-colors"
            >
              ⏹ Stop
            </button>
          </div>
        </div>
      )}

      {/* ═══ Main content ═══════════════════════════════════════ */}
      <main className="flex-1 overflow-hidden">
        {tab === "feed" && (
          <div className="h-full flex flex-col">
            {/* ── Video area ───────────────────────────────────── */}
            <div className="flex-1 flex flex-col lg:flex-row min-h-0">
              {/* Big AI view */}
              <div className="flex-[3] min-h-0 flex flex-col">
                <div className="relative flex-1 bg-black flex items-center justify-center overflow-hidden rounded-2xl m-1">
                  {aiFrame ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={aiFrame}
                      alt="AI View"
                      className="max-w-full max-h-full object-contain"
                    />
                  ) : (
                    <div className="text-center space-y-3 px-4">
                      {connecting ? (
                        <>
                          <Spinner className="w-10 h-10 text-cyan-500 mx-auto" />
                          <p className="text-slate-400 text-sm">
                            Connecting to stream…
                          </p>
                          <p className="text-slate-600 text-[11px]">
                            Resolving URL and initializing models
                          </p>
                        </>
                      ) : (
                        <>
                          <div className="w-16 h-16 mx-auto rounded-2xl bg-slate-800/60 border border-slate-700/40 flex items-center justify-center">
                            <svg
                              className="w-8 h-8 text-slate-600"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                              strokeWidth={1.5}
                            >
                              <path
                                strokeLinecap="round"
                                d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z"
                              />
                            </svg>
                          </div>
                          <p className="text-slate-500 text-sm">
                            Press{" "}
                            <span className="text-cyan-500 font-semibold">
                              ▶ Start
                            </span>{" "}
                            or open{" "}
                            <GearIcon className="inline w-4 h-4 -mt-0.5 text-slate-400" />{" "}
                            settings
                          </p>
                        </>
                      )}
                    </div>
                  )}
                  {connected && (
                    <span
                      className={`absolute top-2 left-2 flex items-center gap-1.5 px-2 py-0.5 rounded-full text-white text-[10px] font-bold ${
                        source === "file" ? "bg-blue-600/90" : "bg-red-600/90"
                      }`}
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />{" "}
                      {source === "file"
                        ? playbackDone
                          ? "DONE"
                          : "FILE"
                        : "LIVE"}
                    </span>
                  )}
                  {/* Action overlay */}
                  {connected && currentAction && currentAction.length > 0 && (
                    <div className="absolute top-2 left-20 flex gap-1.5">
                      <span className="px-2 py-0.5 rounded-full bg-purple-600/90 backdrop-blur-sm text-white text-[10px] font-bold shadow-lg">
                        🏃 {currentAction[0].label}{" "}
                        <span className="opacity-75">
                          {Math.round(currentAction[0].confidence * 100)}%
                        </span>
                      </span>
                    </div>
                  )}
                  {/* Playback progress (video files only) */}
                  {source === "file" && progress && progress.total > 0 && (
                    <div className="absolute bottom-9 inset-x-0 px-3 space-y-1">
                      <div
                        className="h-1.5 w-full rounded-full bg-white/20 overflow-hidden cursor-pointer"
                        onClick={(e) => {
                          const r = e.currentTarget.getBoundingClientRect();
                          const frac = (e.clientX - r.left) / r.width;
                          wsRef.current?.send({ action: "seek", position: frac });
                          setPlaybackDone(false);
                        }}
                      >
                        <div
                          className="h-full bg-cyan-500 transition-[width] duration-150"
                          style={{ width: `${progress.percent}%` }}
                        />
                      </div>
                      <div className="flex justify-between text-[10px] font-mono text-slate-300">
                        <span>
                          {progress.frame} / {progress.total} frames
                          {playbackDone ? " · finished" : ""}
                        </span>
                        <span>
                          {progress.duration_s
                            ? `${fmtTime(Math.round(progress.time_s ?? 0))} / ${fmtTime(Math.round(progress.duration_s))}`
                            : `${progress.percent.toFixed(1)}%`}
                        </span>
                      </div>
                    </div>
                  )}
                  {/* Stats overlay */}
                  <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent px-3 py-2 flex items-end justify-between text-[11px] font-mono">
                    <span className="text-slate-300">
                      {detectionInfo || "Waiting…"}
                    </span>
                    <span className={bwColor}>{fmtBytes(bandwidth)}/s</span>
                  </div>
                  {/* Floating Start / Stop controls */}
                  <div className="absolute top-2 right-2 z-10">
                    {!connected && !connecting && (
                      <div className="relative">
                        <div className="flex">
                          <button
                            onClick={() => {
                              // Nothing to play yet — open the picker instead
                              // of failing with an empty path.
                              if (source === "file" && !selectedVideo) {
                                fileInputRef.current?.click();
                                return;
                              }
                              startFeed();
                            }}
                            className="bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold px-3 py-1.5 rounded-l-full shadow-lg transition-colors"
                          >
                            {source === "file" && !selectedVideo
                              ? "📁 Choose Video"
                              : "▶ Start"}
                          </button>
                          <button
                            onClick={() => setStartOpen(!startOpen)}
                            className="bg-cyan-700 hover:bg-cyan-600 text-white text-xs font-bold px-2 py-1.5 rounded-r-full shadow-lg transition-colors border-l border-cyan-500/40"
                          >
                            {startOpen ? "▴" : "▾"}
                          </button>
                        </div>
                        {startOpen && (
                          <div className="absolute top-full right-0 mt-1.5 w-72 bg-slate-900/95 backdrop-blur-xl border border-slate-700/50 rounded-xl shadow-2xl p-3 space-y-2">
                            <div className="flex gap-1">
                              {(["local", "ip", "file"] as const).map((s) => (
                                <button
                                  key={s}
                                  onClick={() => setSource(s)}
                                  className={`px-2 py-1 rounded-full text-[10px] font-medium transition-colors ${source === s ? "bg-cyan-600 text-white" : "bg-slate-800 text-slate-400 hover:bg-slate-700"}`}
                                >
                                  {s === "local"
                                    ? "Camera"
                                    : s === "ip"
                                      ? "Stream"
                                      : "File"}
                                </button>
                              ))}
                            </div>
                            {source === "ip" && (
                              <>
                                <input
                                  type="text"
                                  value={quickUrl}
                                  onChange={(e) => {
                                    setQuickUrl(e.target.value);
                                    setStreamUrl(e.target.value);
                                  }}
                                  placeholder="Paste YouTube / RTSP / MJPEG URL"
                                  className="w-full bg-slate-800 text-slate-200 text-xs rounded-lg px-3 py-2 border border-slate-700/50 focus:outline-none focus:ring-1 focus:ring-cyan-500/50"
                                />
                                <div className="flex flex-wrap gap-1">
                                  {Object.keys(presets).map((k) => (
                                    <button
                                      key={k}
                                      onClick={() => {
                                        setSelectedPreset(k);
                                        setStreamUrl(presets[k]);
                                        setQuickUrl(presets[k]);
                                      }}
                                      className={`px-2 py-0.5 rounded-full text-[9px] font-medium transition-colors ${selectedPreset === k ? "bg-cyan-600 text-white" : "bg-slate-800 text-slate-400 hover:bg-slate-700 border border-slate-700/50"}`}
                                    >
                                      {k.replace(/ \(.*\)/, "")}
                                    </button>
                                  ))}
                                </div>
                              </>
                            )}
                            {source === "file" && (
                              <>
                                <button
                                  onClick={() => fileInputRef.current?.click()}
                                  className="w-full flex items-center justify-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium py-2 rounded-lg border border-dashed border-slate-600 hover:border-cyan-500/60 transition-colors"
                                >
                                  📁 Choose video file…
                                </button>

                                {uploadPct !== null && (
                                  <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
                                    <div
                                      className="h-full bg-cyan-500 transition-all"
                                      style={{ width: `${uploadPct}%` }}
                                    />
                                  </div>
                                )}
                                {uploadError && (
                                  <p className="text-[10px] text-red-400">
                                    {uploadError}
                                  </p>
                                )}

                                {videos.length > 0 && (
                                  <div className="space-y-1 max-h-32 overflow-y-auto">
                                    {videos.map((v) => (
                                      <button
                                        key={v.path}
                                        onClick={() => setSelectedVideo(v.path)}
                                        className={`w-full text-left truncate px-2 py-1 rounded-lg text-[10px] transition-colors ${
                                          selectedVideo === v.path
                                            ? "bg-cyan-600/20 border border-cyan-500/40 text-cyan-200"
                                            : "bg-slate-800/60 border border-transparent text-slate-400 hover:bg-slate-700/60"
                                        }`}
                                        title={v.path}
                                      >
                                        {v.name}
                                      </button>
                                    ))}
                                  </div>
                                )}

                                <input
                                  type="text"
                                  value={selectedVideo}
                                  onChange={(e) => setSelectedVideo(e.target.value)}
                                  placeholder="…or paste a full path"
                                  className="w-full bg-slate-800 text-slate-200 text-xs rounded-lg px-3 py-2 border border-slate-700/50 focus:outline-none focus:ring-1 focus:ring-cyan-500/50"
                                />
                              </>
                            )}
                            <button
                              onClick={() => {
                                startFeed();
                                setStartOpen(false);
                              }}
                              disabled={source === "file" && !selectedVideo}
                              className="w-full bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-white text-xs font-bold py-1.5 rounded-lg transition-colors"
                            >
                              {source === "file" ? "▶ Analyze Video" : "▶ Start Stream"}
                            </button>
                          </div>
                        )}
                      </div>
                    )}
                    {connecting && (
                      <span className="bg-yellow-600/90 text-white text-xs font-bold px-3 py-1.5 rounded-full shadow-lg flex items-center gap-1">
                        <Spinner className="w-3 h-3" /> Connecting…
                      </span>
                    )}
                    {connected && (
                      <button
                        onClick={stopFeed}
                        className="bg-slate-700/90 hover:bg-slate-600 text-slate-200 text-xs font-bold px-3 py-1.5 rounded-full shadow-lg transition-colors"
                      >
                        ⏹ Stop
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Right sidebar: raw feed + captions.
                  min-h-0 + overflow-hidden keep it inside the video row: without
                  them the thumbnail's fixed aspect ratio makes the column taller
                  than its parent and the captions spill over the panel below. */}
              <div className="hidden lg:flex lg:w-72 xl:w-80 shrink-0 min-h-0 overflow-hidden flex-col bg-slate-900/60 border-l border-slate-800/40">
                <div className="aspect-video max-h-[45%] bg-black flex items-center justify-center overflow-hidden shrink-0 rounded-2xl m-1">
                  {rawFrame ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={rawFrame}
                      alt="Raw Feed"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <span className="text-slate-700 text-[10px]">Raw Feed</span>
                  )}
                </div>
                <div className="px-3 py-2 border-t border-slate-800/40 flex-1 min-h-0 overflow-y-auto flex flex-col">
                  <div className="flex items-center justify-between mb-2 shrink-0">
                    <h4 className="text-[10px] font-semibold uppercase tracking-wider text-cyan-400">
                      Scene Captions
                    </h4>
                    <span className="text-[8px] text-slate-600 font-mono">
                      {captions.length}
                    </span>
                  </div>
                  {captions.length > 3 && (
                    <input
                      type="text"
                      value={captionSearch}
                      onChange={(e) => setCaptionSearch(e.target.value)}
                      placeholder="Search captions…"
                      className="w-full mb-1.5 bg-slate-800/60 text-slate-300 text-[11px] rounded-lg px-2 py-1 border border-slate-700/40 focus:outline-none focus:ring-1 focus:ring-cyan-500/40"
                    />
                  )}
                  {captions.length === 0 ? (
                    <p className="text-[11px] text-slate-600 italic">
                      {connected || connecting
                        ? "Waiting for captions…"
                        : "No captions yet."}
                    </p>
                  ) : (
                    <div className="space-y-1.5 overflow-y-auto max-h-[50vh]">
                      {[...captions]
                        .reverse()
                        .filter(
                          (c) =>
                            !captionSearch ||
                            c.text
                              .toLowerCase()
                              .includes(captionSearch.toLowerCase()),
                        )
                        .map((c, i) => (
                          <ExpandableCard
                            key={`${c.timestamp}-${i}`}
                            timestamp={c.timestamp}
                            preview={c.text}
                            defaultOpen={i === 0}
                          >
                            {c.text.split("\n").map((l, j) => (
                              <p
                                key={j}
                                className="text-[11px] text-slate-300 leading-relaxed"
                              >
                                {l}
                              </p>
                            ))}
                          </ExpandableCard>
                        ))}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* ── Incident alerts ─────────────────────────────────
                Height is fixed per state — a slim ticker when collapsed, a
                fixed 28vh when expanded — so incoming alerts scroll *within*
                the strip instead of growing it and squeezing the video. */}
            <div
              className={`shrink-0 border-t flex flex-col overflow-hidden transition-[height] duration-200 ${
                unackedCritical > 0
                  ? "border-red-500/40 bg-red-950/10"
                  : "border-slate-800/40"
              }`}
              style={{ height: alertsExpanded ? "28vh" : "2.25rem" }}
            >
              {/* Header — always visible, always the same height */}
              <div className="flex items-center justify-between gap-2 px-3 h-9 shrink-0">
                <button
                  onClick={() => setAlertsExpanded((v) => !v)}
                  className="flex items-center gap-2 min-w-0 flex-1 text-left group"
                  aria-expanded={alertsExpanded}
                  title={alertsExpanded ? "Collapse alerts" : "Expand alerts"}
                >
                  <span className="text-[10px] text-slate-500 group-hover:text-slate-300 transition-colors shrink-0">
                    {alertsExpanded ? "▾" : "▸"}
                  </span>
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 group-hover:text-slate-300 transition-colors shrink-0">
                    Incident Alerts
                  </h4>
                  {unackedCritical > 0 && (
                    <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded-full bg-red-500/20 text-red-300 border border-red-500/40 animate-pulse shrink-0">
                      {unackedCritical} needs review
                    </span>
                  )}
                  {/* Collapsed: the newest alert reads as a one-line ticker, so
                      nothing is missed without opening the panel. */}
                  {!alertsExpanded && visibleAlerts.length > 0 && (
                    <span className="text-[10px] text-slate-400 truncate min-w-0">
                      <span className="text-slate-600 font-mono">
                        {visibleAlerts.length}
                      </span>{" "}
                      · latest:{" "}
                      <span
                        className={
                          visibleAlerts[0].severity === "critical"
                            ? "text-red-300"
                            : "text-slate-300"
                        }
                      >
                        {visibleAlerts[0].label}
                      </span>
                    </span>
                  )}
                  {alertsExpanded && threatStats && (
                    <span className="text-[9px] text-slate-600 font-mono truncate">
                      {threatStats.total} total · {threatStats.active_tracks}{" "}
                      tracked
                    </span>
                  )}
                </button>

                <div className="flex items-center gap-1.5 shrink-0">
                  {alertsExpanded && (
                    <>
                      <select
                        value={alertFilter}
                        onChange={(e) =>
                          setAlertFilter(e.target.value as Severity | "all")
                        }
                        className="bg-slate-800/60 text-slate-400 text-[10px] rounded-lg px-1.5 py-0.5 border border-slate-700/40 focus:outline-none"
                      >
                        <option value="all">All</option>
                        <option value="critical">Critical</option>
                        <option value="high">High</option>
                        <option value="medium">Medium</option>
                        <option value="low">Low</option>
                      </select>
                      <button
                        onClick={handleClearAlerts}
                        className="text-[10px] px-2 py-0.5 rounded-lg bg-slate-800/60 hover:bg-slate-700 text-slate-400 border border-slate-700/40 transition-colors"
                      >
                        Clear
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* List — scrolls inside the fixed strip */}
              {alertsExpanded && (
                <div className="flex-1 min-h-0 overflow-y-auto px-3 pb-2">
                  {visibleAlerts.length === 0 ? (
                    <p className="text-[11px] text-slate-600 italic">
                      {connected
                        ? enableThreat
                          ? "Monitoring — no incidents flagged."
                          : "Threat detection is off."
                        : "No alerts."}
                    </p>
                  ) : (
                    <div className="space-y-1.5">
                      {visibleAlerts.map((a) => (
                        <AlertCard key={a.id} alert={a} onAck={handleAckAlert} />
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ── Bottom panel — independent scroll columns ──── */}
            <div
              className="shrink-0 border-t border-slate-800/40 flex flex-col md:flex-row"
              style={{ height: "40vh" }}
            >
              {/* Mobile: caption cards (hidden on lg, shown above in sidebar) */}
              <div className="lg:hidden px-3 py-2 border-b md:border-b-0 md:border-r border-slate-800/30 min-w-0 flex-1 overflow-y-auto">
                <div className="flex items-center justify-between mb-1">
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-slate-600">
                    Scene Captions
                  </h4>
                  <span className="text-[9px] text-slate-600 font-mono">
                    {captions.length}
                  </span>
                </div>
                {captions.length > 3 && (
                  <input
                    type="text"
                    value={captionSearch}
                    onChange={(e) => setCaptionSearch(e.target.value)}
                    placeholder="Search captions…"
                    className="w-full mb-1.5 bg-slate-800/60 text-slate-300 text-[11px] rounded-lg px-2 py-1 border border-slate-700/40 focus:outline-none focus:ring-1 focus:ring-cyan-500/40"
                  />
                )}
                {captions.length === 0 ? (
                  <p className="text-[11px] text-slate-600 italic">
                    {connected || connecting
                      ? "Waiting for captions…"
                      : "No captions yet."}
                  </p>
                ) : (
                  <div className="space-y-1.5">
                    {[...captions]
                      .reverse()
                      .filter(
                        (c) =>
                          !captionSearch ||
                          c.text
                            .toLowerCase()
                            .includes(captionSearch.toLowerCase()),
                      )
                      .map((c, i) => (
                        <ExpandableCard
                          key={`${c.timestamp}-${i}`}
                          timestamp={c.timestamp}
                          preview={c.text}
                          defaultOpen={i === 0}
                        >
                          <p className="text-[11px] text-slate-300 leading-relaxed">
                            {c.text}
                          </p>
                        </ExpandableCard>
                      ))}
                  </div>
                )}
              </div>

              {/* Detections — independent scroll */}
              <div className="flex-1 p-3 md:border-r border-slate-800/30 min-w-0 flex flex-col overflow-hidden">
                <div className="flex items-center justify-between mb-1.5 shrink-0">
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-cyan-400">
                    Detections
                  </h4>
                  <span className="text-[9px] text-slate-600 font-mono">
                    {log.length}
                  </span>
                </div>
                {/* Object frequency stats */}
                {Object.keys(objectCounts).length > 0 && (
                  <div className="flex flex-wrap gap-1 mb-1.5 shrink-0">
                    {Object.entries(objectCounts)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 8)
                      .map(([cls, count]) => (
                        <span
                          key={cls}
                          className="px-1.5 py-0.5 rounded-full bg-slate-800/60 border border-slate-700/40 text-[9px] font-mono text-slate-400"
                        >
                          {cls} <span className="text-cyan-400">{count}</span>
                        </span>
                      ))}
                    {Object.keys(objectCounts).length > 8 && (
                      <span className="text-[9px] text-slate-600">
                        +{Object.keys(objectCounts).length - 8} more
                      </span>
                    )}
                  </div>
                )}
                {log.length > 3 && (
                  <input
                    type="text"
                    value={logSearch}
                    onChange={(e) => setLogSearch(e.target.value)}
                    placeholder="Search detections…"
                    className="w-full mb-1.5 shrink-0 bg-slate-800/60 text-slate-300 text-[11px] rounded-lg px-2 py-1 border border-slate-700/40 focus:outline-none focus:ring-1 focus:ring-cyan-500/40"
                  />
                )}
                <div className="flex-1 overflow-y-auto">
                  {log.length === 0 && !connected && !connecting ? (
                    <p className="text-[11px] text-slate-600 italic">
                      No detections yet.
                    </p>
                  ) : log.length === 0 && (connected || connecting) ? (
                    <div className="flex items-center gap-2 text-[11px] text-slate-500">
                      <Spinner className="w-3.5 h-3.5" /> Waiting for
                      detections…
                    </div>
                  ) : (
                    <pre className="text-[11px] text-slate-400 font-mono whitespace-pre-wrap leading-relaxed">
                      {(logSearch
                        ? log.filter((l) =>
                            l.toLowerCase().includes(logSearch.toLowerCase()),
                          )
                        : log
                      ).join("\n")}
                    </pre>
                  )}
                </div>
              </div>

              {/* Actions — independent scroll */}
              <div className="flex-1 p-3 md:border-r border-slate-800/30 min-w-0 flex flex-col overflow-hidden">
                <div className="flex items-center justify-between mb-1.5 shrink-0">
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-purple-400">
                    Actions
                  </h4>
                  <span className="text-[9px] text-slate-600 font-mono">
                    {actionLog.length}
                  </span>
                </div>
                {/* Current action highlight */}
                {currentAction && currentAction.length > 0 && (
                  <div className="flex flex-wrap gap-1 mb-1.5 shrink-0">
                    <span className="text-[9px] text-purple-400 uppercase tracking-wider font-semibold mr-1 self-center">
                      Now
                    </span>
                    {currentAction.map((a, i) => (
                      <span
                        key={`bottom-${a.label}-${i}`}
                        className={`px-1.5 py-0.5 rounded-full text-[9px] font-mono border ${
                          i === 0
                            ? "bg-purple-900/40 border-purple-600/50 text-purple-300"
                            : "bg-slate-800/60 border-slate-700/40 text-slate-400"
                        }`}
                      >
                        {a.label}{" "}
                        <span
                          className={
                            i === 0 ? "text-purple-400" : "text-slate-500"
                          }
                        >
                          {Math.round(a.confidence * 100)}%
                        </span>
                      </span>
                    ))}
                  </div>
                )}
                {poseInfo && (
                  <div className="text-[9px] text-slate-500 font-mono mb-1 shrink-0">
                    🦴 {poseInfo}
                  </div>
                )}
                {actionLog.length > 3 && (
                  <input
                    type="text"
                    value={actionSearch}
                    onChange={(e) => setActionSearch(e.target.value)}
                    placeholder="Search actions…"
                    className="w-full mb-1.5 shrink-0 bg-slate-800/60 text-slate-300 text-[11px] rounded-lg px-2 py-1 border border-purple-700/40 focus:outline-none focus:ring-1 focus:ring-purple-500/40"
                  />
                )}
                <div className="flex-1 overflow-y-auto space-y-1">
                  {actionLog.length === 0 ? (
                    <p className="text-[11px] text-slate-600 italic">
                      {connected || connecting
                        ? enablePose
                          ? "Accumulating pose data (100 frames)…"
                          : "Pose + Action disabled in settings"
                        : "No actions captured yet."}
                    </p>
                  ) : (
                    [...actionLog]
                      .reverse()
                      .filter(
                        (a) =>
                          !actionSearch ||
                          a.actions.some((act) =>
                            act.label
                              .toLowerCase()
                              .includes(actionSearch.toLowerCase()),
                          ),
                      )
                      .map((a, i) => (
                        <div
                          key={`bl-${a.timestamp}-${i}`}
                          className="flex items-center gap-2 text-[11px]"
                        >
                          <span className="text-[9px] text-slate-600 font-mono shrink-0">
                            {new Date(a.timestamp).toLocaleTimeString()}
                          </span>
                          <div className="flex flex-wrap gap-1">
                            {a.actions.map((act, j) => (
                              <span
                                key={`${act.label}-${j}`}
                                className={`px-1.5 py-0.5 rounded-full text-[9px] font-mono ${
                                  j === 0
                                    ? "bg-purple-900/30 text-purple-300"
                                    : "text-slate-500"
                                }`}
                              >
                                {act.label} {Math.round(act.confidence * 100)}%
                              </span>
                            ))}
                          </div>
                        </div>
                      ))
                  )}
                </div>
              </div>

              {/* Report panel — independent scroll */}
              <div className="flex-1 p-3 min-w-0 flex flex-col overflow-hidden">
                <div className="flex items-center justify-between shrink-0">
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-slate-600">
                    LLM Reports
                  </h4>
                  <div className="flex items-center gap-2">
                    {autoReportInterval > 0 && connected && (
                      <span className="text-[9px] text-green-500/80 font-mono">
                        auto {autoReportInterval}s
                      </span>
                    )}
                    <span className="text-[9px] text-slate-600 font-mono">
                      {reports.length}
                    </span>
                  </div>
                </div>
                <button
                  onClick={handleReport}
                  disabled={reportLoading}
                  className="w-full mt-1.5 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-xs font-bold py-1.5 rounded-xl transition-colors shrink-0 flex items-center justify-center gap-1.5"
                >
                  {reportLoading ? (
                    <>
                      <Spinner className="w-3.5 h-3.5" /> Generating…
                    </>
                  ) : (
                    "Generate Report"
                  )}
                </button>

                {reports.length > 3 && (
                  <input
                    type="text"
                    value={reportSearch}
                    onChange={(e) => setReportSearch(e.target.value)}
                    placeholder="Search reports…"
                    className="w-full mt-1.5 shrink-0 bg-slate-800/60 text-slate-300 text-[11px] rounded-lg px-2 py-1 border border-slate-700/40 focus:outline-none focus:ring-1 focus:ring-cyan-500/40"
                  />
                )}

                {/* All reports as scrollable cards — latest expanded */}
                <div className="flex-1 overflow-y-auto mt-1.5 space-y-1.5">
                  {reports.length > 0 ? (
                    [...reports]
                      .reverse()
                      .filter(
                        (r) =>
                          !reportSearch ||
                          r.report
                            .toLowerCase()
                            .includes(reportSearch.toLowerCase()),
                      )
                      .map((r, i) => (
                        <ExpandableCard
                          key={`${r.timestamp}-${i}`}
                          timestamp={r.timestamp}
                          preview={
                            r.report
                              .split("\n")
                              .find((l) => l.trim().length > 0) ?? ""
                          }
                          defaultOpen={i === 0}
                          meta={
                            <>
                              <span>{r.model}</span>
                              <span>{r.detection_count} det</span>
                              <span>{r.caption_count} cap</span>
                            </>
                          }
                        >
                          <MdReport text={r.report} />
                        </ExpandableCard>
                      ))
                  ) : (
                    <p className="text-[11px] text-slate-600 italic">
                      No reports yet.
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ─── Export tab ─────────────────────────────────────────── */}
        {tab === "export" && (
          <div className="p-4 sm:p-6 space-y-4 overflow-y-auto h-full">
            <h2 className="text-lg font-bold text-slate-200">Data Export</h2>
            <p className="text-sm text-slate-400">
              All data is automatically saved to{" "}
              <code className="text-cyan-400/80">logs/</code> on disk.
              Download a combined JSON snapshot below.
            </p>
            <button
              onClick={handleExport}
              className="bg-cyan-600 hover:bg-cyan-500 text-white font-bold px-6 py-2.5 rounded-xl transition-colors"
            >
              Download All Data (JSON)
            </button>
            {sysInfo && (
              <div className="text-xs text-slate-500 font-mono">
                On disk: {sysInfo.log_counts?.detections ?? 0} detections ·{" "}
                {sysInfo.log_counts?.captions ?? 0} captions ·{" "}
                {sysInfo.log_counts?.reports ?? 0} reports
              </div>
            )}
            {exportData && (
              <div className="bg-slate-900 border border-slate-700/40 rounded-2xl p-4 max-h-80 overflow-auto">
                <pre className="text-xs text-slate-300 font-mono whitespace-pre-wrap">
                  {JSON.stringify(exportData, null, 2).slice(0, 5000)}
                  {JSON.stringify(exportData, null, 2).length > 5000
                    ? "\n…(truncated)"
                    : ""}
                </pre>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
