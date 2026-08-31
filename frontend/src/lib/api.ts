/**
 * VEREC frontend API client.
 *
 * Thin wrapper around the FastAPI backend (backend/server.py):
 *   - REST:      /api/presets, /api/system, /api/report, /api/export/all
 *   - WebSocket: /ws/feed  (live annotated frames + AI results)
 *
 * The backend host defaults to the same hostname the UI is served from on
 * port 8000, so it works both at http://localhost:3000 and over the LAN.
 * Override with NEXT_PUBLIC_API_URL if you run the backend elsewhere.
 */

/* ── Base URLs ───────────────────────────────────────────────────────── */

function httpBase(): string {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window !== "undefined") {
    return `http://${window.location.hostname}:8000`;
  }
  return "http://localhost:8000";
}

function wsBase(): string {
  return httpBase().replace(/^http/, "ws");
}

/* ── Types ───────────────────────────────────────────────────────────── */

export interface DetectionObject {
  class: string;
  confidence: number;
  box?: number[];
}

export interface DetectionPayload {
  objects: DetectionObject[];
  count: number;
  time_ms: number;
  fps: number;
}

export interface PosePayload {
  persons?: unknown[];
  count: number;
  time_ms: number;
  fps?: number;
}

export interface ActionPrediction {
  label: string;
  confidence: number;
}

export interface ActionPayload {
  actions: ActionPrediction[];
  time_ms?: number;
}

/** One message frame pushed over the /ws/feed WebSocket. */
export type Severity = "info" | "low" | "medium" | "high" | "critical";

export interface ThreatAlert {
  id: string;
  timestamp: string;
  type: string;
  label: string;
  category: string;
  severity: Severity;
  score: number;
  rule: string;
  detail: string;
  evidence?: Record<string, unknown>;
  snapshot?: string | null;
  acknowledged?: boolean;
  /** Set once an operator has sent this alert on to officers. */
  dispatched_at?: string;
  dispatched_to?: string[];
}

export interface ThreatStats {
  total: number;
  by_severity: Record<string, number>;
  by_type: Record<string, number>;
  active_tracks: number;
}

export interface ThreatConfig {
  enabled: boolean;
  min_severity: Severity;
  cooldown_s: number;
  min_score: number;
  weapon_conf: number;
  loiter_seconds: number;
  unattended_seconds: number;
  crowd_person_threshold: number;
  caption_screening: boolean;
  after_hours_enabled: boolean;
  zones: { name: string; polygon: number[][]; severity?: Severity }[];
  [key: string]: unknown;
}

export interface FeedFrame {
  timestamp: string;
  raw_frame?: string; // base64 JPEG
  ai_frame?: string; // base64 JPEG (with overlays)
  frame_bytes?: number;
  caption?: string;
  detection?: DetectionPayload;
  object_counts?: Record<string, number>;
  pose?: PosePayload;
  action?: ActionPayload;
  vlm?: { text: string; time_s?: number; tokens?: number; tokens_per_s?: number };
  alerts?: ThreatAlert[];
  threat?: ThreatStats;
  error?: string;
  /** Lifecycle events for file playback: "video_opened" | "loop" | "eof" */
  event?: string;
  video?: VideoMeta;
  progress?: PlaybackProgress;
}

/* ── Local video files ───────────────────────────────────────────────── */

export type SourceKind = "local" | "ip" | "file";

export interface VideoMeta {
  name: string;
  path: string;
  size_bytes?: number;
  fps: number;
  frames: number;
  duration_s: number;
  width?: number;
  height?: number;
}

export interface PlaybackProgress {
  frame: number;
  total: number;
  percent: number;
  time_s?: number;
  duration_s?: number;
}

export interface ReportResult {
  timestamp: string;
  model: string;
  report: string;
  detection_count: number;
  caption_count: number;
}

export interface SystemInfo {
  platform: string;
  machine: string;
  python: string;
  device: string;
  local_ip?: string;
  models?: {
    detector?: string;
    pose?: string;
    action?: string;
    vlm?: string;
    llm?: string;
  };
  uptime_s?: number;
  log_counts?: {
    detections: number;
    captions: number;
    reports: number;
  };
  log_dir?: string;
}

/* ── REST calls ──────────────────────────────────────────────────────── */

export async function fetchPresets(): Promise<Record<string, string>> {
  const r = await fetch(`${httpBase()}/api/presets`);
  if (!r.ok) throw new Error(`presets: HTTP ${r.status}`);
  return r.json();
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const r = await fetch(`${httpBase()}/api/system`);
  if (!r.ok) throw new Error(`system: HTTP ${r.status}`);
  return r.json();
}

export async function fetchReport(): Promise<ReportResult> {
  const r = await fetch(`${httpBase()}/api/report`, { method: "POST" });
  if (!r.ok) throw new Error(`report: HTTP ${r.status}`);
  return r.json();
}

export async function fetchExportAll(): Promise<object> {
  const r = await fetch(`${httpBase()}/api/export/all`);
  if (!r.ok) throw new Error(`export: HTTP ${r.status}`);
  return r.json();
}

/* ── Threat / alerts ─────────────────────────────────────────────────── */

export async function fetchAlerts(
  limit = 100,
  minSeverity?: Severity,
): Promise<{ count: number; alerts: ThreatAlert[]; stats: ThreatStats }> {
  const qs = new URLSearchParams({ limit: String(limit) });
  if (minSeverity) qs.set("min_severity", minSeverity);
  const r = await fetch(`${httpBase()}/api/alerts?${qs}`);
  if (!r.ok) throw new Error(`alerts: HTTP ${r.status}`);
  return r.json();
}

export async function acknowledgeAlert(id: string): Promise<void> {
  await fetch(`${httpBase()}/api/alerts/${id}/ack`, { method: "POST" });
}

/* ── Dispatch: sending an alert to officers ──────────────────────────── */

export interface Camera {
  name: string;
  address?: string;
  area?: string | null;
  lat?: number | null;
  lng?: number | null;
  police_post?: string | null;
}

export interface Recipient {
  name: string;
  areas: string[];
  email?: string;
  whatsapp?: string;
  min_severity: Severity;
}

export interface DispatchResult {
  mode: string;
  sent: boolean;
  reason?: string;
  delivered?: number;
  attempted?: number;
  results?: {
    channel: string;
    to: string;
    sent: boolean;
    reason?: string;
  }[];
  recipients: Recipient[];
  preview: {
    subject: string;
    text: string;
    camera: Camera & { id: string };
    snapshot_name?: string | null;
  };
}

export async function fetchCameras(): Promise<Record<string, Camera>> {
  const r = await fetch(`${httpBase()}/api/cameras`);
  if (!r.ok) throw new Error(`cameras: HTTP ${r.status}`);
  return r.json();
}

export async function fetchRecipients(): Promise<{
  recipients: Recipient[];
  mode: string;
  email_ready: boolean;
  whatsapp_ready: boolean;
}> {
  const r = await fetch(`${httpBase()}/api/recipients`);
  if (!r.ok) throw new Error(`recipients: HTTP ${r.status}`);
  return r.json();
}

export async function saveRecipients(
  recipients: Recipient[],
): Promise<{ recipients: Recipient[]; warnings: string[] }> {
  const r = await fetch(`${httpBase()}/api/recipients`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ recipients }),
  });
  if (!r.ok) throw new Error(`save recipients: HTTP ${r.status}`);
  return r.json();
}

/** Who would be told and what it would say. Sends nothing. */
export async function previewDispatch(
  id: string,
  cameraId: string,
): Promise<DispatchResult> {
  const qs = new URLSearchParams({ camera_id: cameraId });
  const r = await fetch(`${httpBase()}/api/alerts/${id}/dispatch?${qs}`);
  if (!r.ok) throw new Error(`dispatch preview: HTTP ${r.status}`);
  return r.json();
}

/** Actually send. `confirmedBy` records who took responsibility. */
export async function sendDispatch(
  id: string,
  cameraId: string,
  confirmedBy: string,
): Promise<DispatchResult> {
  const r = await fetch(`${httpBase()}/api/alerts/${id}/dispatch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      camera_id: cameraId,
      confirmed_by: confirmedBy,
      force: true,
    }),
  });
  if (!r.ok) throw new Error(`dispatch: HTTP ${r.status}`);
  return r.json();
}

export async function clearAlerts(): Promise<void> {
  await fetch(`${httpBase()}/api/alerts/clear`, { method: "POST" });
}

export function snapshotUrl(snapshot: string): string {
  // Backend stores paths as "alerts/<name>.jpg"
  const name = snapshot.replace(/^alerts\//, "");
  return `${httpBase()}/api/alerts/snapshot/${name}`;
}

export async function fetchThreatConfig(): Promise<ThreatConfig> {
  const r = await fetch(`${httpBase()}/api/threat/config`);
  if (!r.ok) throw new Error(`threat config: HTTP ${r.status}`);
  return r.json();
}

export async function updateThreatConfig(
  patch: Partial<ThreatConfig>,
): Promise<ThreatConfig> {
  const r = await fetch(`${httpBase()}/api/threat/config`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!r.ok) throw new Error(`threat config: HTTP ${r.status}`);
  return r.json();
}

/* ── Client-side JSON download ───────────────────────────────────────── */

export function downloadJson(data: unknown, filename: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* ── Local video files ───────────────────────────────────────────────── */

export async function fetchVideos(): Promise<{ videos: VideoMeta[]; dir: string }> {
  const r = await fetch(`${httpBase()}/api/videos`);
  if (!r.ok) throw new Error("Failed to list videos");
  return r.json();
}

/**
 * Upload a video file to the backend's `videos/` directory.
 * Uses XHR rather than fetch so we can report upload progress.
 */
export function uploadVideo(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<VideoMeta> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${httpBase()}/api/videos/upload`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.((e.loaded / e.total) * 100);
    };

    xhr.onload = () => {
      let body: unknown;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        reject(new Error(`Upload failed (${xhr.status})`));
        return;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as VideoMeta);
      } else {
        const detail = (body as { detail?: string })?.detail;
        reject(new Error(detail || `Upload failed (${xhr.status})`));
      }
    };

    xhr.onerror = () => reject(new Error("Upload failed: network error"));
    xhr.send(form);
  });
}

export async function deleteVideo(name: string): Promise<void> {
  const r = await fetch(`${httpBase()}/api/videos/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!r.ok) throw new Error("Failed to delete video");
}

export async function probeVideo(path: string): Promise<VideoMeta> {
  const r = await fetch(
    `${httpBase()}/api/videos/probe?path=${encodeURIComponent(path)}`,
  );
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || "Could not read that video path");
  }
  return r.json();
}

/* ── WebSocket feed ──────────────────────────────────────────────────── */

export interface FeedParams {
  source: SourceKind;
  url: string;
  loop?: boolean;
  conf: number;
  iou: number;
  vlm_interval: number;
  enable_det: boolean;
  enable_vlm: boolean;
  enable_pose: boolean;
  enable_threat?: boolean;
}

export interface FeedCallbacks {
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (message: string) => void;
  onMessage?: (data: FeedFrame) => void;
}

export interface ManagedSocket {
  close: () => void;
  /** Send a control message (settings toggle, seek, loop) to the backend. */
  send: (msg: Record<string, unknown>) => void;
}

/**
 * Open a managed WebSocket to /ws/feed. Returns a handle whose close()
 * cleanly tells the backend to stop and tears the socket down without
 * firing further callbacks.
 */
export function createManagedFeedSocket(
  params: FeedParams,
  callbacks: FeedCallbacks,
): ManagedSocket {
  const qs = new URLSearchParams({
    source: params.source,
    url: params.url ?? "",
    conf: String(params.conf),
    iou: String(params.iou),
    vlm_interval: String(params.vlm_interval),
    enable_det: String(params.enable_det),
    enable_vlm: String(params.enable_vlm),
    enable_pose: String(params.enable_pose),
    enable_threat: String(params.enable_threat ?? true),
    loop: String(params.loop ?? false),
  });

  let closed = false;
  let ws: WebSocket;

  try {
    ws = new WebSocket(`${wsBase()}/ws/feed?${qs.toString()}`);
  } catch (e) {
    callbacks.onError?.(`Failed to open socket: ${String(e)}`);
    return { close: () => {}, send: () => {} };
  }

  ws.onopen = () => {
    if (closed) return;
    callbacks.onOpen?.();
  };

  ws.onmessage = (evt) => {
    if (closed) return;
    let data: FeedFrame;
    try {
      data = JSON.parse(evt.data as string);
    } catch {
      return;
    }
    if (data.error) {
      callbacks.onError?.(data.error);
      return;
    }
    callbacks.onMessage?.(data);
  };

  ws.onerror = () => {
    if (closed) return;
    callbacks.onError?.("WebSocket connection error");
  };

  ws.onclose = () => {
    if (closed) return;
    callbacks.onClose?.();
  };

  return {
    send: (msg: Record<string, unknown>) => {
      if (closed) return;
      try {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
      } catch {
        /* socket went away */
      }
    },
    close: () => {
      closed = true;
      try {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: "stop" }));
        }
        ws.close();
      } catch {
        /* already closing */
      }
    },
  };
}
