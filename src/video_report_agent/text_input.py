"""Import local transcripts or public YouTube captions into the report pipeline."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .ingest import MAX_VIDEO_SECONDS, UrlIngestError
from .storage import Store

MAX_TEXT_BYTES = 10 * 1024 * 1024
FORMATS = {".txt", ".md", ".srt", ".vtt"}


def youtube_url(value: str) -> tuple[str, str]:
    """Validate one HTTPS watch, short or shorts URL without network access."""
    try:
        parsed = urlsplit(value.strip())
        if (
            parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or parsed.port is not None
            or parsed.fragment
        ):
            raise ValueError
        query = parse_qs(parsed.query, keep_blank_values=True)
        if parsed.hostname == "youtu.be":
            video_id = parsed.path.removeprefix("/")
        elif parsed.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            if parsed.path == "/watch" and len(query.get("v", [])) == 1:
                video_id = query["v"][0]
            elif re.fullmatch(r"/shorts/[\w-]{11}/?", parsed.path):
                video_id = parsed.path.split("/")[2]
            else:
                raise ValueError
        else:
            raise ValueError
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError
    except ValueError:
        raise ValueError("需要公开 YouTube 单视频 HTTPS 链接。") from None
    return video_id, f"https://www.youtube.com/watch?v={video_id}"


def read_text(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > MAX_TEXT_BYTES:
        raise ValueError("转写文件必须存在，且不超过 10 MiB。")
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip() or "\x00" in text:
        raise ValueError("转写文件为空或不是有效文本。")
    return text


def _timestamp(value: str, suffix: str) -> int:
    separator = r"\." if suffix == ".vtt" else ","
    hours = r"(?:(\d{2,}):)?" if suffix == ".vtt" else r"(\d{2,}):"
    match = re.fullmatch(hours + r"([0-5]\d):([0-5]\d)" + separator + r"(\d{3})", value)
    if not match:
        raise ValueError("字幕时间格式无效。")
    h, m, seconds, ms = match.groups()
    return ((int(h or 0) * 60 + int(m)) * 60 + int(seconds)) * 1000 + int(ms)


def normalize_text(text: str, suffix: str) -> str:
    """Assign stable source IDs. Do not invent times for plain text."""
    if suffix not in FORMATS:
        raise ValueError("支持 UTF-8 TXT、Markdown、SRT 和 VTT 文件。")
    units = []
    if suffix in {".srt", ".vtt"}:
        blocks = re.split(r"\n[ \t]*\n", text.replace("\r\n", "\n").strip())
        if suffix == ".vtt":
            if not blocks or not re.match(r"^WEBVTT(?:[ \t]|$)", blocks[0].splitlines()[0]):
                raise ValueError("VTT 文件缺少 WEBVTT 标记。")
            blocks = blocks[1:]
        previous = -1
        for block in blocks:
            lines = block.splitlines()
            if not lines:
                continue
            if suffix == ".vtt" and (
                re.match(r"^NOTE(?:[ \t]|$)", lines[0]) or lines[0] in {"STYLE", "REGION"}
            ):
                continue
            timing_index = 0 if "-->" in lines[0] else 1
            if len(lines) <= timing_index:
                raise ValueError("字幕缺少时间行。")
            match = re.fullmatch(r"(\S+)[ \t]+-->[ \t]+(\S+)(?:[ \t]+.*)?", lines[timing_index])
            if not match:
                raise ValueError("字幕时间行无效。")
            start, end = (_timestamp(value, suffix) for value in match.groups())
            if end <= start or start < previous or end > MAX_VIDEO_SECONDS * 1000:
                raise ValueError("字幕时间必须有效、按起点排序，且不超过 5 小时。")
            previous = start
            content = "\n".join(lines[timing_index + 1 :])
            if suffix == ".vtt":
                content = re.sub(r"<(?:\d{2,}:)?\d{2}:\d{2}\.\d{3}>", "", content)
                content = re.sub(r"<v[ \t]+([^>]+)>", r"\1: ", content)
            content = html.unescape(
                re.sub(r"</?(?:b|i|u|c|v|lang|ruby|rt)(?:[ .\t][^>]*)?>", "", content)
            ).strip()
            if content:
                units.append((f"{start / 1000:.3f}–{end / 1000:.3f}s", content))
    else:
        units = [("时间未知", p.strip()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not units:
        raise ValueError("没有可用的字幕或转写正文。")
    return (
        "\n\n".join(
            f"[text-{i:05d} | {timing}] {content}" for i, (timing, content) in enumerate(units, 1)
        )
        + "\n"
    )


def create_text_run(
    root: Path,
    *,
    path: Path | None = None,
    url: str | None = None,
    language="en",
    automatic=False,
    title=None,
    report_mode="standard",
    transcript_only=False,
    database_path=None,
) -> Path:
    if (path is None) == (url is None):
        raise ValueError("请选择一个本地转写文件或 YouTube 链接。")
    if report_mode not in {"standard", "brief"}:
        raise ValueError("report_mode must be standard or brief")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", language):
        raise ValueError("字幕语言必须是单个语言代码，例如 en 或 zh-Hans。")
    raw = transcript = None
    if path is not None:
        raw = read_text(path)
        transcript = normalize_text(raw, path.suffix.lower())
        identity = "text-" + hashlib.sha256(raw.encode()).hexdigest()
        kind, canonical = "local_text", ""
        title = title or path.stem
    else:
        video_id, canonical = youtube_url(url)
        identity, kind = "youtube-" + video_id, "youtube_captions"
    run = root.resolve() / uuid.uuid4().hex
    metadata = {
        "video_id": identity,
        "url": canonical,
        "input_kind": kind,
        "title": title,
        "report_mode": report_mode,
        "transcript_only": transcript_only,
        "caption_language": language,
        "automatic_captions": automatic,
        "transcript_mode": "prepared",
        "visual_enhancement": False,
        "term_correction": False,
        "diagrams": False,
    }
    status = {"run_id": run.name, "state": "QUEUED", "stage": "QUEUED", "started_at": time.time()}
    store = Store(database_path or root / "state.sqlite3")
    with store.transaction() as db:
        store.mark_writes_open(db=db)
        store.register(run, metadata, status, db=db)
        if transcript is not None:
            store.document(run.name, "transcript.md", transcript, db=db)
            store.document(run.name, "original-transcript" + path.suffix.lower(), raw, db=db)
    store.export(run.name)
    return run


def _sanitize_info(path: Path, destination: Path) -> dict:
    """Remove transport metadata before success or failure can persist it."""
    if not path.is_file():
        return {}
    try:
        info = json.loads(read_text(path))
        if not isinstance(info, dict):
            raise ValueError("Invalid metadata")
    except (ValueError, UnicodeError):
        path.unlink(missing_ok=True)
        return {}
    safe = {k: info.get(k) for k in ("id", "title", "uploader", "description", "duration")}
    destination.write_text(json.dumps(safe, ensure_ascii=False))
    path.unlink(missing_ok=True)
    return safe


def fetch_youtube_captions(run: Path, metadata: dict) -> dict:
    """Download one caption track, with no cookies, media or ASR fallback."""
    _, url = youtube_url(metadata["url"])
    executable = shutil.which("yt-dlp")
    if executable is None:
        raise UrlIngestError("CAPTION_TOOL_UNAVAILABLE", "yt-dlp is unavailable")
    target = run / "download"
    target.mkdir(exist_ok=True)
    command = [
        executable,
        "--ignore-config",
        "--no-playlist",
        "--skip-download",
        "--write-auto-subs" if metadata["automatic_captions"] else "--write-subs",
        "--sub-langs",
        metadata["caption_language"],
        "--sub-format",
        "vtt",
        "--write-info-json",
        "--socket-timeout",
        "15",
        "--retries",
        "0",
        "--extractor-retries",
        "0",
        "--no-progress",
        "-o",
        str(target / "raw-source.%(ext)s"),
        url,
    ]
    info_path = target / "source.info.json"
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise UrlIngestError("CAPTION_TIMEOUT", "字幕获取超时；可改用本地转写文件。") from None
    finally:
        info = _sanitize_info(target / "raw-source.info.json", info_path)
    # Omit signed URLs and third-party error content from persisted diagnostics.
    (run / "download.log").write_text(f"YouTube caption command exit={result.returncode}\n")
    if result.returncode:
        raise UrlIngestError("CAPTION_FETCH_FAILED", "YouTube 字幕获取失败；可改用本地转写文件。")
    caption = target / f"raw-source.{metadata['caption_language']}.vtt"
    if not info_path.is_file() or not caption.is_file():
        raise UrlIngestError(
            "CAPTION_UNAVAILABLE", "所选字幕不可用；请更换语言、选择自动字幕或导入本地转写。"
        )
    if info.get("id") != metadata["video_id"].removeprefix("youtube-"):
        raise UrlIngestError("URL_INVALID", "视频标识与请求不一致。")
    duration = info.get("duration")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not 0 < duration <= MAX_VIDEO_SECONDS
    ):
        raise UrlIngestError("VIDEO_DURATION_INVALID", "视频时长无效或超过 5 小时。")
    transcript = normalize_text(read_text(caption), ".vtt")
    (run / "transcript.md").write_text(transcript)
    return {
        "title": metadata.get("title") or info.get("title") or info["id"],
        "uploader": info.get("uploader") or "",
        "duration_ms": round(duration * 1000),
        "attribution": (
            f"YouTube；{info.get('uploader') or ''}；《{info.get('title') or info['id']}》；{url}"
        ),
    }
