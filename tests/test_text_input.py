import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_report_agent import pipeline, text_input
from video_report_agent.ingest import UrlIngestError
from video_report_agent.storage import Store

VTT = "WEBVTT\n\n00:00:01.200 --> 00:00:03.360\nHello world\n"
URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


@pytest.mark.parametrize(
    "url", [URL, "https://youtu.be/jNQXAC9IVRw?t=2", "https://www.youtube.com/shorts/jNQXAC9IVRw"]
)
def test_youtube_identity(url):
    assert text_input.youtube_url(url) == ("jNQXAC9IVRw", URL)


@pytest.mark.parametrize(
    "url",
    [
        "http://www.youtube.com/watch?v=jNQXAC9IVRw",
        "https://youtube.com.evil.test/watch?v=jNQXAC9IVRw",
        "https://user@www.youtube.com/watch?v=jNQXAC9IVRw",
        URL + "&v=AAAAAAAAAAA",
        URL + "#fragment",
        "https://www.youtube.com:443/watch?v=jNQXAC9IVRw",
        "https://www.youtube.com/playlist?list=test",
        "https://www.xiaohongshu.com/explore/6411cf99000000001300b6d9",
    ],
)
def test_reject_other_destinations(url):
    with pytest.raises(ValueError):
        text_input.youtube_url(url)


def test_normalize_preserves_text_and_does_not_invent_times():
    text = text_input.normalize_text("先说明条件。\n\n结果可能改善。", ".txt")
    assert "时间未知" in text
    assert "结果可能改善。" in text
    assert "0.000" not in text
    assert "[text-00002 |" in text
    assert "1.200–3.360s" in text_input.normalize_text(VTT, ".vtt")


@pytest.mark.parametrize(
    "text,suffix",
    [
        ("", ".txt"),
        ("not subtitles", ".srt"),
        ("WEBVTT\n\n00:00:03.000 --> 00:00:01.000\nBad\n", ".vtt"),
        ("test", ".exe"),
    ],
)
def test_invalid_transcripts(text, suffix):
    with pytest.raises(ValueError):
        text_input.normalize_text(text, suffix)


def test_local_input_is_snapshotted_and_has_no_network_or_asr(tmp_path, monkeypatch):
    source = tmp_path / "captions.vtt"
    source.write_text(VTT)
    run = text_input.create_text_run(tmp_path / "runs", path=source, transcript_only=True)
    source.unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("Imported text must not call Bilibili, ASR or Pi")

    for name in ("fetch_subtitles", "transcribe_audio", "PiRunner", "validate_bilibili_url"):
        monkeypatch.setattr(pipeline, name, forbidden)
    result = pipeline.generate(run)
    assert result["state"] == "TRANSCRIPT_READY"
    assert result["asr_performed"] is False
    store = Store(run.parent / "state.sqlite3", initialize=False)
    assert store.documents(run.name)["original-transcript.vtt"] == VTT
    assert "Hello world" in store.documents(run.name)["transcript.md"]


def mock_captions(monkeypatch, *, missing=False, invalid_id=False):
    monkeypatch.setattr(text_input.shutil, "which", lambda _: "/bin/yt-dlp")

    def download(command, **kwargs):
        assert "--ignore-config" in command
        assert "--skip-download" in command
        assert not any("cookie" in arg for arg in command)
        target = Path(command[command.index("-o") + 1]).parent
        (target / "raw-source.info.json").write_text(
            json.dumps(
                {
                    "id": "wrong" if invalid_id else "jNQXAC9IVRw",
                    "duration": 19,
                    "title": "Example",
                    "uploader": "Author",
                    "url": "https://signed.example/secret",
                }
            )
        )
        if not missing:
            (target / "raw-source.en.vtt").write_text(VTT)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(text_input.subprocess, "run", download)


def test_youtube_captions_enter_shared_pipeline(tmp_path, monkeypatch):
    mock_captions(monkeypatch)
    run = text_input.create_text_run(tmp_path, url=URL, transcript_only=True)
    result = pipeline.generate(run)
    assert result["state"] == "TRANSCRIPT_READY"
    assert result["title"] == "Example"
    assert "signed.example" not in (run / "download/source.info.json").read_text()
    store = Store(tmp_path / "state.sqlite3", initialize=False)
    assert store.run(run.name)["input"]["url"] == URL
    assert "Hello world" in store.documents(run.name)["transcript.md"]


def test_missing_captions_fail_without_asr(tmp_path, monkeypatch):
    mock_captions(monkeypatch, missing=True)
    monkeypatch.setattr(pipeline, "transcribe_audio", lambda *a, **kw: pytest.fail("ASR called"))
    run = text_input.create_text_run(tmp_path, url=URL, transcript_only=True)
    result = pipeline.generate(run)
    assert result["state"] == "FAILED"
    assert result["error_code"] == "CAPTION_UNAVAILABLE"
    store = Store(tmp_path / "state.sqlite3", initialize=False)
    assert "signed.example" not in store.documents(run.name)["download/source.info.json"]


def test_wrong_video_is_rejected(tmp_path, monkeypatch):
    mock_captions(monkeypatch, invalid_id=True)
    run = text_input.create_text_run(tmp_path, url=URL, transcript_only=True)
    result = pipeline.generate(run)
    assert result["error_code"] == "URL_INVALID"


def test_local_report_uses_pi_and_keeps_html_when_image_fails(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text("A complete source paragraph.")
    run = text_input.create_text_run(tmp_path / "runs", path=source)

    class FakePi:
        provider = "fixture"
        model = "fixture"

        async def run(self, workspace):
            assert "A complete source paragraph." in (workspace / "transcript.md").read_text()
            (workspace / "report.html").write_text("<html><body>Report</body></html>")

    def fail_image(*args):
        raise RuntimeError("Browser unavailable")

    monkeypatch.setattr(pipeline, "PiRunner", FakePi)
    monkeypatch.setattr(pipeline, "render_report_image", fail_image)
    result = pipeline.generate(run)
    assert result["state"] == "RENDERED"
    assert result["image_error"] == "Browser unavailable"
    assert (run / "report.html").is_file()


def test_download_timeout_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(text_input.shutil, "which", lambda _: "/bin/yt-dlp")

    def timeout(command, **kwargs):
        raise text_input.subprocess.TimeoutExpired(command, 120)

    monkeypatch.setattr(text_input.subprocess, "run", timeout)
    run = text_input.create_text_run(tmp_path, url=URL)
    metadata = json.loads((run / "input.json").read_text())
    with pytest.raises(UrlIngestError, match="字幕获取超时"):
        text_input.fetch_youtube_captions(run, metadata)


def test_vtt_metadata_and_inline_timestamps_are_not_speech():
    captions = (
        "WEBVTT\n\nNOTE private note\nDo not include this\n\n"
        "STYLE\n::cue { color: red; }\n\n"
        "cue-a\n00:01.000 --> 00:02.000 align:start\n"
        "<v Alice><c>Really</c><00:01.200> long &amp; interesting</v>\n\n"
        "NOTE another note\nPrivate comment\n\n"
        "00:03.000 --> 00:04.000\nEnd\n"
    )
    result = text_input.normalize_text(captions, ".vtt")
    assert "Alice: Really long & interesting" in result
    assert "Private" not in result and "NOTE" not in result and "STYLE" not in result
    assert "<00:" not in result


@pytest.mark.parametrize("timestamp", ["-00:00:01,000", "00:61:01,000", "+00:00:01,000"])
def test_srt_rejects_invalid_raw_times(timestamp):
    with pytest.raises(ValueError):
        text_input.normalize_text(f"1\n{timestamp} --> 00:02:00,000\nText\n", ".srt")


def test_unpublished_raw_info_is_not_captured(tmp_path):
    run = text_input.create_text_run(tmp_path, url=URL, transcript_only=True)
    store = Store(tmp_path / "state.sqlite3", initialize=False)
    execution = store.claim_run(run.name)
    (run / "download").mkdir()
    (run / "download/raw-source.info.json").write_text('{"url": "signed-secret"}')
    store.capture(run, execution)
    assert "signed-secret" not in str(store.documents(run.name))
