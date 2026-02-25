from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
import os
import json
import io
import zipfile
from datetime import datetime
from faster_whisper import WhisperModel  # pip install faster-whisper flask

app = Flask(__name__)

# =====================
# パス関連
# =====================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads_raw")   # WAV保存先
TEXT_DIR   = os.path.join(BASE_DIR, "uploads_text")  # STT結果(JSON)保存先
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# =====================
# Whisper 設定・ロード（軽量版）
# =====================
MODEL_SIZE = "tiny"     # 軽量モデル
DEVICE     = "cpu"      # Render 無料は CPU 想定
COMPUTE    = "int8"     # CPU で軽量に動かす設定
LANGUAGE   = "ja"       # 日本語固定（自動検出したければ None）

print(f"[INFO] Loading Whisper model: size={MODEL_SIZE}, device={DEVICE}, compute={COMPUTE}")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE)
print("[INFO] Whisper model loaded.")

# =====================
# ユーティリティ
# =====================

def save_wav_bytes(data: bytes) -> str:
    """生バイトを uploaded_YYYYMMDD_HHMMSS.wav として保存してパスを返す。"""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"uploaded_{now}.wav"
    save_path = os.path.join(UPLOAD_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(data)
    print("Saved WAV:", save_path, "size:", len(data))
    return save_path

def transcribe_wav(path: str) -> dict:
    """faster-whisperでWAVを文字起こしして情報をまとめて返す。"""
    print("[INFO] STT start:", path)

    segments, info = model.transcribe(
        path,
        language=LANGUAGE,  # None なら自動判定
        beam_size=1,        # 軽め
        vad_filter=False,   # まずは切って様子を見る
    )

    text = "".join(seg.text for seg in segments).strip()

    seg_list = []
    for seg in segments:
        seg_list.append({
            "id": seg.id,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        })

    print(f"[INFO] STT done: language={info.language}, duration={info.duration}s")
    return {
        "text": text,
        "language": info.language,
        "duration": info.duration,
        "segments": seg_list,
    }

def save_transcription_json(wav_path: str, stt_result: dict) -> str:
    """WAVファイル名＋STT結果をJSONにして保存しパスを返す。"""
    now = datetime.now().isoformat()
    wav_filename  = os.path.basename(wav_path)
    json_filename = os.path.splitext(wav_filename)[0] + ".json"
    json_path     = os.path.join(TEXT_DIR, json_filename)

    meta = {
        "filename": wav_filename,
        "datetime": now,
        "text": stt_result.get("text", ""),
        "language": stt_result.get("language"),
        "duration": stt_result.get("duration"),
        "segments": stt_result.get("segments", []),
        "source": "cardputer",
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved JSON:", json_path)
    return json_path

def load_all_transcripts():
    """uploads_text/*.json を全部読み込んでリストで返す。"""
    items = []
    if not os.path.isdir(TEXT_DIR):
        return items

    for name in sorted(os.listdir(TEXT_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(TEXT_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            dt_str = data.get("datetime")
            if dt_str:
                try:
                    dt = datetime.fromisoformat(dt_str)
                except ValueError:
                    dt = None
            else:
                dt = None
            data["_dt"] = dt
            items.append(data)
        except Exception as e:
            print(f"[WARN] failed to load {path}: {e}")
    items.sort(key=lambda x: x.get("_dt") or datetime.min, reverse=True)
    return items

def filter_by_range(items, start_dt, end_dt):
    if start_dt is None and end_dt is None:
        return items
    result = []
    for it in items:
        dt = it.get("_dt")
        if dt is None:
            continue
        if start_dt and dt < start_dt:
            continue
        if end_dt and dt > end_dt:
            continue
        result.append(it)
    return result

# =====================
# API: 音声アップロード
# =====================

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    print("REQ:", request.method, request.path)

    data = request.data
    if not data:
        print("No data in request")
        return jsonify({"status": "error", "message": "no data"}), 400

    # 1) WAV保存
    try:
        wav_path = save_wav_bytes(data)
    except Exception as e:
        print("Error saving wav:", e)
        return jsonify({"status": "error", "message": f"save failed: {e}"}), 500

    # 2) STT実行（必要なければここをコメントアウトしてもOK）
    try:
        stt_result = transcribe_wav(wav_path)
    except Exception as e:
        print("Error in STT:", e)
        return jsonify({"status": "error", "message": f"stt failed: {e}"}), 500

    # 3) JSON保存
    try:
        json_path = save_transcription_json(wav_path, stt_result)
    except Exception as e:
        print("Error saving json:", e)
        return jsonify({"status": "error", "message": f"json save failed: {e}"}), 500

    # 4) レスポンス
    response = {
        "status": "ok",
        "filename": os.path.basename(wav_path),
        "json": os.path.basename(json_path),
        "text": stt_result.get("text", ""),
        "language": stt_result.get("language"),
        "duration": stt_result.get("duration"),
    }
    return jsonify(response), 200

# =====================
# 音声ファイル配信（ブラウザ再生用）
# =====================

@app.route("/audio/<path:filename>")
def serve_audio(filename):
    """uploads_raw 内の WAV をそのまま返す。"""
    return send_from_directory(UPLOAD_DIR, filename)

# =====================
# 選択した録音を ZIP で一括ダウンロード
# =====================

@app.route("/download_selected", methods=["POST"])
def download_selected():
    """チェックされた filename の WAV を ZIP にまとめて返す。"""
    filenames = request.form.getlist("files")

    if not filenames:
        return jsonify({"status": "error", "message": "no files selected"}), 400

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in filenames:
            path = os.path.join(UPLOAD_DIR, name)
            if os.path.isfile(path):
                zf.write(path, arcname=name)
    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="selected_recordings.zip",
    )

# =====================
# Web UI: 検索画面
# =====================

@app.route("/", methods=["GET", "POST"])
def index():
    now = datetime.now()
    default_start_date = now.strftime("%Y-%m-%d")
    default_end_date   = now.strftime("%Y-%m-%d")
    default_start_time = "00:00"
    default_end_time   = "23:59"

    if request.method == "POST":
        start_date = request.form.get("start_date") or default_start_date
        start_time = request.form.get("start_time") or default_start_time
        end_date   = request.form.get("end_date")   or default_end_date
        end_time   = request.form.get("end_time")   or default_end_time
    else:
        start_date = default_start_date
        start_time = default_start_time
        end_date   = default_end_date
        end_time   = default_end_time

    def parse_dt(date_str, time_str, is_end=False):
        if not date_str:
            return None
        if not time_str:
            time_str = "00:00" if not is_end else "23:59"
        try:
            dt = datetime.strptime(date_str + " " + time_str, "%Y-%m-%d %H:%M")
            return dt
        except ValueError:
            return None

    start_dt = parse_dt(start_date, start_time, is_end=False)
    end_dt   = parse_dt(end_date,   end_time,   is_end=True)

    items    = load_all_transcripts()
    filtered = filter_by_range(items, start_dt, end_dt)

    return render_template(
        "index.html",
        items=filtered,
        start_date=start_date,
        start_time=start_time,
        end_date=end_date,
        end_time=end_time,
    )

# =====================
# メイン
# =====================

if __name__ == "__main__":
    # ローカルデバッグ用。Render ではプロセスマネージャが起動する
    app.run(host="0.0.0.0", port=5000, debug=True)
