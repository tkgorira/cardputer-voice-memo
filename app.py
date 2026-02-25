from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
import os
import io
import zipfile
from datetime import datetime

app = Flask(__name__)

# =====================
# パス関連
# =====================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads_raw")   # WAV保存先
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

def load_all_records():
    """
    uploads_raw 内の WAV を読み込み、
    ファイル名のタイムスタンプから datetime を作って降順ソートして返す。
    """
    items = []
    if not os.path.isdir(UPLOAD_DIR):
        return items

    for name in sorted(os.listdir(UPLOAD_DIR)):
        if not name.endswith(".wav"):
            continue
        path = os.path.join(UPLOAD_DIR, name)
        if not os.path.isfile(path):
            continue

        # ファイル名: uploaded_YYYYMMDD_HHMMSS.wav
        dt = None
        try:
            base = os.path.splitext(name)[0]  # uploaded_YYYYMMDD_HHMMSS
            ts   = base.replace("uploaded_", "")
            dt   = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        except Exception:
            dt = None

        items.append({
            "filename": name,
            "datetime": dt.isoformat() if dt else "",
            "_dt": dt,
            "text": "",  # 将来ローカルSTT結果を埋め込む余地
        })

    # 日時降順
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
# API: 音声アップロード（保存のみ）
# =====================

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    print("REQ:", request.method, request.path)

    data = request.data
    if not data:
        print("No data in request")
        return jsonify({"status": "error", "message": "no data"}), 400

    # 1) WAV保存のみ
    try:
        wav_path = save_wav_bytes(data)
    except Exception as e:
        print("Error saving wav:", e)
        return jsonify({"status": "error", "message": f"save failed: {e}"}), 500

    # STT は行わない（サーバは軽く保つ）
    return jsonify({
        "status": "ok",
        "filename": os.path.basename(wav_path),
    }), 200

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
# Web UI: 検索 + 再生 + 選択DL
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

    items    = load_all_records()
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
