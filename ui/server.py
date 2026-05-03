import flask
import subprocess
import sys
import os
import threading
import json

app = flask.Flask(__name__)

UI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(UI_DIR)
DATA_FILE  = os.path.join(PROJECT_ROOT, "data", "intel", "season_simulation.json")
EXPL_FILE  = os.path.join(PROJECT_ROOT, "models", "stage9_explanations.json")

_process_running = False
_process_lock = threading.Lock()


def _watch_process(proc):
    global _process_running
    proc.wait()
    with _process_lock:
        _process_running = False


@app.route("/")
def index():
    return flask.send_from_directory(UI_DIR, "index.html")


@app.route("/api/data")
def get_data():
    if not os.path.exists(DATA_FILE):
        return flask.jsonify({"error": "No data found"}), 404
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return flask.jsonify(json.load(f))


@app.route("/api/explanations")
def get_explanations():
    if not os.path.exists(EXPL_FILE):
        return flask.jsonify({}), 200   # return empty, not an error
    with open(EXPL_FILE, "r", encoding="utf-8") as f:
        return flask.jsonify(json.load(f))


@app.route("/api/run", methods=["POST"])
def run_sim():
    global _process_running
    with _process_lock:
        if _process_running:
            return flask.jsonify({"status": "already_running"})
        script = os.path.join(PROJECT_ROOT, "pipeline", "season_simulator.py")
        proc = subprocess.Popen([sys.executable, script], cwd=PROJECT_ROOT)
        _process_running = True
    t = threading.Thread(target=_watch_process, args=(proc,), daemon=True)
    t.start()
    return flask.jsonify({"status": "started"})


@app.route("/api/status")
def status():
    with _process_lock:
        return flask.jsonify({"running": _process_running})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
