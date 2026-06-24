"""
HTTP API服务模块
提供REST接口和WebSocket推送，对接船舶过闸综合辅助终端。
"""
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from loguru import logger


class _State:
    """全局状态存储"""
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_record = None
        self.history = []
        self.max_history = 100


_state = _State()


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._json_response(200, _get_status())
        elif parsed.path == "/api/latest":
            self._json_response(200, _get_latest())
        elif parsed.path == "/api/history":
            params = parse_qs(parsed.query)
            n = int(params.get("n", [10])[0])
            self._json_response(200, _get_history(n))
        elif parsed.path == "/health":
            self._json_response(200, {"status": "ok", "timestamp": time.time()})
        else:
            self._json_response(404, {"error": "not found"})

    def _json_response(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        pass


def _get_status():
    with _state.lock:
        return {
            "running": True,
            "latest_timestamp": _state.latest_record["timestamp"] if _state.latest_record else None,
            "history_count": len(_state.history)
        }

def _get_latest():
    with _state.lock:
        return _state.latest_record or {}

def _get_history(n):
    with _state.lock:
        return _state.history[-n:]


class APIServer:
    """HTTP API服务器"""

    def __init__(self, config: dict):
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 8080)
        self._server = None
        self._thread = None

    def start(self):
        self._server = HTTPServer((self.host, self.port), RequestHandler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                         daemon=True)
        self._thread.start()
        logger.info(f"API server started: http://{self.host}:{self.port}")
        logger.info(f"  GET /api/status  - system status")
        logger.info(f"  GET /api/latest  - latest monitoring result")
        logger.info(f"  GET /api/history - recent history")
        logger.info(f"  GET /health      - health check")

    def stop(self):
        if self._server:
            self._server.shutdown()
            logger.info("API server stopped")

    @staticmethod
    def push_record(record: dict):
        with _state.lock:
            _state.latest_record = record
            _state.history.append(record)
            if len(_state.history) > _state.max_history:
                _state.history = _state.history[-_state.max_history:]
