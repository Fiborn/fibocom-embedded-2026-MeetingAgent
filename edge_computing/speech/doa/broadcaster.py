# -*- coding: utf-8 -*-
"""DOA JSON 广播：控制台 / JSONL 文件 / 可选 TCP"""
import json
import socket
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from config_doa import JSON_LOG, OUTPUT_DIR
from doa_types import DoaResult


class DoaBroadcaster:
    def __init__(
        self,
        console=True,
        log_file=True,
        tcp_port=None,
        on_result: Optional[Callable[[DoaResult], None]] = None,
    ):
        self.console = console
        self.log_file = log_file
        self.tcp_port = tcp_port
        self.on_result = on_result
        self._log_path = Path(JSON_LOG)
        self._clients = []
        self._lock = threading.Lock()
        self._server = None

        if self.log_file:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if self.tcp_port:
            self._start_tcp_server(self.tcp_port)

    def _start_tcp_server(self, port):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", port))
        self._server.listen(4)
        threading.Thread(target=self._accept_loop, daemon=True).start()
        print("[DOA 广播] TCP 监听 127.0.0.1:{}".format(port))

    def _accept_loop(self):
        while self._server:
            try:
                client, _ = self._server.accept()
                with self._lock:
                    self._clients.append(client)
            except OSError:
                break

    def publish(self, result: DoaResult):
        payload = result.to_json()
        if self.console:
            self._print_console(result)
        if self.log_file:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(payload + "\n")
        if self.tcp_port:
            self._broadcast_tcp(payload)
        if self.on_result:
            self.on_result(result)

    @staticmethod
    def _print_console(result: DoaResult):
        if not result.vad_active:
            line = "[DOA] 静音 | ts={}".format(result.timestamp)
        else:
            sd = result.source_data
            line = (
                "[DOA] 说话中 | 方位 {:.0f}° 俯仰 {:.0f}° "
                "置信度 {:.2f} 波束 {} | ts={}".format(
                    sd.azimuth,
                    sd.elevation,
                    sd.confidence,
                    sd.beam_index,
                    result.timestamp,
                )
            )
        sys.stdout.write("\r" + line.ljust(96))
        sys.stdout.flush()

    def _broadcast_tcp(self, payload):
        dead = []
        line = (payload + "\n").encode("utf-8")
        with self._lock:
            for client in self._clients:
                try:
                    client.sendall(line)
                except OSError:
                    dead.append(client)
            for client in dead:
                self._clients.remove(client)

    def close(self):
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        with self._lock:
            for client in self._clients:
                try:
                    client.close()
                except OSError:
                    pass
            self._clients.clear()


def load_jsonl(path):
    """读取历史 DOA JSONL 日志"""
    path = Path(path)
    results = []
    if not path.exists():
        return results
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(DoaResult.from_json_dict(json.loads(line)))
    return results
