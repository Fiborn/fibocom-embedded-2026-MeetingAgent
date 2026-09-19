# -*- coding: utf-8 -*-
"""DOA 数据类型：对应 PDF 中的 JSON 与 C 结构体 DoaResult_t"""
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class SourceData:
    azimuth: float
    elevation: float
    confidence: float
    beam_index: int


@dataclass
class DoaResult:
    """与 PDF JSON / DoaResult_t 字段一一对应"""

    timestamp: int
    vad_active: bool
    source_data: SourceData

    @classmethod
    def silent(cls, timestamp_ms=None):
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        return cls(
            timestamp=ts,
            vad_active=False,
            source_data=SourceData(
                azimuth=0.0,
                elevation=0.0,
                confidence=0.0,
                beam_index=0,
            ),
        )

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "vad_active": self.vad_active,
            "source_data": asdict(self.source_data),
        }

    def to_json(self, indent=None) -> str:
        return json.dumps(self.to_json_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "DoaResult":
        sd = data.get("source_data") or {}
        return cls(
            timestamp=int(data["timestamp"]),
            vad_active=bool(data.get("vad_active", False)),
            source_data=SourceData(
                azimuth=float(sd.get("azimuth", 0.0)),
                elevation=float(sd.get("elevation", 0.0)),
                confidence=float(sd.get("confidence", 0.0)),
                beam_index=int(sd.get("beam_index", 0)),
            ),
        )

    def to_c_struct_comment(self) -> str:
        """便于与底层 C IPC 对照"""
        sd = self.source_data
        return (
            "DoaResult_t {{ timestamp_ms={}, vad_flag={}, "
            "azimuth={:.2f}, elevation={:.2f}, confidence={:.2f}, beam_idx={} }}"
        ).format(
            self.timestamp,
            1 if self.vad_active else 0,
            sd.azimuth,
            sd.elevation,
            sd.confidence,
            sd.beam_index,
        )
