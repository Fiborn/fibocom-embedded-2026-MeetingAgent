# -*- coding: utf-8 -*-
"""麦克风阵列几何：位置矩阵、TDOA 预测、角度量化"""
import math

import numpy as np

from config_doa import (
    ARRAY_RADIUS_M,
    AZIMUTH_QUANTIZE_DEG,
    MIC_START_ANGLE_DEG,
    NUM_MICS,
    SPEED_OF_SOUND,
)


def mic_positions(num_mics=NUM_MICS, radius=ARRAY_RADIUS_M, start_deg=MIC_START_ANGLE_DEG):
    """环形阵列各麦克风在 XY 平面上的坐标 (N, 2)，单位：米"""
    angles = np.deg2rad(start_deg + np.arange(num_mics) * (360.0 / num_mics))
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.stack([x, y], axis=1)


def direction_from_angles(azimuth_deg, elevation_deg=0.0):
    """方位角/俯仰角 → 单位方向向量 (dx, dy, dz)"""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    dx = math.cos(el) * math.cos(az)
    dy = math.cos(el) * math.sin(az)
    dz = math.sin(el)
    return np.array([dx, dy, dz], dtype=np.float64)


def predict_tdoa_matrix(azimuth_deg, elevation_deg=0.0, positions=None):
    """
    远场近似：τ_i = (p_i · s_xy) / c
    返回相对参考麦（Mic0）的 TDOA 向量，长度 N-1。
    """
    if positions is None:
        positions = mic_positions()
    direction = direction_from_angles(azimuth_deg, elevation_deg)
    s_xy = direction[:2]
    ref = positions[0]
    taus = []
    for i in range(1, len(positions)):
        delta = np.dot(positions[i] - ref, s_xy) / SPEED_OF_SOUND
        taus.append(delta)
    return np.array(taus, dtype=np.float64)


def quantize_azimuth(azimuth_deg, step=AZIMUTH_QUANTIZE_DEG):
    """量化到 8 方向：0°, 45°, …, 315°"""
    step = float(step)
    q = round(azimuth_deg / step) * step
    return q % 360.0


def azimuth_to_beam_index(azimuth_deg, step=AZIMUTH_QUANTIZE_DEG):
    """0°→beam 0, 45°→beam 1, …"""
    q = quantize_azimuth(azimuth_deg, step)
    return int(round(q / step)) % int(360 / step)


def estimate_azimuth_elevation_from_tdoas(tdoas, positions=None):
    """
    由测得的 TDOA（相对 Mic0）网格搜索估计 azimuth / elevation。
    tdoas: shape (N-1,)
    """
    if positions is None:
        positions = mic_positions()

    best_score = -1.0
    best_az = 0.0
    best_el = 0.0

    for az in np.arange(0.0, 360.0, 5.0):
        for el in (0.0, 15.0, 30.0, 45.0):
            pred = predict_tdoa_matrix(az, el, positions)
            err = np.linalg.norm(pred - tdoas)
            score = 1.0 / (1.0 + err * 1000.0)
            if score > best_score:
                best_score = score
                best_az = az
                best_el = el

    return best_az, best_el, float(best_score)
