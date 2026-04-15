#!/usr/bin/env python3
"""
==============================================================================
hikvision_thermal.py  –  Hikvision / HIKMicro USB Thermal Camera Server
==============================================================================

Beschreibung:
    Liest Wärmebilddaten von einer Hikvision USB-Kamera (UVC-Protokoll via
    Video4Linux2), wandelt die Rohdaten in falschfarbige JPEG-Frames um und
    stellt diese als MJPEG-Stream über einen eingebetteten HTTP-Server bereit.

    Zusätzlich werden echte Temperaturdaten aus den Kamera-Metadaten gelesen
    und als Overlay (Hotspot, Coldspot, OSD-Leiste) in das Bild eingeblendet.

Getestet mit:
    - HIK Camera 2bdf:0102  (HIKMicro Pocket-Serie)
    - Ubuntu 22.04 in Proxmox LXC-Container
    - Python 3.10, numpy 2.x, opencv-python-headless 4.x, ffmpeg 4.4

Abhängigkeiten:
    pip install numpy opencv-python-headless
    apt  install ffmpeg

Verwendung:
    python3 hikvision_thermal.py

Web-UI:
    http://<IP>:8890/

Autor:     Reverse-engineered & implementiert mit Claude (Anthropic)
Lizenz:    MIT
==============================================================================
"""

import os
import sys
import threading
import time
import subprocess
import json
import struct
from datetime import datetime
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

import numpy as np
import cv2


# ==============================================================================
#  KONFIGURATION
# ==============================================================================

PORT            = 8890    # HTTP-Port des Web-Servers
DEVICE          = "/dev/video0"  # V4L2-Gerät der Kamera

# Kamera-Auflösung (UVC meldet 256×344, enthält aber Metadaten in den unteren Zeilen)
WIDTH           = 256
HEIGHT          = 344           # Gesamthöhe des UVC-YUYV-Frames
THERMAL_HEIGHT  = 192           # Tatsächliche Thermalbild-Zeilen (Zeilen 0–191)

FPS             = 9             # Gewünschte Framerate (Kamera unterstützt max. ~9 fps)
FRAME_BYTES     = WIDTH * HEIGHT * 2  # YUYV = 2 Bytes pro Pixel

# Speicherpfade
RECORDINGS_DIR  = "/var/lib/vz/recordings/streams/hikvision"
SNAPSHOT_DIR    = "/var/lib/vz/recordings/snapshots"
LOG_FILE        = "/root/recordings/hikvision.log"


# ==============================================================================
#  FARBPALETTEN
# ==============================================================================
# OpenCV-Colormaps + zwei Sonderfälle (None = Graustufen, "invert" = invertiert)

PALETTES = {
    "ironbow":   cv2.COLORMAP_INFERNO,   # Blau → Lila → Rot → Gelb  (Standard Thermal)
    "rainbow":   cv2.COLORMAP_TURBO,    # Blau → Grün → Gelb → Rot
    "white_hot": None,                  # Graustufen: heiß = weiß
    "black_hot": "invert",              # Graustufen invertiert: heiß = schwarz
    "lava":      cv2.COLORMAP_HOT,      # Schwarz → Rot → Orange → Weiß
    "arctic":    cv2.COLORMAP_COOL,     # Cyan → Magenta
    "fusion":    cv2.COLORMAP_PLASMA,   # Lila → Rot → Gelb (Plasma)
    "viridis":   cv2.COLORMAP_VIRIDIS,  # Lila → Blau → Grün → Gelb (wissenschaftlich)
}
DEFAULT_PALETTE = "ironbow"


# ==============================================================================
#  LOGGING
# ==============================================================================

def log(msg: str):
    """Schreibt eine Zeile mit Zeitstempel auf stdout und in LOG_FILE."""
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass  # Logging darf nie den Hauptprozess zum Absturz bringen


# ==============================================================================
#  METADATEN-PARSING (Temperatur aus Kamera-Frame)
# ==============================================================================
#
# Die Hikvision-Kamera (2bdf:0102) liefert im UVC-YUYV-Frame (256×344) ab
# Zeile 192 binäre Metadaten. Alle 512 Bytes der Zeile werden als linearer
# Binär-Buffer genutzt (YUYV-Bytes werden nicht getrennt ausgewertet).
#
# Bekannte Offsets in Zeile 192 (uint16, Little-Endian):
#   Byte  0-1:  T_min der aktuellen Szene (°C × 100)
#   Byte  2-3:  T_max der aktuellen Szene (°C × 100)
#   Byte 16-17: Frame-Höhe (Kontroll-Wert, sollte 192 sein)
#   Byte 18-19: Frame-Breite (Kontroll-Wert, sollte 256 sein)
#   Byte 40-41: Hotspot-Temperatur (°C × 100)
#   Byte 42-43: Hotspot X-Koordinate (Pixel)
#   Byte 44-45: Hotspot Y-Koordinate (Pixel)
#   Byte 50-51: Coldspot-Temperatur (°C × 100)
#
# Zeile 193 beginnt mit Magic 0xAABBCCDD und enthält den Sensor-Kalibrierbereich.

def parse_metadata(raw_bytes: bytes) -> dict:
    """
    Liest Temperaturdaten aus dem Metadaten-Bereich des UVC-Frames.

    Args:
        raw_bytes: Rohes YUYV-Frame (WIDTH × HEIGHT × 2 Bytes)

    Returns:
        dict mit "valid": True und Temperaturwerten, oder {"valid": False}
    """
    try:
        data = np.frombuffer(raw_bytes, dtype=np.uint8)
        # Zeile 192 als flachen Byte-Buffer extrahieren (512 Bytes)
        row = bytes(data.reshape(HEIGHT, WIDTH * 2)[192])

        # Hilfsfunktion: uint16 Little-Endian an Byte-Offset lesen
        u16 = lambda off: struct.unpack_from("<H", row, off)[0]

        # Plausibilitätsprüfung: stimmen Breite/Höhe mit unseren Konstanten überein?
        if u16(16) != THERMAL_HEIGHT or u16(18) != WIDTH:
            return {"valid": False}

        t_min = u16(0) / 100.0  # z.B. 1373 → 13.73 °C
        t_max = u16(2) / 100.0  # z.B. 1660 → 16.60 °C

        # Weiterer Plausibilitätscheck: Temperaturbereich sinnvoll?
        if not (-40 <= t_min <= 500 and t_max > t_min):
            return {"valid": False}

        return {
            "valid":   True,
            "t_min":   t_min,               # Kälteste Stelle der Szene
            "t_max":   t_max,               # Heißeste Stelle der Szene
            "hs_temp": u16(40) / 100.0,     # Hotspot-Temperatur
            "hs_x":    min(int(u16(42)), WIDTH - 1),          # Hotspot X
            "hs_y":    min(int(u16(44)), THERMAL_HEIGHT - 1), # Hotspot Y
            "cs_temp": u16(50) / 100.0,     # Coldspot-Temperatur
        }
    except Exception:
        return {"valid": False}


def y_to_temp(y_val: float, t_min: float, t_max: float) -> float:
    """
    Rechnet einen Y-Kanal-Wert (0–255) in eine Temperatur um.

    Die Kamera normiert das Thermalbild so, dass Y=0 → T_min und Y=255 → T_max.
    Die Zuordnung ist nach dem Kontrast-Stretch näherungsweise linear.
    """
    return t_min + (float(y_val) / 255.0) * (t_max - t_min)


# ==============================================================================
#  COLORBAR (Farbskala mit Temperaturbeschriftung)
# ==============================================================================

def make_colorbar(palette_id, t_min=None, t_max=None, h=192, w=28) -> np.ndarray:
    """
    Erzeugt eine vertikale Colorbar (BGR-Bild) mit optionaler Temp-Beschriftung.

    Args:
        palette_id: OpenCV COLORMAP_* Konstante, None (Graustufen) oder "invert"
        t_min:      Untere Temperatur für Beschriftung (°C)
        t_max:      Obere Temperatur für Beschriftung (°C)
        h:          Höhe der Bar in Pixeln (sollte = Bildhöhe sein)
        w:          Breite der Bar in Pixeln

    Returns:
        numpy-Array (h, w, 3) BGR
    """
    # Gradient von 255 (oben = heiß) nach 0 (unten = kalt)
    g = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    g = np.tile(g, (1, w))

    # Palette anwenden
    if palette_id is None:
        bar = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    elif palette_id == "invert":
        bar = cv2.cvtColor(255 - g, cv2.COLOR_GRAY2BGR)
    else:
        bar = cv2.applyColorMap(g, palette_id)

    # Temperatur-Labels oben, mitte, unten einblenden
    if t_min is not None and t_max is not None:
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.28
        for temp, frac in [(t_max, 0.06), ((t_min + t_max) / 2, 0.51), (t_min, 0.95)]:
            y0  = max(8, int(frac * h))
            lbl = f"{temp:.0f}"
            # Schwarze Kontur für Lesbarkeit auf allen Hintergründen
            cv2.putText(bar, lbl, (1, y0), font, scale, (0, 0, 0),     2, cv2.LINE_AA)
            cv2.putText(bar, lbl, (1, y0), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    return bar


# ==============================================================================
#  OVERLAY-ZEICHENFUNKTIONEN
# ==============================================================================

def _txt(img, text: str, pos: tuple, scale: float, fg: tuple, bg=(0, 0, 0)):
    """Schreibt Text mit schwarzer Kontur (für Lesbarkeit auf beliebigem Hintergrund)."""
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, f, scale, bg, 2, cv2.LINE_AA)  # Kontur
    cv2.putText(img, text, pos, f, scale, fg, 1, cv2.LINE_AA)  # Vordergrund


def _cross(img, x: int, y: int, color: tuple, size: int = 6):
    """Zeichnet ein Fadenkreuz-Marker an Position (x, y)."""
    # Schwarze Kontur
    cv2.line(img, (x - size, y), (x + size, y), (0, 0, 0), 2)
    cv2.line(img, (x, y - size), (x, y + size), (0, 0, 0), 2)
    # Farbige Linie
    cv2.line(img, (x - size, y), (x + size, y), color, 1)
    cv2.line(img, (x, y - size), (x, y + size), color, 1)


def draw_overlay(colored: np.ndarray, y_raw: np.ndarray,
                 meta: dict, t_min_disp: float, t_max_disp: float,
                 show_osd: bool, show_hotspot: bool,
                 show_coldspot: bool, show_crosshair: bool,
                 alert_temp, blink: bool) -> np.ndarray:
    """
    Zeichnet Temperatur-Overlays auf das bereits eingefärbte BGR-Bild.

    Hinweis: Diese Funktion wird VOR der Rotation aufgerufen, damit die
    Pixelkoordinaten aus den Metadaten (Hotspot X/Y) korrekt sind.

    Args:
        colored:      BGR-Bild (THERMAL_HEIGHT × WIDTH × 3)
        y_raw:        Original Y-Kanal (THERMAL_HEIGHT × WIDTH), uint8
        meta:         Ausgabe von parse_metadata()
        t_min_disp:   Anzeigebereich Minimum (für OSD und Fadenkreuz-Temp)
        t_max_disp:   Anzeigebereich Maximum
        show_osd:     OSD-Leiste am unteren Bildrand anzeigen
        show_hotspot: Hotspot-Marker anzeigen
        show_coldspot: Coldspot-Marker anzeigen
        show_crosshair: Temperatur am Bildmittelpunkt anzeigen
        alert_temp:   Alarm-Schwelle (°C) oder None
        blink:        True/False wechselnd (~2 Hz) für Alarm-Blinken

    Returns:
        Modifiziertes BGR-Bild
    """
    out   = colored.copy()
    ih, iw = out.shape[:2]
    valid = meta.get("valid") and t_min_disp is not None

    # ── Hotspot-Marker (rot) ──────────────────────────────────────────────────
    if show_hotspot and valid:
        hx, hy = meta["hs_x"], meta["hs_y"]
        _cross(out, hx, hy, (0, 80, 255))
        # Temperatur-Label neben dem Marker (Position so wählen, dass kein Rand-Clipping)
        tx = max(2, min(hx - 14, iw - 52))
        ty = max(10, hy - 5)
        _txt(out, f"{meta['hs_temp']:.1f}C", (tx, ty), 0.33, (0, 140, 255))

    # ── Coldspot-Marker (orange) ──────────────────────────────────────────────
    if show_coldspot and valid:
        # Coldspot-Position: Pixel mit minimalem Y-Wert im Thermalbild
        idx   = int(np.argmin(y_raw))
        cy_p  = idx // WIDTH
        cx_p  = idx % WIDTH
        _cross(out, cx_p, cy_p, (255, 160, 0))
        tx = max(2, min(cx_p - 14, iw - 52))
        ty = max(10, cy_p - 5)
        _txt(out, f"{meta['cs_temp']:.1f}C", (tx, ty), 0.33, (255, 160, 0))

    # ── Fadenkreuz-Temperatur (Bildmitte) ─────────────────────────────────────
    if show_crosshair and valid:
        cy_c = THERMAL_HEIGHT // 2
        cx_c = WIDTH // 2
        ct   = y_to_temp(int(y_raw[cy_c, cx_c]), t_min_disp, t_max_disp)
        cv2.drawMarker(out, (cx_c, cy_c), (0, 255, 200), cv2.MARKER_CROSS, 12, 1)
        _txt(out, f"{ct:.1f}C", (cx_c + 8, cy_c - 4), 0.35, (0, 255, 200))

    # ── Alarm-Blitz (wenn Hotspot-Temp >= Schwelle) ───────────────────────────
    if alert_temp is not None and valid and meta["hs_temp"] >= alert_temp and blink:
        cv2.rectangle(out, (0, 0), (iw - 1, ih - 1), (0, 0, 255), 3)
        _txt(out, f"ALARM > {alert_temp:.0f}C", (4, 14), 0.45, (0, 0, 255), (255, 255, 255))

    # ── OSD-Leiste (unterer Bildrand) ─────────────────────────────────────────
    if show_osd and valid:
        bar_h = 17
        # Halbtransparenter schwarzer Balken
        cv2.rectangle(out, (0, ih - bar_h), (iw, ih), (12, 12, 12), -1)
        # Temperatur am Bildmittelpunkt berechnen
        ctr_v = int(y_raw[THERMAL_HEIGHT // 2, WIDTH // 2])
        ctr_t = y_to_temp(ctr_v, t_min_disp, t_max_disp)
        txt = f"MIN {t_min_disp:.1f}  CTR {ctr_t:.1f}  MAX {t_max_disp:.1f} C"
        _txt(out, txt, (4, ih - 4), 0.30, (210, 210, 210), (0, 0, 0))

    return out


# ==============================================================================
#  FRAME-BUILDER (Haupt-Verarbeitungspipeline)
# ==============================================================================

def build_frame(y: np.ndarray,
                palette_id,
                colorbar: np.ndarray,
                rotation: int, flip_h: bool, flip_v: bool,
                meta: dict,
                cal_mode: str, cal_level: float, cal_span: float,
                sharpen: float, emissivity: float,
                show_osd: bool, show_hotspot: bool,
                show_coldspot: bool, show_crosshair: bool,
                alert_temp, blink: bool,
                frozen_frame) -> bytes | None:
    """
    Vollständige Verarbeitungspipeline: Y-Kanal → JPEG-Bytes.

    Pipeline-Schritte:
        1. Kalibrierung / Kontrast-Stretch
        2. Schärfen (Unsharp Mask)
        3. Colormap anwenden
        4. Temperatur-Overlays zeichnen  ← VOR Rotation (Koordinaten stimmen)
        5. Rotation & Flip
        6. Colorbar links anfügen
        7. JPEG-Encoding

    Args:
        y:            Y-Kanal des Thermalbilds (THERMAL_HEIGHT × WIDTH), uint8
        palette_id:   Colormap-ID (OpenCV-Konstante, None oder "invert")
        colorbar:     Vorberechnete Colorbar (BGR-Array)
        rotation:     Rotation in Grad (0, 90, 180, 270)
        flip_h/v:     Spiegelung
        meta:         Temperatur-Metadaten aus parse_metadata()
        cal_mode:     "auto" oder "manual"
        cal_level:    Manueller Level (Mitte der Anzeigetemperatur) in °C
        cal_span:     Manueller Span (Breite) in °C
        sharpen:      Schärfe-Stärke 0–100 (%)
        emissivity:   Emissivitäts-Korrekturfaktor 0.1–1.0
        show_*:       OSD-Optionen
        alert_temp:   Alarm-Schwelle oder None
        blink:        Blinkzustand für Alarm-Visualisierung
        frozen_frame: Wenn gesetzt, diesen Frame unverändert zurückgeben

    Returns:
        JPEG-komprimierte Bytes oder None bei Fehler
    """

    # Eingefroren: aktuellen verarbeiteten Frame unverändert zurückgeben
    if frozen_frame is not None:
        return frozen_frame

    # ── Schritt 1: Kalibrierung / Kontrast-Stretch ────────────────────────────
    t_min_d = t_max_d = None

    if cal_mode == "auto" and meta.get("valid"):
        # AUTO: Kamera-Metadaten geben T_min/T_max der aktuellen Szene vor.
        # Kontrast-Stretch mit 1%/99%-Perzentil (robust gegen Ausreißer).
        t_min_d, t_max_d = meta["t_min"], meta["t_max"]
        lo, hi = np.percentile(y, 1), np.percentile(y, 99)
        if hi > lo:
            s = np.clip(
                (y.astype(np.float32) - lo) / (hi - lo) * 255,
                0, 255
            ).astype(np.uint8)
        else:
            s = y.copy()

    elif cal_mode == "manual" and meta.get("valid"):
        # MANUELL: Nutzer definiert Level (Mitteltemperatur) und Span (Breite).
        # Y-Werte → absolute °C → auf Benutzerbereich mappen.
        cal_lo  = cal_level - cal_span / 2
        cal_hi  = cal_level + cal_span / 2
        t_min_d, t_max_d = cal_lo, cal_hi

        # Y-Wert → absolute Pixel-Temperatur (laut Kamera-Metadaten)
        t_px = (meta["t_min"]
                + (y.astype(np.float32) / 255.0)
                * (meta["t_max"] - meta["t_min"]))

        # Emissivitäts-Korrektur (vereinfachtes Modell, Ambient = Szenen-Mitte)
        # Formel: T_true ≈ (T_gemessen - (1 - ε) × T_ambient) / ε
        if emissivity < 0.99:
            t_ambient = (meta["t_min"] + meta["t_max"]) / 2
            t_px = (t_px - (1 - emissivity) * t_ambient) / emissivity

        # Auf Benutzer-Bereich abbilden und auf [0, 255] clippen
        s = np.clip(
            (t_px - cal_lo) / (cal_hi - cal_lo) * 255,
            0, 255
        ).astype(np.uint8)

    else:
        # Fallback: simpler Percentile-Stretch ohne Metadaten
        lo, hi = np.percentile(y, 1), np.percentile(y, 99)
        if hi > lo:
            s = np.clip(
                (y.astype(np.float32) - lo) / (hi - lo) * 255,
                0, 255
            ).astype(np.uint8)
        else:
            s = y.copy()

    # ── Schritt 2: Schärfen (Unsharp Mask) ───────────────────────────────────
    # Unsharp Mask: geschärft = original + strength × (original - weichgezeichnet)
    if sharpen > 0:
        strength = sharpen / 100.0 * 1.5   # 0 → 0.0, 100 → 1.5
        blur = cv2.GaussianBlur(s, (3, 3), 0)
        s = np.clip(
            s.astype(np.float32) + strength * (s.astype(np.float32) - blur.astype(np.float32)),
            0, 255
        ).astype(np.uint8)

    # ── Schritt 3: Colormap anwenden ──────────────────────────────────────────
    if palette_id is None:
        colored = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)    # White Hot (Graustufen)
    elif palette_id == "invert":
        colored = cv2.cvtColor(255 - s, cv2.COLOR_GRAY2BGR)  # Black Hot
    else:
        colored = cv2.applyColorMap(s, palette_id)        # OpenCV-Colormap

    # ── Schritt 4: Overlays (VOR Rotation!) ──────────────────────────────────
    # Wichtig: Hotspot/Coldspot-Koordinaten stammen aus Kamera-Metadaten und
    # beziehen sich auf das unkrotierten Frame. Deshalb Overlay vor Rotation.
    colored = draw_overlay(
        colored, y, meta, t_min_d, t_max_d,
        show_osd, show_hotspot, show_coldspot, show_crosshair,
        alert_temp, blink
    )

    # ── Schritt 5: Rotation & Flip ────────────────────────────────────────────
    if rotation == 90:
        colored = cv2.rotate(colored, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        colored = cv2.rotate(colored, cv2.ROTATE_180)
    elif rotation == 270:
        colored = cv2.rotate(colored, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if flip_h:
        colored = cv2.flip(colored, 1)  # 1 = horizontal
    if flip_v:
        colored = cv2.flip(colored, 0)  # 0 = vertikal

    # ── Schritt 6: Colorbar links anfügen ────────────────────────────────────
    # Colorbar auf aktuelle Bildhöhe skalieren (bei Rotation ändert sich die Höhe)
    bar     = cv2.resize(colorbar, (colorbar.shape[1], colored.shape[0]))
    combined = np.hstack([bar, colored])

    # ── Schritt 7: JPEG-Encoding ──────────────────────────────────────────────
    ok, buf = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


# ==============================================================================
#  KAMERA-KLASSE
# ==============================================================================

class ThermalCamera:
    """
    Verwaltet den Kamera-Zustand und startet Capture- und Recording-Threads.

    Thread-Sicherheit:
        - self.current_frame wird durch self.lock geschützt (MJPEG-Stream-Zugriff)
        - self.last_meta wird durch self.meta_lock geschützt
        - Alle anderen Attribute (rotation, palette, ...) werden nur aus dem
          HTTP-Handler heraus gesetzt – dieser ist threaded, aber die Werte
          sind primitiv (int, bool, float), deren Lesen/Schreiben in Python
          durch den GIL atomar ist.
    """

    def __init__(self):
        # Aktueller codierter Frame (JPEG-Bytes) für den MJPEG-Stream
        self.current_frame    = None
        self.lock             = threading.Lock()

        self.running          = False
        self.frames_received  = 0

        # Bild-Einstellungen
        self.palette_name     = DEFAULT_PALETTE
        self.rotation         = 0      # 0 / 90 / 180 / 270
        self.flip_h           = False
        self.flip_v           = False

        # OSD-Sichtbarkeit
        self.show_osd         = True
        self.show_hotspot     = True
        self.show_coldspot    = True
        self.show_crosshair   = False

        # Kalibrierung
        self.cal_mode         = "auto"   # "auto" | "manual"
        self.cal_level        = 30.0     # °C (Mitte des manuellen Bereichs)
        self.cal_span         = 20.0     # °C (Breite des manuellen Bereichs)

        # Erweiterte Einstellungen
        self.sharpen          = 0        # Schärfe 0–100 %
        self.emissivity       = 1.0      # Emissivität 0.1–1.0 (1.0 = kein Effekt)
        self.alert_temp       = None     # Alarm-Schwelle (°C) oder None

        # Interne Zustände
        self.last_meta        = {"valid": False}
        self.meta_lock        = threading.Lock()
        self._blink           = False    # Wechselt ~2 Hz für Alarm-Animation
        self._frozen          = None     # Eingefrorener Frame (bytes) oder None

        # Vorberechnete Colorbar und Palette-ID
        self._palette_id      = None
        self._colorbar        = None
        self._update_palette()

    # ── Palette ───────────────────────────────────────────────────────────────

    def _update_palette(self):
        """Aktualisiert die interne Colorbar und Palette-ID nach Änderungen."""
        pid = PALETTES[self.palette_name]
        self._palette_id = pid
        # Temperatur-Labels für Colorbar aus aktuellem Kalibriermodus holen
        with self.meta_lock:
            m = self.last_meta
        if self.cal_mode == "manual":
            t_min = self.cal_level - self.cal_span / 2
            t_max = self.cal_level + self.cal_span / 2
        elif m.get("valid"):
            t_min, t_max = m["t_min"], m["t_max"]
        else:
            t_min = t_max = None
        self._colorbar = make_colorbar(pid, t_min, t_max)

    def set_palette(self, name: str):
        if name in PALETTES:
            self.palette_name = name
            self._update_palette()
            log(f"Palette: {name}")

    # ── Rotation & Flip ───────────────────────────────────────────────────────

    def rotate(self, action: str):
        """Rotiert das Bild. action: 'cw', 'ccw', '180', '0'"""
        if action == "cw":    self.rotation = (self.rotation + 90)  % 360
        elif action == "ccw": self.rotation = (self.rotation - 90)  % 360
        elif action == "180": self.rotation = (self.rotation + 180) % 360
        elif action == "0":   self.rotation = 0
        log(f"Rotation: {self.rotation}°")

    def toggle_flip_h(self):
        self.flip_h = not self.flip_h
        log(f"Flip H: {self.flip_h}")

    def toggle_flip_v(self):
        self.flip_v = not self.flip_v
        log(f"Flip V: {self.flip_v}")

    # ── Einfrieren ────────────────────────────────────────────────────────────

    def toggle_freeze(self) -> bool:
        """Friert den aktuellen Frame ein oder gibt ihn frei. Gibt neuen Zustand zurück."""
        with self.lock:
            if self._frozen is None:
                self._frozen = self.current_frame  # Aktuellen Frame merken
            else:
                self._frozen = None                # Freigeben
        return self._frozen is not None

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def save_snapshot(self) -> str | None:
        """Speichert aktuellen Frame als JPEG auf Disk. Gibt Dateipfad zurück."""
        with self.lock:
            frame = self.current_frame
        if not frame:
            return None
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SNAPSHOT_DIR, f"snap_{ts}.jpg")
        with open(path, "wb") as f:
            f.write(frame)
        log(f"Snapshot gespeichert: {path}")
        return path

    # ── Threads starten ───────────────────────────────────────────────────────

    def start_capture(self):
        """Startet Capture-Thread und Blink-Thread."""
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._blink_loop,   daemon=True).start()

    def _blink_loop(self):
        """Wechselt self._blink alle 500 ms für Alarm-Animation."""
        while self.running:
            self._blink = not self._blink
            time.sleep(0.5)

    def _capture_loop(self):
        """
        Haupt-Capture-Loop: ffmpeg liest v4l2, Python verarbeitet YUYV-Frames.

        ffmpeg gibt rohe YUYV-Frames als Byte-Stream auf stdout aus.
        Python liest exakt FRAME_BYTES Bytes pro Frame, parst Metadaten,
        extrahiert den Y-Kanal und ruft build_frame() auf.
        """
        cmd = [
            "ffmpeg",
            "-f", "v4l2",
            "-input_format", "yuyv422",
            "-video_size", f"{WIDTH}x{HEIGHT}",
            "-framerate", str(FPS),
            "-i", DEVICE,
            "-f", "rawvideo",
            "-pix_fmt", "yuyv422",
            "-",   # Ausgabe auf stdout
        ]
        log(f"Capture gestartet: {DEVICE} {WIDTH}x{HEIGHT}@{FPS}fps")

        while self.running:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL  # ffmpeg-Statusmeldungen unterdrücken
                )

                while self.running:
                    # Genau einen vollständigen YUYV-Frame lesen
                    raw = proc.stdout.read(FRAME_BYTES)
                    if len(raw) < FRAME_BYTES:
                        break  # ffmpeg-Prozess beendet oder Gerät getrennt

                    # ── Metadaten aus Zeile 192 lesen ──────────────────────
                    meta = parse_metadata(raw)
                    if meta["valid"]:
                        with self.meta_lock:
                            prev = self.last_meta
                            self.last_meta = meta
                        # Colorbar nur neu berechnen wenn sich T-Bereich signifikant ändert
                        if (not prev.get("valid")
                                or abs(prev["t_min"] - meta["t_min"]) > 0.4
                                or abs(prev["t_max"] - meta["t_max"]) > 0.4):
                            self._update_palette()

                    # ── Y-Kanal extrahieren (jedes 2. Byte = Luma) ─────────
                    # YUYV-Layout: Y0 U0 Y1 V0 Y2 U1 Y3 V1 ...
                    # → arr[::2] gibt alle Y-Werte in Reihe
                    arr = np.frombuffer(raw, dtype=np.uint8)
                    y   = arr[::2].reshape(HEIGHT, WIDTH)[:THERMAL_HEIGHT]
                    # Nur die oberen THERMAL_HEIGHT Zeilen = echtes Thermalbild

                    # ── Aktuellen Metadaten-Snapshot für Build holen ────────
                    with self.meta_lock:
                        m = self.last_meta

                    # ── Frame verarbeiten ───────────────────────────────────
                    frame = build_frame(
                        y, self._palette_id, self._colorbar,
                        self.rotation, self.flip_h, self.flip_v,
                        m,
                        self.cal_mode, self.cal_level, self.cal_span,
                        self.sharpen, self.emissivity,
                        self.show_osd, self.show_hotspot,
                        self.show_coldspot, self.show_crosshair,
                        self.alert_temp, self._blink,
                        self._frozen
                    )

                    if frame:
                        with self.lock:
                            self.current_frame = frame
                            self.frames_received += 1
                        if self.frames_received % (FPS * 30) == 0:
                            log(f"Frames: {self.frames_received}"
                                + (f", T: {m['t_min']:.1f}–{m['t_max']:.1f}°C"
                                   if m.get("valid") else ""))

                proc.stdout.close()
                proc.wait()

            except Exception as e:
                log(f"Capture-Fehler: {e}")

            if self.running:
                log("Kamera-Neustart in 3s...")
                time.sleep(3)

    def start_recording(self):
        """Startet den stündlichen Aufzeichnungs-Thread."""
        threading.Thread(target=self._record_loop, daemon=True).start()

    def _record_loop(self):
        """
        Stündliche MP4-Aufzeichnung via ffmpeg.

        Schreibt die verarbeiteten JPEG-Frames (inklusive Palette und Overlays)
        in eine MP4-Datei. Wechselt automatisch zur vollen Stunde.
        """
        while self.running:
            today = datetime.now().strftime("%Y-%m-%d")
            hour  = datetime.now().strftime("%H")
            rec_dir = os.path.join(RECORDINGS_DIR, today)
            os.makedirs(rec_dir, exist_ok=True)
            out = os.path.join(rec_dir, f"hikvision_{today}_{hour}00.mp4")

            log(f"Aufzeichnung: {out}")
            cmd = [
                "ffmpeg", "-y",
                "-f", "image2pipe",  # Eingabe: JPEG-Frames über stdin
                "-i", "-",
                "-framerate", str(FPS),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "28",          # Qualität (0=beste, 51=schlechteste)
                "-pix_fmt", "yuv420p", # Kompatibles Pixelformat
                out,
            ]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                start_hour = datetime.now().strftime("%H")
                last       = None

                while self.running and proc.poll() is None:
                    with self.lock:
                        fr = self.current_frame
                    if fr and fr is not last:
                        try:
                            proc.stdin.write(fr)  # JPEG-Frame an ffmpeg schicken
                            last = fr
                        except Exception:
                            break
                    # Stundenwechsel → neue Datei beginnen
                    if datetime.now().strftime("%H") != start_hour:
                        break
                    time.sleep(1.0 / FPS)

                try:
                    proc.stdin.close()
                    proc.wait(timeout=10)
                except Exception:
                    pass

                if os.path.exists(out):
                    mb = os.path.getsize(out) / 1_048_576
                    log(f"Gespeichert: {out} ({mb:.1f} MB)")

            except Exception as e:
                log(f"Recording-Fehler: {e}")
                time.sleep(10)


# ==============================================================================
#  WEB-UI (HTML/CSS/JS – komplett eingebettet, keine externen Abhängigkeiten)
# ==============================================================================

HTML_UI = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Thermal Camera Viewer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg:#0a0a0f;--panel:#111827;--border:#1f2937;--accent:#3b82f6;
  --text:#d1d5db;--sub:#6b7280;--hot:#ff6030;--cold:#40b0ff;
}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
     min-height:100vh;display:flex;flex-direction:column;}
header{display:flex;align-items:center;gap:10px;padding:8px 16px;
  background:#0d1117;border-bottom:1px solid var(--border);flex-shrink:0;}
.live-dot{width:8px;height:8px;border-radius:50%;background:#22c55e;
  box-shadow:0 0 6px #22c55e;animation:pulse 2s infinite;}
@keyframes pulse{50%{opacity:.3}}
header h1{font-size:.88rem;font-weight:600;letter-spacing:.1em;color:#f9fafb;flex:1;}
.badge{font-size:.68rem;color:var(--sub);border:1px solid var(--border);
  padding:2px 8px;border-radius:4px;white-space:nowrap;}
.layout{display:grid;grid-template-columns:160px 1fr 200px;flex:1;overflow:hidden;}
.sidebar{background:var(--panel);border-right:1px solid var(--border);
  overflow-y:auto;padding:10px 8px;display:flex;flex-direction:column;gap:10px;}
.sidebar-right{border-right:none;border-left:1px solid var(--border);}
.sec-title{font-size:.63rem;text-transform:uppercase;letter-spacing:.1em;
  color:var(--sub);margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid var(--border);}
section.ctrl{display:flex;flex-direction:column;gap:4px;}
.btn{padding:6px 8px;border:1px solid var(--border);background:#1c2333;color:var(--text);
  border-radius:5px;cursor:pointer;font-size:.75rem;transition:.12s;
  display:flex;align-items:center;justify-content:center;gap:5px;width:100%;}
.btn:hover{border-color:#4b5563;color:#fff;background:#263047;}
.btn.active{border-color:var(--accent);color:#93c5fd;background:#1e3a5f;}
.btn.success{border-color:#166534;color:#86efac;}
.btn.success:hover{background:#052e16;border-color:#22c55e;}
.btn.danger{border-color:#7f1d1d;color:#fca5a5;}
.btn.warn{border-color:#78350f;color:#fcd34d;}
.btn-row{display:flex;gap:4px;}.btn-row .btn{flex:1;}
.btn-sm{padding:4px 6px;font-size:.7rem;}
.pal-list{display:flex;flex-direction:column;gap:3px;}
.pal-swatch{display:flex;align-items:center;gap:6px;padding:5px 6px;
  border:1px solid var(--border);border-radius:4px;cursor:pointer;
  font-size:.74rem;color:var(--text);background:#1c2333;transition:.12s;}
.pal-swatch:hover{border-color:#4b5563;background:#263047;}
.pal-swatch.active{border-color:var(--accent);background:#1e3a5f;}
.swatch-bar{width:18px;height:12px;border-radius:2px;flex-shrink:0;}
.stream-col{display:flex;flex-direction:column;overflow:hidden;}
.stream-wrap{display:flex;align-items:center;justify-content:center;
  overflow:auto;padding:12px;background:var(--bg);flex:1;}
#stream{display:block;image-rendering:pixelated;cursor:crosshair;
  border:2px solid #1f2937;border-radius:4px;}
.zoom-bar{display:flex;align-items:center;gap:8px;padding:6px 12px;
  background:var(--panel);border-top:1px solid var(--border);flex-shrink:0;}
.zoom-bar .btn{width:auto;padding:4px 10px;}
#zoom-label{font-size:.75rem;color:var(--sub);min-width:38px;text-align:center;}
.zoom-slider{flex:1;accent-color:var(--accent);}
.temp-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px;}
.temp-card{background:#0d1117;border:1px solid var(--border);border-radius:5px;
  padding:6px 5px;text-align:center;}
.tc-lbl{font-size:.6rem;text-transform:uppercase;letter-spacing:.06em;color:var(--sub);margin-bottom:2px;}
.tc-val{font-size:.98rem;font-weight:700;font-variant-numeric:tabular-nums;}
.slider-row{display:flex;flex-direction:column;gap:2px;margin-top:4px;}
.slider-row label{font-size:.68rem;color:var(--sub);display:flex;justify-content:space-between;}
.slider-row input[type=range]{width:100%;accent-color:var(--accent);}
.radio-row{display:flex;gap:6px;margin-bottom:4px;}
.radio-row label{font-size:.74rem;display:flex;align-items:center;gap:4px;cursor:pointer;}
.radio-row input{accent-color:var(--accent);}
.alert-row{display:flex;align-items:center;gap:5px;margin-top:4px;flex-wrap:wrap;}
.alert-row input[type=number]{width:58px;background:#0d1117;border:1px solid var(--border);
  color:var(--text);border-radius:4px;padding:3px 5px;font-size:.74rem;}
.alert-row label{font-size:.72rem;color:var(--sub);}
footer{display:flex;gap:14px;padding:5px 12px;background:#0d1117;
  border-top:1px solid var(--border);font-size:.68rem;color:var(--sub);flex-shrink:0;flex-wrap:wrap;}
footer b{color:#9ca3af;}
#toast{position:fixed;bottom:16px;right:16px;background:#1f2937;color:#fff;
  padding:8px 16px;border-radius:6px;font-size:.8rem;border:1px solid var(--border);
  opacity:0;transition:opacity .25s;pointer-events:none;z-index:99;}
#toast.show{opacity:1;}
@media(max-width:700px){.layout{grid-template-columns:1fr;}.sidebar{display:none;}}
</style>
</head>
<body>
<header>
  <div class="live-dot"></div>
  <h1>Thermal Camera Viewer</h1>
  <div class="badge" id="fps-badge">— fps</div>
</header>

<div class="layout">
<!-- LINKE SIDEBAR: Palette, Bild-Controls, OSD, Aufnahme -->
<div class="sidebar">
  <section class="ctrl">
    <div class="sec-title">Farbpalette</div>
    <div class="pal-list" id="pal-list">
      <div class="pal-swatch active" data-p="ironbow"><div class="swatch-bar" style="background:linear-gradient(to right,#00008b,#4b0082,#ff0000,#ffd700)"></div>Ironbow</div>
      <div class="pal-swatch" data-p="rainbow"><div class="swatch-bar" style="background:linear-gradient(to right,#00f,#0f0,#ff0,#f00)"></div>Rainbow</div>
      <div class="pal-swatch" data-p="white_hot"><div class="swatch-bar" style="background:linear-gradient(to right,#000,#fff)"></div>White Hot</div>
      <div class="pal-swatch" data-p="black_hot"><div class="swatch-bar" style="background:linear-gradient(to right,#fff,#000)"></div>Black Hot</div>
      <div class="pal-swatch" data-p="lava"><div class="swatch-bar" style="background:linear-gradient(to right,#000,#8b0000,#ff8c00,#fff)"></div>Lava</div>
      <div class="pal-swatch" data-p="arctic"><div class="swatch-bar" style="background:linear-gradient(to right,#0ff,#f0f)"></div>Arctic</div>
      <div class="pal-swatch" data-p="fusion"><div class="swatch-bar" style="background:linear-gradient(to right,#0d0887,#cc4778,#f0f921)"></div>Fusion</div>
      <div class="pal-swatch" data-p="viridis"><div class="swatch-bar" style="background:linear-gradient(to right,#440154,#31688e,#35b779,#fde725)"></div>Viridis</div>
    </div>
  </section>
  <section class="ctrl">
    <div class="sec-title">Bild</div>
    <div class="btn-row">
      <button class="btn btn-sm" id="rot-ccw">↺</button>
      <button class="btn btn-sm" id="rot-cw">↻</button>
      <button class="btn btn-sm" id="rot-180">↕</button>
      <button class="btn btn-sm" id="rot-0">⊙</button>
    </div>
    <div class="btn-row">
      <button class="btn btn-sm" id="flip-h">⇔ H</button>
      <button class="btn btn-sm" id="flip-v">⇕ V</button>
    </div>
  </section>
  <section class="ctrl">
    <div class="sec-title">OSD</div>
    <button class="btn btn-sm active" id="btn-osd">OSD-Leiste</button>
    <button class="btn btn-sm active" id="btn-hotspot">🔥 Hotspot</button>
    <button class="btn btn-sm active" id="btn-coldspot">❄ Coldspot</button>
    <button class="btn btn-sm"        id="btn-crosshair">⊕ Fadenkreuz</button>
    <button class="btn btn-sm"        id="btn-freeze">⏸ Einfrieren</button>
  </section>
  <section class="ctrl">
    <div class="sec-title">Aufnahme</div>
    <button class="btn btn-sm success" id="snap-dl">📸 Download</button>
    <button class="btn btn-sm warn"    id="snap-save">💾 Speichern</button>
  </section>
</div>

<!-- MITTE: Stream + Zoom-Leiste -->
<div class="stream-col">
  <div class="stream-wrap" id="stream-wrap">
    <img id="stream" src="/stream.mjpeg" width="284" alt="thermal stream">
  </div>
  <div class="zoom-bar">
    <span style="font-size:.7rem;color:var(--sub)">Zoom</span>
    <button class="btn" id="zoom-out">－</button>
    <input type="range" class="zoom-slider" id="zoom-slider" min="50" max="400" value="100" step="5">
    <button class="btn" id="zoom-in">＋</button>
    <span id="zoom-label">100%</span>
    <button class="btn" id="zoom-fit">⊙</button>
    <button class="btn" id="fullscreen">⛶</button>
  </div>
</div>

<!-- RECHTE SIDEBAR: Temperatur, Kalibrierung, Erweitert -->
<div class="sidebar sidebar-right">
  <section class="ctrl">
    <div class="sec-title">Temperatur (live)</div>
    <div class="temp-grid">
      <div class="temp-card"><div class="tc-lbl">MIN</div><div class="tc-val" id="tv-min" style="color:var(--cold)">—</div></div>
      <div class="temp-card"><div class="tc-lbl">MAX</div><div class="tc-val" id="tv-max" style="color:var(--hot)">—</div></div>
      <div class="temp-card"><div class="tc-lbl">🔥 Hotspot</div><div class="tc-val" id="tv-hot" style="color:var(--hot)">—</div></div>
      <div class="temp-card"><div class="tc-lbl">❄ Coldspot</div><div class="tc-val" id="tv-cold" style="color:var(--cold)">—</div></div>
    </div>
    <div class="temp-card" style="margin-top:5px;">
      <div class="tc-lbl">Spanne (Szene)</div>
      <div class="tc-val" id="tv-span" style="font-size:.85rem;color:#9ca3af">—</div>
    </div>
  </section>
  <section class="ctrl">
    <div class="sec-title">Kalibrierung</div>
    <div class="radio-row">
      <label><input type="radio" name="cal" value="auto" checked id="cal-auto"> Auto</label>
      <label><input type="radio" name="cal" value="manual" id="cal-manual"> Manuell</label>
    </div>
    <div id="cal-manual-panel" style="display:none">
      <div class="slider-row">
        <label>Level (Mitte) <span id="lbl-level">30.0 °C</span></label>
        <input type="range" id="sl-level" min="-20" max="150" value="30" step="0.5">
      </div>
      <div class="slider-row">
        <label>Span (Breite) <span id="lbl-span">20.0 °C</span></label>
        <input type="range" id="sl-span" min="1" max="100" value="20" step="0.5">
      </div>
      <button class="btn btn-sm" id="cal-from-scene" style="margin-top:4px">↻ Aus Szene</button>
    </div>
  </section>
  <section class="ctrl">
    <div class="sec-title">Erweitert</div>
    <div class="slider-row">
      <label>Schärfe <span id="lbl-sharp">0%</span></label>
      <input type="range" id="sl-sharp" min="0" max="100" value="0" step="5">
    </div>
    <div class="slider-row">
      <label>Emissivität <span id="lbl-emiss">1.00</span></label>
      <input type="range" id="sl-emiss" min="10" max="100" value="100" step="1">
    </div>
    <div class="alert-row">
      <label>🚨 Alarm &gt;</label>
      <input type="number" id="inp-alert" placeholder="°C" step="0.5">
      <button class="btn btn-sm danger" id="btn-alert-set">Set</button>
      <button class="btn btn-sm"        id="btn-alert-clr">✕</button>
    </div>
  </section>
</div>
</div><!-- /layout -->

<footer>
  <span>Frames: <b id="st-frames">—</b></span>
  <span>Palette: <b id="st-pal">—</b></span>
  <span>Rotation: <b id="st-rot">0°</b></span>
  <span>Kalibr: <b id="st-cal">auto</b></span>
  <span id="st-alert-wrap" style="display:none">🚨 Alarm: <b id="st-alert-val">—</b></span>
</footer>
<div id="toast"></div>

<script>
const $ = id => document.getElementById(id);
function toast(msg,dur=2200){const t=$('toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),dur);}
function fmt(v){return v!=null?v.toFixed(1)+' °C':'—';}

// Palette
document.querySelectorAll('.pal-swatch').forEach(el=>{
  el.addEventListener('click',()=>{
    fetch('/palette/'+el.dataset.p).then(r=>r.json()).then(d=>{
      document.querySelectorAll('.pal-swatch').forEach(e=>e.classList.remove('active'));
      el.classList.add('active'); $('st-pal').textContent=d.palette; toast('Palette: '+d.palette);
    });
  });
});

// Rotation & Flip
['ccw','cw','180','0'].forEach(a=>$('rot-'+a).addEventListener('click',()=>{
  fetch('/rotate/'+a).then(r=>r.json()).then(d=>{$('st-rot').textContent=d.rotation+'°';toast('Rotation: '+d.rotation+'°');});
}));
$('flip-h').addEventListener('click',()=>fetch('/flip/h').then(r=>r.json()).then(d=>{$('flip-h').classList.toggle('active',d.flip_h);toast('H-Flip: '+(d.flip_h?'an':'aus'));}));
$('flip-v').addEventListener('click',()=>fetch('/flip/v').then(r=>r.json()).then(d=>{$('flip-v').classList.toggle('active',d.flip_v);toast('V-Flip: '+(d.flip_v?'an':'aus'));}));

// OSD
['osd','hotspot','coldspot','crosshair'].forEach(k=>{
  $('btn-'+k).addEventListener('click',()=>{
    fetch('/osd/'+k).then(r=>r.json()).then(d=>{$('btn-'+k).classList.toggle('active',d[k]);toast(k+': '+(d[k]?'an':'aus'));});
  });
});

// Einfrieren
$('btn-freeze').addEventListener('click',()=>{
  fetch('/freeze').then(r=>r.json()).then(d=>{
    $('btn-freeze').classList.toggle('active',d.frozen);
    $('btn-freeze').textContent=d.frozen?'▶ Fortsetzen':'⏸ Einfrieren';
    toast(d.frozen?'Bild eingefroren':'Stream fortgesetzt');
  });
});

// Zoom: setzt img.style.width in Pixeln (nicht CSS transform!)
const stream=$('stream');
const BASE_W=284; // Natürliche Bildbreite in Pixeln (28px Colorbar + 256px Thermal)
let zoomPct=100;
function applyZoom(){
  stream.style.width=Math.round(BASE_W*zoomPct/100)+'px';
  stream.style.height='auto';
  $('zoom-label').textContent=zoomPct+'%';
  $('zoom-slider').value=zoomPct;
}
$('zoom-in').addEventListener('click',()=>{zoomPct=Math.min(zoomPct+25,400);applyZoom();});
$('zoom-out').addEventListener('click',()=>{zoomPct=Math.max(zoomPct-25,50);applyZoom();});
$('zoom-fit').addEventListener('click',()=>{zoomPct=100;applyZoom();});
$('zoom-slider').addEventListener('input',e=>{zoomPct=parseInt(e.target.value);applyZoom();});
$('stream-wrap').addEventListener('wheel',e=>{e.preventDefault();zoomPct=e.deltaY<0?Math.min(zoomPct+10,400):Math.max(zoomPct-10,50);applyZoom();},{passive:false});
$('fullscreen').addEventListener('click',()=>{const el=$('stream-wrap');document.fullscreenElement?document.exitFullscreen():el.requestFullscreen();});

// Snapshot
$('snap-dl').addEventListener('click',()=>{
  fetch('/snapshot.jpg').then(r=>r.blob()).then(blob=>{
    const a=document.createElement('a');a.href=URL.createObjectURL(blob);
    a.download='thermal_'+new Date().toISOString().slice(0,19).replace(/[T:]/g,'-')+'.jpg';a.click();toast('Snapshot heruntergeladen');
  });
});
$('snap-save').addEventListener('click',()=>{
  fetch('/snapshot/save').then(r=>r.json()).then(d=>toast('Gespeichert: '+d.file,3500));
});

// Kalibrierung
let currentMeta=null;
function calModeChange(){
  const mode=document.querySelector('input[name=cal]:checked').value;
  $('cal-manual-panel').style.display=mode==='manual'?'block':'none';
  if(mode==='manual'&&currentMeta&&currentMeta.valid){
    const level=((currentMeta.t_min+currentMeta.t_max)/2).toFixed(1);
    const span=(currentMeta.t_max-currentMeta.t_min).toFixed(1);
    $('sl-level').value=level;$('lbl-level').textContent=level+' °C';
    $('sl-span').value=span;$('lbl-span').textContent=span+' °C';
  }
  sendCal(mode);
}
document.querySelectorAll('input[name=cal]').forEach(r=>r.addEventListener('change',calModeChange));
function sendCal(mode){
  const level=parseFloat($('sl-level').value),span=parseFloat($('sl-span').value);
  fetch(`/cal/set?mode=${mode}&level=${level}&span=${span}`).then(r=>r.json()).then(d=>{$('st-cal').textContent=d.mode;toast('Kalibr: '+d.mode);});
}
$('sl-level').addEventListener('input',e=>{$('lbl-level').textContent=parseFloat(e.target.value).toFixed(1)+' °C';if($('cal-manual').checked)sendCal('manual');});
$('sl-span').addEventListener('input',e=>{$('lbl-span').textContent=parseFloat(e.target.value).toFixed(1)+' °C';if($('cal-manual').checked)sendCal('manual');});
$('cal-from-scene').addEventListener('click',()=>{
  if(!currentMeta||!currentMeta.valid){toast('Keine Metadaten');return;}
  const level=((currentMeta.t_min+currentMeta.t_max)/2).toFixed(1),span=(currentMeta.t_max-currentMeta.t_min).toFixed(1);
  $('sl-level').value=level;$('lbl-level').textContent=level+' °C';
  $('sl-span').value=span;$('lbl-span').textContent=span+' °C';
  sendCal('manual');toast('Level/Span aus Szene übernommen');
});

// Erweitert
$('sl-sharp').addEventListener('input',e=>{const v=parseInt(e.target.value);$('lbl-sharp').textContent=v+'%';fetch('/enhance/sharpen?v='+v);});
$('sl-emiss').addEventListener('input',e=>{const v=(parseInt(e.target.value)/100).toFixed(2);$('lbl-emiss').textContent=v;fetch('/enhance/emissivity?v='+v);});
$('btn-alert-set').addEventListener('click',()=>{
  const v=parseFloat($('inp-alert').value);if(isNaN(v)){toast('Ungültig');return;}
  fetch('/alert/set?temp='+v).then(r=>r.json()).then(d=>{
    $('st-alert-wrap').style.display='';$('st-alert-val').textContent=d.alert.toFixed(1)+'°C';toast('Alarm > '+d.alert.toFixed(1)+'°C');
  });
});
$('btn-alert-clr').addEventListener('click',()=>{
  fetch('/alert/clear').then(r=>r.json()).then(()=>{$('st-alert-wrap').style.display='none';$('inp-alert').value='';toast('Alarm deaktiviert');});
});

// Status-Polling alle 2 Sekunden
let frameCount=0;
setInterval(()=>{
  fetch('/status').then(r=>r.json()).then(d=>{
    const delta=d.frames-frameCount;frameCount=d.frames;
    $('fps-badge').textContent=Math.round(delta/2)+' fps';
    $('st-frames').textContent=d.frames;$('st-pal').textContent=d.palette;
    $('st-rot').textContent=d.rotation+'°';$('st-cal').textContent=d.cal_mode;
    document.querySelectorAll('.pal-swatch').forEach(e=>e.classList.toggle('active',e.dataset.p===d.palette));
    ['osd','hotspot','coldspot','crosshair'].forEach(k=>{const b=$('btn-'+k);if(b)b.classList.toggle('active',!!d[k]);});
    $('flip-h').classList.toggle('active',d.flip_h);$('flip-v').classList.toggle('active',d.flip_v);
    const t=d.temp;currentMeta=t;
    if(t&&t.valid){
      $('tv-min').textContent=fmt(t.t_min);$('tv-max').textContent=fmt(t.t_max);
      $('tv-hot').textContent=fmt(t.hs_temp);$('tv-cold').textContent=fmt(t.cs_temp);
      $('tv-span').textContent=(t.t_max-t.t_min).toFixed(1)+' °C';
    }
  }).catch(()=>{});
},2000);
</script>
</body></html>
"""


# ==============================================================================
#  HTTP HANDLER
# ==============================================================================

class StreamHandler(http.server.BaseHTTPRequestHandler):
    """
    HTTP-Request-Handler für alle Endpunkte des Thermal-Servers.

    Jede Verbindung läuft in einem eigenen Thread (ThreadingMixIn), daher
    kann der MJPEG-Stream parallel zu allen anderen Anfragen bedient werden.
    """
    camera = None  # Wird in main() gesetzt (Klassenattribut, global für alle Threads)

    def send_json(self, data: dict, code: int = 200):
        """Sendet eine JSON-Antwort."""
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        p      = parsed.path.rstrip("/")
        qs     = parse_qs(parsed.query)
        cam    = StreamHandler.camera

        # ── MJPEG-Stream (Endlos-Loop, hält Verbindung offen) ────────────────
        if p == "/stream.mjpeg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with cam.lock:
                        frame = cam.current_frame
                    if frame:
                        self.wfile.write(b"--FRAME\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(1.0 / FPS)
            except Exception:
                pass  # Client hat Verbindung getrennt

        # ── Einzelbild ────────────────────────────────────────────────────────
        elif p == "/snapshot.jpg":
            with cam.lock:
                frame = cam.current_frame
            if frame:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
            else:
                self.send_response(503)
                self.end_headers()

        elif p == "/snapshot/save":
            path = cam.save_snapshot()
            self.send_json({"ok": bool(path), "file": os.path.basename(path or "")})

        # ── Bild-Einstellungen ────────────────────────────────────────────────
        elif p.startswith("/palette/"):
            cam.set_palette(p.split("/palette/")[1])
            self.send_json({"palette": cam.palette_name})

        elif p.startswith("/rotate/"):
            cam.rotate(p.split("/rotate/")[1])
            self.send_json({"rotation": cam.rotation})

        elif p.startswith("/flip/"):
            axis = p.split("/flip/")[1]
            if axis == "h": cam.toggle_flip_h()
            elif axis == "v": cam.toggle_flip_v()
            self.send_json({"flip_h": cam.flip_h, "flip_v": cam.flip_v})

        # ── OSD-Toggles ───────────────────────────────────────────────────────
        elif p.startswith("/osd/"):
            key = p.split("/osd/")[1]
            if key == "osd":        cam.show_osd       = not cam.show_osd
            elif key == "hotspot":  cam.show_hotspot    = not cam.show_hotspot
            elif key == "coldspot": cam.show_coldspot   = not cam.show_coldspot
            elif key == "crosshair":cam.show_crosshair  = not cam.show_crosshair
            self.send_json({
                "osd": cam.show_osd, "hotspot": cam.show_hotspot,
                "coldspot": cam.show_coldspot, "crosshair": cam.show_crosshair,
            })

        elif p == "/freeze":
            frozen = cam.toggle_freeze()
            self.send_json({"frozen": frozen})

        # ── Kalibrierung ──────────────────────────────────────────────────────
        elif p == "/cal/set":
            mode  = qs.get("mode",  ["auto"])[0]
            level = float(qs.get("level", [str(cam.cal_level)])[0])
            span  = float(qs.get("span",  [str(cam.cal_span)])[0])
            cam.cal_mode  = mode if mode in ("auto", "manual") else "auto"
            cam.cal_level = max(-40, min(500, level))
            cam.cal_span  = max(1,   min(200, span))
            cam._update_palette()
            self.send_json({"mode": cam.cal_mode, "level": cam.cal_level, "span": cam.cal_span})

        # ── Erweiterte Einstellungen ──────────────────────────────────────────
        elif p.startswith("/enhance/sharpen"):
            v = float(qs.get("v", ["0"])[0])
            cam.sharpen = max(0, min(100, v))
            self.send_json({"sharpen": cam.sharpen})

        elif p.startswith("/enhance/emissivity"):
            v = float(qs.get("v", ["1.0"])[0])
            cam.emissivity = max(0.1, min(1.0, v))
            self.send_json({"emissivity": cam.emissivity})

        elif p == "/alert/set":
            v = float(qs.get("temp", ["50"])[0])
            cam.alert_temp = v
            self.send_json({"alert": cam.alert_temp})

        elif p == "/alert/clear":
            cam.alert_temp = None
            self.send_json({"alert": None})

        # ── Status (alle Zustände als JSON) ───────────────────────────────────
        elif p == "/status":
            with cam.meta_lock:
                m = cam.last_meta
            self.send_json({
                "palette":    cam.palette_name,
                "frames":     cam.frames_received,
                "rotation":   cam.rotation,
                "flip_h":     cam.flip_h,
                "flip_v":     cam.flip_v,
                "osd":        cam.show_osd,
                "hotspot":    cam.show_hotspot,
                "coldspot":   cam.show_coldspot,
                "crosshair":  cam.show_crosshair,
                "cal_mode":   cam.cal_mode,
                "cal_level":  cam.cal_level,
                "cal_span":   cam.cal_span,
                "sharpen":    cam.sharpen,
                "emissivity": cam.emissivity,
                "alert":      cam.alert_temp,
                "frozen":     cam._frozen is not None,
                "temp": {
                    "valid":   m.get("valid", False),
                    "t_min":   round(m["t_min"],   2) if m.get("valid") else None,
                    "t_max":   round(m["t_max"],   2) if m.get("valid") else None,
                    "hs_temp": round(m["hs_temp"], 2) if m.get("valid") else None,
                    "cs_temp": round(m["cs_temp"], 2) if m.get("valid") else None,
                },
            })

        # ── Web-UI ────────────────────────────────────────────────────────────
        elif p in ("", "/", "/index.html"):
            body = HTML_UI.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # HTTP-Access-Log unterdrücken (Rauschen im Journal)


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    # Gerät vorhanden?
    if not os.path.exists(DEVICE):
        log(f"FEHLER: {DEVICE} nicht gefunden. Kamera angesteckt und Permissions gesetzt?")
        sys.exit(1)

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR,   exist_ok=True)
    log("=== Hikvision Thermal Camera Server ===")

    # Kamera initialisieren und Threads starten
    cam = ThermalCamera()
    StreamHandler.camera = cam
    cam.start_capture()

    # Auf ersten Frame warten (max. 12 Sekunden)
    for _ in range(60):
        if cam.current_frame:
            break
        time.sleep(0.2)

    if cam.current_frame:
        log(f"Erster Frame: {len(cam.current_frame)} Bytes")
    else:
        log("Warnung: kein Frame nach 12s – Server startet trotzdem")

    cam.start_recording()

    # Threaded TCP-Server: jede Verbindung bekommt eigenen Thread
    # → MJPEG-Stream blockiert nicht andere Requests (Buttons, Palette etc.)
    class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True  # Sofortiger Neustart ohne TIME_WAIT
        daemon_threads      = True  # Threads enden wenn Hauptprozess endet

    with Server(("0.0.0.0", PORT), StreamHandler) as httpd:
        log(f"Web-UI:   http://0.0.0.0:{PORT}/")
        log(f"Stream:   http://0.0.0.0:{PORT}/stream.mjpeg")
        log(f"Snapshot: http://0.0.0.0:{PORT}/snapshot.jpg")
        log(f"Status:   http://0.0.0.0:{PORT}/status")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            log("Server gestoppt")
            cam.running = False


if __name__ == "__main__":
    main()
