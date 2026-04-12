# Hikvision Thermal Camera - MJPEG Stream Server

A complete Python server for HIKMicro/Hikvision USB thermal cameras (tested with HIK Camera `2bdf:0102`).
Runs in a Proxmox LXC container (Ubuntu 22.04), but works on any Linux system with USB access to the camera.

## Features

- **MJPEG live stream** over HTTP (port 8890)
- **8 color palettes**: Ironbow, Rainbow, White Hot, Black Hot, Lava, Arctic, Fusion, Viridis
- **Real temperature values** from camera metadata (no guessing, no external calibration needed)
- **Hotspot/Coldspot markers** with temperature display in the image
- **OSD bar** (MIN/CENTER/MAX temperature)
- **Calibration Auto/Manual** with Level & Span (like the Hikvision app)
- **Sharpening filter** (Unsharp Mask)
- **Emissivity correction** (for non-blackbody surfaces)
- **Temperature alarm** with visual alert (red border flash)
- **Freeze frame** for detail inspection
- **Rotation** (0/90/180/270) and **Flip** (H/V) - server-side
- **Zoom** (50%-400%, mouse wheel) - client-side
- **Snapshot download** (browser) and **server-side saving**
- **Hourly MP4 recording** with ffmpeg (h264)
- **Web UI** in a 3-column layout (no external dependencies, pure HTML/JS)

## Screenshots

The web UI provides a full thermal camera control interface directly in the browser:

![Web UI](https://raw.githubusercontent.com/pit711/hikivision/main/screenshot.png)

*(Add a screenshot.png to the repo to display it here)*

## Requirements

### Hardware

- HIKMicro / Hikvision USB thermal camera (UVC compatible)
  - Tested: `2bdf:0102` (HIK Camera)
  - Other UVC thermal cameras may work - adjust `WIDTH`/`HEIGHT` if needed

### Software

```bash
# Ubuntu/Debian
sudo apt install python3 python3-pip ffmpeg v4l-utils

pip3 install numpy opencv-python-headless
```

> `opencv-python-headless` instead of `opencv-python` - no GUI needed, smaller installation.

## Installation

### 1. Connect camera and verify

```bash
lsusb | grep -i hik
# Output: Bus 001 Device 008: ID 2bdf:0102 HIK Camera

v4l2-ctl --list-devices
# Output: HIK Camera (usb-...): /dev/video0

# Check available formats:
ffmpeg -f v4l2 -list_formats all -i /dev/video0
# Expected: yuyv422 : YUYV 4:2:2 : 256x344 ...
```

### 2. Copy script and service

```bash
cp hikvision_thermal.py /root/recordings/
cp hikvision-thermal.service /etc/systemd/system/

# Create directories
mkdir -p /var/lib/vz/recordings/streams/hikvision
mkdir -p /var/lib/vz/recordings/snapshots
```

> Recording paths can be changed in the `.py` file under `RECORDINGS_DIR` and `SNAPSHOT_DIR`.

### 3. Permissions for /dev/video0

```bash
# Temporary (until next reboot):
chmod 666 /dev/video0

# Permanent via udev rule:
echo 'SUBSYSTEM=="video4linux", ATTRS{idVendor}=="2bdf", ATTRS{idProduct}=="0102", MODE="0666"' \
  > /etc/udev/rules.d/99-hikvision-thermal.rules
udevadm control --reload-rules && udevadm trigger
```

### 4. Enable the service

```bash
systemctl daemon-reload
systemctl enable --now hikvision-thermal

# Check status:
systemctl status hikvision-thermal

# Live logs:
journalctl -u hikvision-thermal -f
```

### 5. Open in browser

```
Web UI:   http://<IP>:8890/
Stream:   http://<IP>:8890/stream.mjpeg
Snapshot: http://<IP>:8890/snapshot.jpg
Status:   http://<IP>:8890/status
```

## Proxmox LXC Setup

If running inside a Proxmox LXC container, you need to pass through the USB device.

### On the Proxmox host

```bash
# 1. Find device numbers
ls -la /dev/video0
# crw-rw---- 1 root video 81, 0 ...  -> Major=81, Minor=0

# 2. Add to LXC config (/etc/pve/lxc/<CTID>.conf)
cat >> /etc/pve/lxc/<CTID>.conf << 'EOF'
lxc.cgroup2.devices.allow: c 81:0 rwm
lxc.cgroup2.devices.allow: c 81:1 rwm
lxc.mount.entry: /dev/video0 dev/video0 none bind,optional,create=file
lxc.mount.entry: /dev/video1 dev/video1 none bind,optional,create=file
EOF

# 3. Set permissions on host (for unprivileged containers)
chmod 666 /dev/video0 /dev/video1

# 4. Restart the container
pct stop <CTID> && pct start <CTID>

# 5. Verify
pct exec <CTID> -- ls -la /dev/video0
```

### Inside the container (Ubuntu 22.04)

```bash
apt install python3 python3-pip ffmpeg
pip3 install numpy opencv-python-headless

mkdir -p /root/recordings
mkdir -p /var/lib/vz/recordings/streams/hikvision
mkdir -p /var/lib/vz/recordings/snapshots

cp hikvision_thermal.py /root/recordings/
cp hikvision-thermal.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now hikvision-thermal
```

## Camera Frame Structure (Technical Background)

The UVC device delivers a YUYV frame of 256x344 pixels containing **three regions**:

```
Rows   0 - 191  -> Thermal image (256x192 pixels, YUYV)
Rows 192 - 295  -> Metadata & visible camera thumbnail
Rows 296 - 343  -> Padding / empty (Y=128)
```

### Metadata Format (Row 192)

Row 192 contains binary structure data in the YUYV bytestream (all 512 bytes are used):

| Offset (Bytes) | Type      | Content                         |
|-----------------|-----------|---------------------------------|
| 0-1             | uint16 LE | T_min scene (degC x 100)        |
| 2-3             | uint16 LE | T_max scene (degC x 100)        |
| 16-17           | uint16 LE | Frame height (= 192)            |
| 18-19           | uint16 LE | Frame width (= 256)             |
| 40-41           | uint16 LE | Hotspot temperature (degC x 100)|
| 42-43           | uint16 LE | Hotspot X (pixel)               |
| 44-45           | uint16 LE | Hotspot Y (pixel)               |
| 50-51           | uint16 LE | Coldspot temperature (degC x 100)|

Row 193 starts with magic bytes `0xAABBCCDD` (LE) and contains the sensor calibration range (~-40 degC to 327 degC).

### Temperature Calculation

```
T_pixel = T_min + (Y_value / 255.0) x (T_max - T_min)
```

In manual calibration mode:

```
T_pixel_absolute = meta_T_min + (Y / 255) x (meta_T_max - meta_T_min)
T_display        = clip((T_pixel - cal_Level + cal_Span/2) / cal_Span x 255, 0, 255)
```

## HTTP API

| Endpoint                                      | Method | Description                          |
|-----------------------------------------------|--------|--------------------------------------|
| `/`                                           | GET    | Web UI                               |
| `/stream.mjpeg`                               | GET    | MJPEG live stream                    |
| `/snapshot.jpg`                               | GET    | Current frame as JPEG                |
| `/snapshot/save`                              | GET    | Save frame to server disk            |
| `/status`                                     | GET    | JSON with all current states         |
| `/palette/<name>`                             | GET    | Switch color palette                 |
| `/rotate/<cw\|ccw\|180\|0>`                  | GET    | Rotate image                         |
| `/flip/<h\|v>`                                | GET    | Mirror image                         |
| `/osd/<osd\|hotspot\|coldspot\|crosshair>`   | GET    | Toggle OSD elements                  |
| `/freeze`                                     | GET    | Freeze/resume frame                  |
| `/cal/set?mode=auto`                          | GET    | Set calibration to auto              |
| `/cal/set?mode=manual&level=X&span=Y`        | GET    | Manual calibration                   |
| `/enhance/sharpen?v=0-100`                    | GET    | Set sharpening strength              |
| `/enhance/emissivity?v=0.1-1.0`              | GET    | Set emissivity                       |
| `/alert/set?temp=X`                           | GET    | Set alarm threshold                  |
| `/alert/clear`                                | GET    | Disable alarm                        |

## Configuration

These variables can be adjusted at the top of `hikvision_thermal.py`:

| Variable         | Default                                      | Description                     |
|------------------|----------------------------------------------|---------------------------------|
| `PORT`           | `8890`                                       | HTTP port                       |
| `DEVICE`         | `/dev/video0`                                | V4L2 device                     |
| `WIDTH`          | `256`                                        | Camera width (pixels)           |
| `HEIGHT`         | `344`                                        | Full UVC frame height           |
| `THERMAL_HEIGHT` | `192`                                        | Actual thermal image height     |
| `FPS`            | `9`                                          | Target framerate                |
| `RECORDINGS_DIR` | `/var/lib/vz/recordings/streams/hikvision`   | MP4 recording directory         |
| `SNAPSHOT_DIR`   | `/var/lib/vz/recordings/snapshots`           | Snapshot storage directory      |

## Known Limitations

- The camera cannot be opened by two processes simultaneously. The service must be stopped before using other tools (v4l2-ctl, cheese, etc.).
- In an **unprivileged** LXC container, device permissions cannot be set from inside - must be done from the Proxmox host (`chmod 666 /dev/video0`).
- The metadata offsets were determined through reverse engineering and are confirmed for `2bdf:0102`. Other Hikvision models may differ.

## License

MIT - free to use, modify, and redistribute.
Not affiliated with or supported by Hikvision / HIKMicro.
