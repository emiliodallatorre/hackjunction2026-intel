"""Device utilisation and temperature for the dashboard's CPU / iGPU / NPU widgets.

Everything is read from sysfs and /proc, so there is no dependency on intel_gpu_top (which needs
root) and no sampling subprocess. Each reading is best effort: a device whose counters this kernel
does not expose reports ``None`` and the widget shows it as unavailable rather than as 0%.

Sources, in the order they are tried:

* CPU   ``/proc/stat`` busy jiffies between two reads; temperature from a hwmon named
        ``coretemp``, ``k10temp`` or ``zenpower``.
* iGPU  ``gpu_busy_percent`` (i915 on some kernels), else the ratio of ``engine/*/busy``
        milliseconds to wall time. The ``xe`` driver publishes neither, so there the actual clock
        (``freq0/act_freq`` over ``max_freq``) stands in: it is an activity proxy, not true
        occupancy, and it is labelled as such. True occupancy would need intel_gpu_top as root.
* NPU   ``/sys/class/accel/accel*/device/npu_busy_time_us`` (intel_vpu), a microsecond busy
        counter, plus ``npu_memory_utilization`` in bytes.

Temperatures come from hwmon, which on many laptops only publishes the CPU package: the iGPU and
NPU then show usage without a temperature rather than a made-up one.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# hwmon names vary by platform; acpitz_0 is what the HP OmniBook reports alongside coretemp.
CPU_TEMP_NAMES = ("coretemp", "k10temp", "zenpower", "acpitz", "acpitz_0")
GPU_HWMON_NAMES = ("i915", "xe", "amdgpu")
NPU_HWMON_NAMES = ("intel_vpu", "ivpu")


@dataclass
class Reading:
    """One device's state. ``percent`` and ``celsius`` are None when unavailable.

    ``approximate`` marks a percentage derived from a proxy (the GPU clock) rather than a real
    busy counter, so the dashboard can show it as an estimate.
    """

    name: str
    percent: float | None = None
    celsius: float | None = None
    approximate: bool = False


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _hwmon_temp(names: tuple[str, ...]) -> float | None:
    """Highest temperature from the first hwmon whose name matches, in degrees Celsius."""
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (hwmon / "name").read_text().strip() not in names:
                continue
        except OSError:
            continue
        values = [v for v in (_read_int(p) for p in sorted(hwmon.glob("temp*_input"))) if v is not None]
        if values:
            return max(values) / 1000.0
    return None


class _Rate:
    """Turns a monotonically increasing busy counter into a percentage between two samples."""

    def __init__(self, scale: float):
        self._scale = scale  # counter units per second at 100% busy
        self._last: tuple[float, int] | None = None

    def update(self, counter: int | None) -> float | None:
        if counter is None:
            self._last = None
            return None
        now = time.monotonic()
        previous, self._last = self._last, (now, counter)
        if previous is None:
            return None
        elapsed = now - previous[0]
        if elapsed <= 0:
            return None
        delta = counter - previous[1]
        if delta < 0:  # counter reset (driver reload)
            return None
        return max(0.0, min(100.0, 100.0 * delta / (elapsed * self._scale)))


class Telemetry:
    """Samples CPU, iGPU and NPU. Call ``read()`` on a timer; the first call has no percentages."""

    def __init__(self) -> None:
        self._cpu_last: tuple[int, int] | None = None
        self._gpu_rate = _Rate(1_000.0)  # engine busy is milliseconds of wall time
        self._npu_rate = _Rate(1_000_000.0)  # npu_busy_time_us is microseconds
        self._gpu_card = self._find_card()
        self._npu_dir = self._find_npu()
        logger.info("Telemetry: gpu=%s npu=%s", self._gpu_card or "none", self._npu_dir or "none")

    @staticmethod
    def _find_card() -> Path | None:
        # "card0", not connectors like "card1-DP-1", which also match a looser glob.
        for card in sorted(c for c in Path("/sys/class/drm").glob("card[0-9]*") if "-" not in c.name):
            device = card / "device"
            if (device / "gpu_busy_percent").exists() or (device / "engine").is_dir():
                return device
            if next(device.glob("tile*/gt*/freq0/act_freq"), None) is not None:
                return device
        return None

    @staticmethod
    def _find_npu() -> Path | None:
        """The directory holding ``npu_busy_time_us`` (the accel device's PCI node, not accel0)."""
        for accel in sorted(Path("/sys/class/accel").glob("accel*")):
            for candidate in (accel / "device", accel):
                if (candidate / "npu_busy_time_us").exists():
                    return candidate
        for platform_dir in sorted(Path("/sys/devices/platform").glob("*vpu*")):
            if (platform_dir / "npu_busy_time_us").exists():
                return platform_dir
        return None

    def _cpu(self) -> Reading:
        percent = None
        try:
            fields = [int(v) for v in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
        except (OSError, ValueError, IndexError):
            fields = []
        if fields:
            idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
            total = sum(fields)
            previous, self._cpu_last = self._cpu_last, (total, idle)
            if previous is not None:
                d_total, d_idle = total - previous[0], idle - previous[1]
                if d_total > 0:
                    percent = max(0.0, min(100.0, 100.0 * (d_total - d_idle) / d_total))
        return Reading("CPU", percent, _hwmon_temp(CPU_TEMP_NAMES))

    def _gpu(self) -> Reading:
        percent = None
        approximate = False
        if self._gpu_card is not None:
            direct = _read_int(self._gpu_card / "gpu_busy_percent")
            if direct is not None:
                percent = float(max(0, min(100, direct)))
            else:
                busy = [_read_int(p) for p in sorted(self._gpu_card.glob("engine/*/busy"))]
                known = [v for v in busy if v is not None]
                percent = self._gpu_rate.update(max(known) if known else None)
                if percent is None:
                    percent = self._gpu_clock_ratio()
                    approximate = percent is not None
        return Reading("iGPU", percent, _hwmon_temp(GPU_HWMON_NAMES), approximate)

    def _gpu_clock_ratio(self) -> float | None:
        """Current clock as a fraction of the maximum: an activity proxy for the xe driver."""
        if self._gpu_card is None:
            return None
        act = next(self._gpu_card.glob("tile*/gt*/freq0/act_freq"), None)
        if act is None:
            return None
        current, maximum = _read_int(act), _read_int(act.with_name("max_freq"))
        if current is None or not maximum:
            return None
        return max(0.0, min(100.0, 100.0 * current / maximum))

    def _npu(self) -> Reading:
        counter = None
        if self._npu_dir is not None:
            for name in ("npu_busy_time_us", "busy_time_us"):
                counter = _read_int(self._npu_dir / name)
                if counter is not None:
                    break
        return Reading("NPU", self._npu_rate.update(counter), _hwmon_temp(NPU_HWMON_NAMES))

    def read(self) -> list[Reading]:
        return [self._cpu(), self._gpu(), self._npu()]
