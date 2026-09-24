/// One point-in-time reading of device/app resource usage. Any field can be
/// null when that reading isn't available on the current platform — the UI
/// must handle that honestly (show "not available"), never substitute a
/// fake number.
class ResourceSample {
  final DateTime time;

  /// Aggregate CPU usage across all cores, 0-100. Computed from `/proc/stat`
  /// deltas — Android/Linux only.
  final double? systemCpuPercent;

  /// This app's own resident memory (VmRSS from `/proc/self/status`), MB.
  final double? processMemoryMb;

  /// System-wide memory in use (MemTotal - MemAvailable), MB.
  final double? systemMemoryUsedMb;
  final double? systemMemoryTotalMb;

  /// Best-effort Adreno GPU clock in MHz, read from a vendor-specific sysfs
  /// path. Null on every non-Adreno device (Mali, PowerVR, Apple GPUs, and
  /// desktop GPUs all use different, unreadable-without-root interfaces) —
  /// this is NOT a general GPU-utilization reading, only a device-specific
  /// clock speed that happens to be exposed on some Qualcomm SoCs.
  final int? adrenoGpuClockMhz;

  /// Live generation speed from the loaded chat model, tokens/sec. 0 when
  /// idle (not an "unavailable" case — 0 is a real, meaningful value here).
  final double tokensPerSecond;

  const ResourceSample({
    required this.time,
    required this.systemCpuPercent,
    required this.processMemoryMb,
    required this.systemMemoryUsedMb,
    required this.systemMemoryTotalMb,
    required this.adrenoGpuClockMhz,
    required this.tokensPerSecond,
  });
}
