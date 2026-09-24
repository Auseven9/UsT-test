import 'dart:async';
import 'dart:io';

import 'package:get/get.dart';

import '../models/resource_sample.dart';
import 'llm_service.dart';

/// Live CPU/memory/generation-speed monitoring.
///
/// Honesty note, because this is easy to fake and that would be worse than
/// not having it: Android exposes no public, permission-free API for GPU
/// utilization or NPU utilization at all — those chips' counters are vendor
/// APIs (Qualcomm QNN, Mali profiler, Exynos NPU SDK, …), not something a
/// normal app can read. This service reads real CPU and memory numbers from
/// `/proc` (Android/Linux only), plus a best-effort Adreno-only GPU clock
/// speed where the sysfs path happens to exist. It never fabricates a GPU or
/// NPU utilization percentage — where that data isn't obtainable, [isSupported]
/// and null fields say so, and the UI must show "not available", not a chart.
class ResourceMonitorService extends GetxService {
  static const int maxSamples = 60; // 60s of history at 1 sample/sec
  static const _pollInterval = Duration(seconds: 1);

  final samples = <ResourceSample>[].obs;
  final isMonitoring = false.obs;

  /// Whether `/proc`-based reads are even attempted on this platform. False
  /// on iOS/macOS/Windows (sandboxed or simply not `/proc`-based) — the UI
  /// uses this to show an honest "not supported on this platform" state
  /// instead of empty charts.
  bool get isSupported => Platform.isAndroid || Platform.isLinux;

  Timer? _timer;
  List<int>? _lastCpuJiffies; // [idleJiffies, totalJiffies]

  Future<ResourceMonitorService> init() async => this;

  void start() {
    if (!isSupported || _timer != null) return;
    isMonitoring.value = true;
    _timer = Timer.periodic(_pollInterval, (_) => _sample());
    unawaited(_sample()); // first sample immediately, don't wait a full second
  }

  void stop() {
    _timer?.cancel();
    _timer = null;
    isMonitoring.value = false;
  }

  void clear() {
    samples.clear();
    _lastCpuJiffies = null;
  }

  Future<void> _sample() async {
    final cpu = await _readSystemCpuPercent();
    final procMem = await _readProcessMemoryMb();
    final sysMem = await _readSystemMemoryMb();
    final gpuClock = await _readAdrenoGpuClockMhz();

    var tps = 0.0;
    try {
      final llm = Get.find<LlmService>();
      tps = llm.isGenerating.value ? llm.tokensPerSecond.value : 0.0;
    } catch (_) {}

    final sample = ResourceSample(
      time: DateTime.now(),
      systemCpuPercent: cpu,
      processMemoryMb: procMem,
      systemMemoryUsedMb: sysMem?.$1,
      systemMemoryTotalMb: sysMem?.$2,
      adrenoGpuClockMhz: gpuClock,
      tokensPerSecond: tps,
    );

    samples.add(sample);
    while (samples.length > maxSamples) {
      samples.removeAt(0);
    }
  }

  /// System-wide CPU usage across all cores, via `/proc/stat` deltas. This
  /// ratio-of-deltas approach sidesteps needing to know the kernel's jiffy
  /// rate (`USER_HZ`, not portably queryable from Dart) — only relies on
  /// idle-time growing slower than total time when the CPU is busier.
  Future<double?> _readSystemCpuPercent() async {
    try {
      final lines = await File('/proc/stat').readAsLines();
      final cpuLine = lines.firstWhere(
        (l) => l.startsWith('cpu '),
        orElse: () => '',
      );
      if (cpuLine.isEmpty) return null;

      final parts = cpuLine
          .trim()
          .split(RegExp(r'\s+'))
          .skip(1)
          .map((s) => int.tryParse(s) ?? 0)
          .toList();
      if (parts.length < 4) return null;

      // user nice system idle iowait irq softirq steal ...
      final idle = parts[3] + (parts.length > 4 ? parts[4] : 0);
      final total = parts.fold<int>(0, (a, b) => a + b);

      final last = _lastCpuJiffies;
      _lastCpuJiffies = [idle, total];
      if (last == null) return null; // no delta on the very first read

      final deltaIdle = idle - last[0];
      final deltaTotal = total - last[1];
      if (deltaTotal <= 0) return null;

      return (100 * (1 - deltaIdle / deltaTotal)).clamp(0, 100).toDouble();
    } catch (_) {
      return null;
    }
  }

  /// This app's own resident memory usage, from `/proc/self/status`.
  Future<double?> _readProcessMemoryMb() async {
    try {
      final lines = await File('/proc/self/status').readAsLines();
      final line = lines.firstWhere(
        (l) => l.startsWith('VmRSS:'),
        orElse: () => '',
      );
      if (line.isEmpty) return null;
      final kb = int.tryParse(line.replaceAll(RegExp(r'[^0-9]'), ''));
      return kb == null ? null : kb / 1024.0;
    } catch (_) {
      return null;
    }
  }

  /// System-wide memory in use vs. total, from `/proc/meminfo`.
  Future<(double, double)?> _readSystemMemoryMb() async {
    try {
      final lines = await File('/proc/meminfo').readAsLines();
      int? totalKb;
      int? availableKb;
      for (final line in lines) {
        if (line.startsWith('MemTotal:')) {
          totalKb = int.tryParse(line.replaceAll(RegExp(r'[^0-9]'), ''));
        } else if (line.startsWith('MemAvailable:')) {
          availableKb = int.tryParse(line.replaceAll(RegExp(r'[^0-9]'), ''));
        }
      }
      if (totalKb == null || availableKb == null) return null;
      final usedKb = totalKb - availableKb;
      return (usedKb / 1024.0, totalKb / 1024.0);
    } catch (_) {
      return null;
    }
  }

  /// Best-effort Adreno-only GPU clock speed. Returns null on every device
  /// where this sysfs path doesn't exist — which is most non-Qualcomm
  /// devices, and even some Qualcomm ones depending on kernel config.
  Future<int?> _readAdrenoGpuClockMhz() async {
    try {
      final file = File('/sys/class/kgsl/kgsl-3d0/gpuclk');
      if (!await file.exists()) return null;
      final raw = (await file.readAsString()).trim();
      final hz = int.tryParse(raw);
      if (hz == null) return null;
      return (hz / 1000000).round();
    } catch (_) {
      return null;
    }
  }

  @override
  void onClose() {
    stop();
    super.onClose();
  }
}
