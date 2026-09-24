import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../theme/app_colors.dart';
import '../services/resource_monitor_service.dart';
import '../services/chat_storage_service.dart';
import '../widgets/live_line_chart.dart';

/// Live CPU/memory/generation-speed monitor. Only polls while this screen is
/// open — starts monitoring in [initState], stops in [dispose] — so it never
/// costs battery in the background.
class ResourceMonitorScreen extends StatefulWidget {
  const ResourceMonitorScreen({super.key});

  @override
  State<ResourceMonitorScreen> createState() => _ResourceMonitorScreenState();
}

class _ResourceMonitorScreenState extends State<ResourceMonitorScreen> {
  late final ResourceMonitorService _monitor;

  @override
  void initState() {
    super.initState();
    _monitor = Get.find<ResourceMonitorService>();
    _monitor.start();
  }

  @override
  void dispose() {
    _monitor.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final storage = Get.find<ChatStorageService>();

    return Scaffold(
      backgroundColor: context.bg,
      appBar: AppBar(
        backgroundColor: context.bgPanel,
        title: const Text('Resource Monitor', style: TextStyle(fontSize: 16)),
        elevation: 0,
        centerTitle: true,
      ),
      body: Obx(() {
        if (!_monitor.isSupported) {
          return _unsupportedNotice(context);
        }

        final samples = _monitor.samples;
        final cpu = samples.map((s) => s.systemCpuPercent).toList();
        final appMem = samples.map((s) => s.processMemoryMb).toList();
        final sysMemPercent = samples
            .map((s) => (s.systemMemoryUsedMb != null &&
                    s.systemMemoryTotalMb != null &&
                    s.systemMemoryTotalMb! > 0)
                ? (s.systemMemoryUsedMb! / s.systemMemoryTotalMb! * 100)
                : null)
            .toList();
        final tps = samples.map<double?>((s) => s.tokensPerSecond).toList();
        final gpuClock = samples
            .map<double?>((s) => s.adrenoGpuClockMhz?.toDouble())
            .toList();
        final hasGpuReading = gpuClock.any((v) => v != null);

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            LiveLineChart(
              title: 'CPU (all cores)',
              values: cpu,
              color: AppColors.accent,
              fixedMax: 100,
              formatValue: (v) => '${v.toStringAsFixed(0)}%',
            ),
            const SizedBox(height: 12),
            LiveLineChart(
              title: 'App Memory',
              values: appMem,
              color: AppColors.standard,
              formatValue: (v) => '${v.toStringAsFixed(0)} MB',
            ),
            const SizedBox(height: 12),
            LiveLineChart(
              title: 'System Memory',
              values: sysMemPercent,
              color: AppColors.orange,
              fixedMax: 100,
              formatValue: (v) => '${v.toStringAsFixed(0)}%',
            ),
            const SizedBox(height: 12),
            LiveLineChart(
              title: 'Generation Speed',
              values: tps,
              color: AppColors.green,
              formatValue: (v) => '${v.toStringAsFixed(1)} t/s',
            ),
            const SizedBox(height: 12),
            _gpuSection(context, storage.backendType, hasGpuReading, gpuClock),
            const SizedBox(height: 12),
            _npuSection(context),
            const SizedBox(height: 24),
            _honestyNotice(context),
          ],
        );
      }),
    );
  }

  Widget _gpuSection(
    BuildContext context,
    String backendType,
    bool hasReading,
    List<double?> gpuClock,
  ) {
    if (hasReading) {
      return LiveLineChart(
        title: 'GPU Clock (Adreno, best-effort)',
        values: gpuClock,
        color: AppColors.accentHi,
        formatValue: (v) => '${v.toStringAsFixed(0)} MHz',
      );
    }
    return _infoCard(
      context,
      icon: Icons.memory_rounded,
      title: 'GPU',
      body:
          'Active backend: ${backendType.toUpperCase()}. Android exposes no '
          'public, permission-free API for GPU utilization — this reading '
          'only works on some Qualcomm Adreno devices, and this one didn\'t '
          'expose it. Not fabricated as a number.',
    );
  }

  Widget _npuSection(BuildContext context) {
    return _infoCard(
      context,
      icon: Icons.hub_outlined,
      title: 'NPU',
      body: 'Not available. NPU utilization is exposed only through '
          'vendor-specific SDKs (Qualcomm QNN, Samsung Exynos NPU, etc.) that '
          'this app does not integrate — there is no generic Android API for '
          'it, so this is never faked with a placeholder number.',
    );
  }

  Widget _infoCard(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String body,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: context.bgPanel,
        border: Border.all(color: context.border),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 18, color: context.textM),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: context.text)),
                const SizedBox(height: 4),
                Text(body,
                    style: TextStyle(
                        fontSize: 12, color: context.textM, height: 1.4)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _honestyNotice(BuildContext context) {
    return Text(
      'Readings come from /proc on Android/Linux and update once per second '
      'while this screen is open.',
      style: TextStyle(fontSize: 11, color: context.textD),
      textAlign: TextAlign.center,
    );
  }

  Widget _unsupportedNotice(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.speed_rounded, size: 48, color: context.textD),
            const SizedBox(height: 16),
            Text(
              'Not available on this platform',
              style: TextStyle(
                  fontSize: 15, fontWeight: FontWeight.w600, color: context.text),
            ),
            const SizedBox(height: 8),
            Text(
              'Live resource monitoring reads Android/Linux\'s /proc '
              'filesystem, which doesn\'t exist on this platform.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 13, color: context.textM),
            ),
          ],
        ),
      ),
    );
  }
}
