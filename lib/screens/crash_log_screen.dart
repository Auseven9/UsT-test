import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';

import '../theme/app_colors.dart';
import '../models/crash_log_entry.dart';
import '../services/crash_log_service.dart';

/// Persistent crash log — survives app restarts (unlike the regular in-memory
/// [LogsScreen]), because its whole purpose is to still be there after the
/// crash that created an entry. Fully read/write: view every entry, delete
/// individual ones, clear everything, or export the full log as text.
class CrashLogScreen extends StatelessWidget {
  const CrashLogScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final crashLog = Get.find<CrashLogService>();

    return Scaffold(
      backgroundColor: context.bg,
      body: Column(
        children: [
          Container(
            padding: EdgeInsets.only(
              top: MediaQuery.of(context).padding.top,
              left: 4,
              right: 4,
            ),
            decoration: BoxDecoration(
              color: context.bg,
              border: Border(bottom: BorderSide(color: context.border, width: 0.5)),
            ),
            child: SizedBox(
              height: 52,
              child: Row(
                children: [
                  IconButton(
                    icon: Icon(Icons.arrow_back_rounded, color: context.text),
                    onPressed: () => Get.back(),
                  ),
                  Text(
                    'Crash Log',
                    style: TextStyle(
                      fontSize: 17,
                      fontWeight: FontWeight.w600,
                      color: context.text,
                    ),
                  ),
                  const Spacer(),
                  IconButton(
                    icon: Icon(Icons.copy_rounded, size: 20, color: context.textM),
                    tooltip: 'Copy full crash log',
                    onPressed: () {
                      Clipboard.setData(ClipboardData(text: crashLog.exportAll()));
                      Get.snackbar(
                        'Copied',
                        'Full crash log copied to clipboard.',
                        snackPosition: SnackPosition.BOTTOM,
                        duration: const Duration(seconds: 2),
                      );
                    },
                  ),
                  Obx(
                    () => IconButton(
                      icon: Icon(Icons.delete_sweep_outlined,
                          size: 20, color: context.textD),
                      tooltip: 'Clear crash log',
                      onPressed: crashLog.entries.isEmpty
                          ? null
                          : () => _confirmClearAll(context, crashLog),
                    ),
                  ),
                  const SizedBox(width: 4),
                ],
              ),
            ),
          ),
          Expanded(
            child: Obx(() {
              final entries = crashLog.entries.toList().reversed.toList();
              if (entries.isEmpty) {
                return Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.check_circle_outline_rounded,
                          size: 48, color: AppColors.green),
                      const SizedBox(height: 16),
                      Text(
                        'No crashes recorded.',
                        style: TextStyle(color: context.textD, fontSize: 14),
                      ),
                    ],
                  ),
                );
              }

              return ListView.builder(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                itemCount: entries.length,
                itemBuilder: (context, index) {
                  // entries is reversed for display; map back to the real
                  // index in crashLog.entries for delete.
                  final realIndex = crashLog.entries.length - 1 - index;
                  return _CrashLogTile(
                    entry: entries[index],
                    onDelete: () => crashLog.deleteAt(realIndex),
                  );
                },
              );
            }),
          ),
        ],
      ),
    );
  }

  void _confirmClearAll(BuildContext context, CrashLogService crashLog) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: context.bgPanel,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Text('Clear Crash Log?', style: TextStyle(color: context.text)),
        content: Text(
          'This deletes every recorded crash entry. This cannot be undone.',
          style: TextStyle(color: context.textM),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text('Cancel', style: TextStyle(color: context.textD)),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(ctx);
              crashLog.clear();
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.red,
              elevation: 0,
            ),
            child: const Text('Clear', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }
}

class _CrashLogTile extends StatelessWidget {
  final CrashLogEntry entry;
  final VoidCallback onDelete;

  const _CrashLogTile({required this.entry, required this.onDelete});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.red.withOpacity(0.05),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: AppColors.red.withOpacity(0.15)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.error_outline_rounded, size: 16, color: AppColors.red),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      entry.timestamp.toIso8601String().replaceFirst('T', ' '),
                      style: TextStyle(
                        fontSize: 10,
                        color: context.textD,
                        fontFamily: 'monospace',
                      ),
                    ),
                    const SizedBox(width: 6),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
                      decoration: BoxDecoration(
                        color: AppColors.red.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(3),
                      ),
                      child: Text(
                        entry.type,
                        style: const TextStyle(
                          fontSize: 9,
                          fontWeight: FontWeight.w600,
                          color: AppColors.red,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  entry.message,
                  style: TextStyle(fontSize: 12, color: context.text, height: 1.3),
                ),
                if (entry.context != null) ...[
                  const SizedBox(height: 2),
                  Text(
                    entry.context!,
                    style: TextStyle(fontSize: 11, color: context.textM),
                  ),
                ],
              ],
            ),
          ),
          IconButton(
            icon: Icon(Icons.close_rounded, size: 14, color: context.textD),
            onPressed: onDelete,
            tooltip: 'Delete this entry',
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
          ),
        ],
      ),
    );
  }
}
