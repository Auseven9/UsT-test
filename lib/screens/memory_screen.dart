import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../theme/app_colors.dart';
import '../models/memory_entry.dart';
import '../services/memory_service.dart';

/// Browser for the persistent memory directory — every distilled note the
/// app has stored across conversations, with the ability to delete
/// individual entries or clear everything.
class MemoryScreen extends StatelessWidget {
  const MemoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final memory = Get.find<MemoryService>();

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
                    'Persistent Memory',
                    style: TextStyle(
                      fontSize: 17,
                      fontWeight: FontWeight.w600,
                      color: context.text,
                    ),
                  ),
                  const Spacer(),
                  Obx(
                    () => IconButton(
                      icon: Icon(Icons.delete_sweep_outlined,
                          size: 20, color: context.textD),
                      tooltip: 'Clear all memories',
                      onPressed: memory.entries.isEmpty
                          ? null
                          : () => _confirmClearAll(context, memory),
                    ),
                  ),
                  const SizedBox(width: 4),
                ],
              ),
            ),
          ),
          Expanded(
            child: Obx(() {
              final entries = memory.entries.toList().reversed.toList();
              if (entries.isEmpty) {
                return Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.psychology_alt_outlined,
                          size: 48, color: context.textD),
                      const SizedBox(height: 16),
                      Text(
                        'Nothing remembered yet.',
                        style: TextStyle(color: context.textD, fontSize: 14),
                      ),
                      const SizedBox(height: 8),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 40),
                        child: Text(
                          'Notes appear here automatically as older parts of '
                          'long conversations get distilled for later recall.',
                          textAlign: TextAlign.center,
                          style: TextStyle(color: context.textD, fontSize: 12),
                        ),
                      ),
                    ],
                  ),
                );
              }

              return ListView.builder(
                padding: const EdgeInsets.all(16),
                itemCount: entries.length,
                itemBuilder: (context, index) {
                  final entry = entries[index];
                  return _MemoryTile(
                    entry: entry,
                    onDelete: () => memory.delete(entry.id),
                  );
                },
              );
            }),
          ),
        ],
      ),
    );
  }

  void _confirmClearAll(BuildContext context, MemoryService memory) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: context.bgPanel,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Text('Clear All Memories?', style: TextStyle(color: context.text)),
        content: Text(
          'This deletes every stored memory note. This cannot be undone.',
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
              memory.clear();
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.red,
              elevation: 0,
            ),
            child: const Text('Clear All', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }
}

class _MemoryTile extends StatelessWidget {
  final MemoryEntry entry;
  final VoidCallback onDelete;

  const _MemoryTile({required this.entry, required this.onDelete});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: context.bgPanel,
        border: Border.all(color: context.border),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  entry.text,
                  style: TextStyle(fontSize: 13, color: context.text, height: 1.4),
                ),
                const SizedBox(height: 6),
                Text(
                  _formatDate(entry.createdAt),
                  style: TextStyle(fontSize: 11, color: context.textD),
                ),
              ],
            ),
          ),
          IconButton(
            icon: Icon(Icons.close_rounded, size: 16, color: context.textD),
            onPressed: onDelete,
            tooltip: 'Delete this memory',
          ),
        ],
      ),
    );
  }

  String _formatDate(DateTime dt) {
    return '${dt.year}-${dt.month.toString().padLeft(2, '0')}-${dt.day.toString().padLeft(2, '0')} '
        '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
  }
}
