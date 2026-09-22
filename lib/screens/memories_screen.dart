import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../theme/app_colors.dart';
import '../models/memory_entry.dart';
import '../services/memory_service.dart';

/// Lets the user see, and delete, everything Smart Recall has remembered
/// about them across every chat. Given how personal this data can be
/// (relationships, health, plans), this screen exists so the memory store
/// is never a black box — full transparency and control, matching the
/// app's own "total privacy, on-device" positioning.
class MemoriesScreen extends StatefulWidget {
  const MemoriesScreen({super.key});

  @override
  State<MemoriesScreen> createState() => _MemoriesScreenState();
}

class _MemoriesScreenState extends State<MemoriesScreen> {
  late List<MemoryEntry> _memories;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  void _refresh() {
    setState(() {
      _memories = Get.find<MemoryService>().getAll();
    });
  }

  Future<void> _delete(MemoryEntry entry) async {
    await Get.find<MemoryService>().delete(entry.id);
    _refresh();
  }

  Future<void> _confirmClearAll() async {
    if (_memories.isEmpty) return;
    final confirmed = await Get.dialog<bool>(
      AlertDialog(
        backgroundColor: context.bgPanel,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Text('Clear All Memories?', style: TextStyle(color: context.text)),
        content: Text(
          'This permanently deletes everything Smart Recall has remembered '
          'about you. This cannot be undone.',
          style: TextStyle(color: context.textM),
        ),
        actions: [
          TextButton(
            onPressed: () => Get.back(result: false),
            child: Text('Cancel', style: TextStyle(color: context.textD)),
          ),
          ElevatedButton(
            onPressed: () => Get.back(result: true),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.red,
              elevation: 0,
            ),
            child: const Text('Clear All', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
    if (confirmed != true) return;
    await Get.find<MemoryService>().clearAll();
    _refresh();
  }

  String _formatDate(DateTime dt) {
    const months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    ];
    return '${months[dt.month - 1]} ${dt.day}, ${dt.year}';
  }

  @override
  Widget build(BuildContext context) {
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
                    'Memories',
                    style: TextStyle(
                      fontSize: 17,
                      fontWeight: FontWeight.w600,
                      color: context.text,
                    ),
                  ),
                  const Spacer(),
                  if (_memories.isNotEmpty)
                    IconButton(
                      icon: const Icon(Icons.delete_sweep_outlined, color: AppColors.red),
                      onPressed: _confirmClearAll,
                      tooltip: 'Clear all',
                    ),
                  const SizedBox(width: 4),
                ],
              ),
            ),
          ),
          Expanded(
            child: _memories.isEmpty
                ? Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.psychology_outlined, size: 48, color: context.textD),
                        const SizedBox(height: 16),
                        Text(
                          'Nothing remembered yet.',
                          style: TextStyle(color: context.textD, fontSize: 14),
                        ),
                        const SizedBox(height: 6),
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 40),
                          child: Text(
                            'When Smart Recall is on, durable facts from your '
                            'chats show up here.',
                            textAlign: TextAlign.center,
                            style: TextStyle(color: context.textD, fontSize: 12),
                          ),
                        ),
                      ],
                    ),
                  )
                : ListView.separated(
                    padding: const EdgeInsets.all(16),
                    itemCount: _memories.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 8),
                    itemBuilder: (_, i) {
                      final m = _memories[i];
                      final date = m.eventDate ?? m.createdAt;
                      return Dismissible(
                        key: ValueKey(m.id),
                        direction: DismissDirection.endToStart,
                        onDismissed: (_) => _delete(m),
                        background: Container(
                          alignment: Alignment.centerRight,
                          padding: const EdgeInsets.only(right: 20),
                          decoration: BoxDecoration(
                            color: AppColors.red.withValues(alpha: 0.15),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: const Icon(Icons.delete_outline_rounded, color: AppColors.red),
                        ),
                        child: Container(
                          padding: const EdgeInsets.all(14),
                          decoration: BoxDecoration(
                            color: context.bgPanel,
                            border: Border.all(color: context.border),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                m.text,
                                style: TextStyle(color: context.text, fontSize: 14, height: 1.4),
                              ),
                              const SizedBox(height: 8),
                              Row(
                                children: [
                                  Icon(Icons.event_outlined, size: 13, color: context.textD),
                                  const SizedBox(width: 4),
                                  Text(
                                    _formatDate(date),
                                    style: TextStyle(color: context.textD, fontSize: 11),
                                  ),
                                  if (!m.hasEmbedding) ...[
                                    const SizedBox(width: 10),
                                    Icon(Icons.hourglass_empty_rounded, size: 12, color: context.textD),
                                    const SizedBox(width: 3),
                                    Text(
                                      'not yet searchable',
                                      style: TextStyle(color: context.textD, fontSize: 11),
                                    ),
                                  ],
                                ],
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}
