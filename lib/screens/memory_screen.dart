import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../theme/app_colors.dart';
import '../models/memory_entry.dart';
import '../services/memory_service.dart';
import '../services/embedding_service.dart';
import '../routes/app_routes.dart';

/// Browser for the persistent memory directory — every distilled note the
/// app has stored across conversations. Supports deleting, manually adding,
/// and manually editing entries in place — the automatic pipelines (per-
/// turn extraction, the model's own remember/update_memory/supersede_memory
/// tools, the consolidation sweep) all write through the same MemoryService
/// methods this screen's manual controls call directly.
class MemoryScreen extends StatelessWidget {
  const MemoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final memory = Get.find<MemoryService>();

    return Scaffold(
      backgroundColor: context.bg,
      floatingActionButton: FloatingActionButton(
        onPressed: () => _openEditDialog(context, memory, existing: null),
        tooltip: 'Add a memory manually',
        backgroundColor: AppColors.accent,
        child: const Icon(Icons.add_rounded, color: Colors.white),
      ),
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
                      icon: Icon(Icons.hub_outlined, size: 20, color: context.textD),
                      tooltip: 'View as graph',
                      onPressed: memory.entries.isEmpty
                          ? null
                          : () => Get.toNamed(AppRoutes.memoryGraph),
                    ),
                  ),
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
                          'long conversations get distilled for later recall '
                          '— or tap + to add one yourself.',
                          textAlign: TextAlign.center,
                          style: TextStyle(color: context.textD, fontSize: 12),
                        ),
                      ),
                    ],
                  ),
                );
              }

              return ListView.builder(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 88),
                itemCount: entries.length,
                itemBuilder: (context, index) {
                  final entry = entries[index];
                  return _MemoryTile(
                    entry: entry,
                    onDelete: () => memory.delete(entry.id),
                    onEdit: () => _openEditDialog(context, memory, existing: entry),
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

  /// Shared dialog for both manual add ([existing] null) and manual edit
  /// ([existing] set) — re-embeds through the live embedding model when
  /// text actually changes (add always embeds; edit only if the text
  /// field differs from what's stored), since a stored memory's vector
  /// must always match its stored text or retrieval silently drifts from
  /// what the text actually says.
  void _openEditDialog(
    BuildContext context,
    MemoryService memory, {
    required MemoryEntry? existing,
  }) {
    final textController = TextEditingController(text: existing?.text ?? '');
    var category = existing?.category ?? 'general';
    var valence = existing?.valence ?? 'neutral';
    var memoryType = existing?.memoryType ?? 'episodic';
    var subject = existing?.subject ?? 'user';
    var entityType = existing?.entityType ?? 'none';
    final entityNameController = TextEditingController(text: existing?.entityName ?? '');
    final locationController = TextEditingController(text: existing?.location ?? '');
    final participantsController =
        TextEditingController(text: (existing?.participants ?? const []).join(', '));
    final connectionController = TextEditingController(text: existing?.connection ?? '');
    // Includes 'summary' — not a category anything picks from this dialog,
    // but memory consolidation (ChatController._runMemoryConsolidation)
    // writes entries with that category, and editing one of those must not
    // crash the dropdown for having a value with no matching item.
    const categories = ['fact', 'preference', 'event', 'instruction', 'general', 'summary'];
    const valences = ['positive', 'negative', 'neutral'];
    const memoryTypes = ['episodic', 'semantic'];
    const subjects = ['user', 'assistant'];
    const entityTypes = [
      'none', 'person', 'place', 'project', 'organization', 'event', 'idea',
    ];
    var busy = false;

    showDialog(
      context: context,
      builder: (dialogContext) => StatefulBuilder(
        builder: (dialogContext, setState) => AlertDialog(
          backgroundColor: dialogContext.bgPanel,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          title: Text(
            existing == null ? 'Add Memory' : 'Edit Memory',
            style: TextStyle(color: dialogContext.text),
          ),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextField(
                  controller: textController,
                  maxLines: 4,
                  autofocus: existing == null,
                  style: TextStyle(color: dialogContext.text, fontSize: 13),
                  decoration: InputDecoration(
                    hintText: 'What should be remembered?',
                    hintStyle: TextStyle(color: dialogContext.textD),
                    filled: true,
                    fillColor: dialogContext.bgInput,
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                ),
                const SizedBox(height: 12),
                _dropdownRow(dialogContext, 'Category', category, categories,
                    (v) => setState(() => category = v)),
                _dropdownRow(dialogContext, 'Valence', valence, valences,
                    (v) => setState(() => valence = v)),
                _dropdownRow(dialogContext, 'Type', memoryType, memoryTypes,
                    (v) => setState(() => memoryType = v)),
                _dropdownRow(dialogContext, 'Subject', subject, subjects,
                    (v) => setState(() => subject = v)),
                const SizedBox(height: 12),
                Text(
                  'Entity — who/what this is about, for building the memory '
                  'network (all optional)',
                  style: TextStyle(fontSize: 11, color: dialogContext.textD),
                ),
                const SizedBox(height: 8),
                _textFieldRow(dialogContext, 'Name', entityNameController,
                    hint: 'e.g. Alex, Project Chimera'),
                _dropdownRow(dialogContext, 'Kind', entityType, entityTypes,
                    (v) => setState(() => entityType = v)),
                _textFieldRow(dialogContext, 'Location', locationController,
                    hint: 'e.g. Seattle'),
                _textFieldRow(dialogContext, 'Participants', participantsController,
                    hint: 'comma-separated names'),
                _textFieldRow(dialogContext, 'Connection', connectionController,
                    hint: 'e.g. coworker of Alex'),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: busy ? null : () => Navigator.pop(dialogContext),
              child: Text('Cancel', style: TextStyle(color: dialogContext.textD)),
            ),
            ElevatedButton(
              onPressed: busy
                  ? null
                  : () async {
                      final text = textController.text.trim();
                      if (text.isEmpty) return;
                      setState(() => busy = true);
                      final embeddingService = Get.find<EmbeddingService>();
                      // Only a new entry or a real text change needs a
                      // fresh vector — a pure category/valence/type/subject
                      // edit on unchanged text doesn't, and shouldn't be
                      // blocked just because no embedding model happens to
                      // be loaded right now.
                      final needsEmbedding = existing == null || text != existing.text;
                      if (needsEmbedding && !embeddingService.isLoaded.value) {
                        Navigator.pop(dialogContext);
                        Get.snackbar(
                          'No Embedding Model',
                          'Set an embedding model in Settings > Persistent '
                              'Memory before adding or changing a memory\'s text.',
                          snackPosition: SnackPosition.BOTTOM,
                        );
                        return;
                      }
                      if (existing == null) {
                        final vector = await embeddingService.embed(text);
                        if (!dialogContext.mounted) return;
                        if (vector == null) {
                          Navigator.pop(dialogContext);
                          Get.snackbar(
                            'Embedding Failed',
                            'Could not add this memory — the embedding '
                                'model didn\'t respond. Nothing was saved.',
                            snackPosition: SnackPosition.BOTTOM,
                          );
                          return;
                        }
                        await memory.add(
                          text,
                          vector,
                          category: category,
                          valence: valence,
                          memoryType: memoryType,
                          subject: subject,
                          tags: const ['manual'],
                          entityName: entityNameController.text.trim(),
                          entityType: entityType,
                          location: locationController.text.trim(),
                          participants: _parseParticipants(participantsController.text),
                          connection: connectionController.text.trim(),
                        );
                      } else {
                        final textChanged = text != existing.text;
                        final vector =
                            textChanged ? await embeddingService.embed(text) : null;
                        if (!dialogContext.mounted) return;
                        if (textChanged && vector == null) {
                          Navigator.pop(dialogContext);
                          Get.snackbar(
                            'Embedding Failed',
                            'Could not update this memory\'s text — the '
                                'embedding model didn\'t respond. Nothing was '
                                'changed.',
                            snackPosition: SnackPosition.BOTTOM,
                          );
                          return;
                        }
                        await memory.updateEntry(
                          existing.id,
                          text: textChanged ? text : null,
                          newEmbedding: vector,
                          category: category,
                          valence: valence,
                          memoryType: memoryType,
                          subject: subject,
                          entityName: entityNameController.text.trim(),
                          entityType: entityType,
                          location: locationController.text.trim(),
                          participants: _parseParticipants(participantsController.text),
                          connection: connectionController.text.trim(),
                        );
                      }
                      if (!dialogContext.mounted) return;
                      Navigator.pop(dialogContext);
                    },
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.accent,
                elevation: 0,
              ),
              child: busy
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                  : Text(existing == null ? 'Add' : 'Save'),
            ),
          ],
        ),
      ),
    ).then((_) {
      textController.dispose();
      entityNameController.dispose();
      locationController.dispose();
      participantsController.dispose();
      connectionController.dispose();
    });
  }

  List<String> _parseParticipants(String raw) => raw
      .split(',')
      .map((s) => s.trim())
      .where((s) => s.isNotEmpty)
      .toList();

  Widget _textFieldRow(
    BuildContext context,
    String label,
    TextEditingController controller, {
    required String hint,
  }) {
    return Padding(
      padding: const EdgeInsets.only(top: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SizedBox(
            width: 70,
            child: Text(label, style: TextStyle(fontSize: 12, color: context.textD)),
          ),
          Expanded(
            child: TextField(
              controller: controller,
              style: TextStyle(fontSize: 13, color: context.text),
              decoration: InputDecoration(
                hintText: hint,
                hintStyle: TextStyle(color: context.textD, fontSize: 12),
                isDense: true,
                filled: true,
                fillColor: context.bgInput,
                contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _dropdownRow(
    BuildContext context,
    String label,
    String value,
    List<String> options,
    ValueChanged<String> onChanged,
  ) {
    return Padding(
      padding: const EdgeInsets.only(top: 8),
      child: Row(
        children: [
          SizedBox(
            width: 70,
            child: Text(label, style: TextStyle(fontSize: 12, color: context.textD)),
          ),
          Expanded(
            child: DropdownButton<String>(
              value: value,
              isExpanded: true,
              dropdownColor: context.bgPanel,
              style: TextStyle(fontSize: 13, color: context.text),
              underline: Container(height: 1, color: context.border),
              items: options
                  .map((o) => DropdownMenuItem(value: o, child: Text(o)))
                  .toList(),
              onChanged: (v) {
                if (v != null) onChanged(v);
              },
            ),
          ),
        ],
      ),
    );
  }
}

class _MemoryTile extends StatelessWidget {
  final MemoryEntry entry;
  final VoidCallback onDelete;
  final VoidCallback onEdit;

  const _MemoryTile({required this.entry, required this.onDelete, required this.onEdit});

  @override
  Widget build(BuildContext context) {
    final superseded = !entry.isActive;
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: context.bgPanel,
        border: Border.all(color: superseded ? context.borderFaint : context.border),
        borderRadius: BorderRadius.circular(12),
      ),
      child: InkWell(
        onTap: onEdit,
        borderRadius: BorderRadius.circular(8),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Wrap(
                    spacing: 6,
                    runSpacing: 4,
                    children: [
                      _chip(context, entry.category, _categoryColor(context, entry.category)),
                      _chip(context, entry.valence, _valenceColor(context, entry.valence)),
                      if (entry.memoryType == 'semantic')
                        _chip(context, 'semantic', AppColors.accentHi),
                      if (entry.subject == 'assistant')
                        _chip(context, 'assistant', AppColors.orange),
                      if (entry.isWorkingMemory)
                        _chip(context, 'working', context.textD),
                      if (entry.confidence < 0.8)
                        _chip(context, 'low confidence', AppColors.orange),
                      for (final tag in entry.tags) _chip(context, tag, context.textD),
                      if (superseded) _chip(context, 'superseded', AppColors.red),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    entry.text,
                    style: TextStyle(
                      fontSize: 13,
                      color: superseded ? context.textD : context.text,
                      height: 1.4,
                      decoration: superseded ? TextDecoration.lineThrough : null,
                    ),
                  ),
                  if (entry.entityName.isNotEmpty ||
                      entry.location.isNotEmpty ||
                      entry.participants.isNotEmpty ||
                      entry.connection.isNotEmpty) ...[
                    const SizedBox(height: 6),
                    Text(
                      [
                        if (entry.entityName.isNotEmpty)
                          '${entry.entityType == 'none' ? '' : '${entry.entityType}: '}${entry.entityName}',
                        if (entry.location.isNotEmpty) 'at ${entry.location}',
                        if (entry.participants.isNotEmpty)
                          'with ${entry.participants.join(', ')}',
                        if (entry.connection.isNotEmpty) entry.connection,
                      ].join(' · '),
                      style: TextStyle(
                        fontSize: 11,
                        fontStyle: FontStyle.italic,
                        color: context.textM,
                      ),
                    ),
                  ],
                  const SizedBox(height: 6),
                  Text(
                    '${_formatDate(entry.createdAt)}'
                    '${entry.accessCount > 0 ? ' · recalled ${entry.accessCount}×' : ''}'
                    '${entry.rehearsalCount > 0 ? ' · rehearsed ${entry.rehearsalCount}×' : ''}'
                    '${entry.linkedIds.isNotEmpty ? ' · ${entry.linkedIds.length} linked' : ''}'
                    '${entry.priorTexts.isNotEmpty ? ' · ${entry.priorTexts.length} prior version${entry.priorTexts.length == 1 ? '' : 's'}' : ''}',
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
      ),
    );
  }

  Widget _chip(BuildContext context, String label, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        label,
        style: TextStyle(fontSize: 10, color: color, fontWeight: FontWeight.w600),
      ),
    );
  }

  Color _categoryColor(BuildContext context, String category) {
    switch (category) {
      case 'fact':
        return AppColors.accent;
      case 'preference':
        return AppColors.green;
      case 'event':
        return AppColors.orange;
      case 'instruction':
        return AppColors.accentHi;
      case 'summary':
        return AppColors.accentDim;
      default:
        return context.textM;
    }
  }

  Color _valenceColor(BuildContext context, String valence) {
    switch (valence) {
      case 'positive':
        return AppColors.green;
      case 'negative':
        return AppColors.red;
      default:
        return context.textM;
    }
  }

  String _formatDate(DateTime dt) {
    return '${dt.year}-${dt.month.toString().padLeft(2, '0')}-${dt.day.toString().padLeft(2, '0')} '
        '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
  }
}
