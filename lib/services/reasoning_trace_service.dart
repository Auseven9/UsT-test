import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:get/get.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../models/reasoning_trace_entry.dart';

/// A second, separate memory lane from `MemoryService` — this one holds
/// distilled gists of the model's own past reasoning (its chain-of-thought,
/// summarized down), not facts about the user or the world. Kept apart
/// deliberately: a reasoning trace is procedural ("how I got to an answer"),
/// not declarative ("this is true"), and mixing the two would have fact
/// retrieval competing against self-reflection for the same ranking.
///
/// Bounded automatically rather than growing forever: `runLaneMaintenance`
/// compresses older entries down to a terse clause once they age past the
/// most recent [_fullDetailCount], and only drops the very oldest outright
/// once the lane exceeds [_maxEntries] even after compression — the lane's
/// footprint stays roughly constant as more turns happen, without ever
/// silently losing every trace of an old turn the moment it's no longer
/// recent.
class ReasoningTraceService extends GetxService {
  final entries = <ReasoningTraceEntry>[].obs;

  static const _fullDetailCount = 20;
  static const _fullSummaryMaxChars = 220;
  static const _compressedMaxChars = 60;
  static const _maxEntries = 150;

  late String _dir;
  late File _indexFile;

  /// Same rationale as `MemoryService._persistChain` — serializes disk
  /// writes so an overlapping add + maintenance pass can't land out of order.
  Future<void> _persistChain = Future.value();

  Future<ReasoningTraceService> init() async {
    final appDir = await getApplicationDocumentsDirectory();
    _dir = p.join(appDir.path, 'PortableAI', 'reasoning');
    await Directory(_dir).create(recursive: true);
    _indexFile = File(p.join(_dir, 'index.json'));
    await _load();
    return this;
  }

  String get traceDirPath => _dir;

  Future<void> _load() async {
    try {
      if (!await _indexFile.exists()) {
        entries.value = [];
        return;
      }
      final raw = await _indexFile.readAsString();
      final list = jsonDecode(raw) as List;
      final parsed = <ReasoningTraceEntry>[];
      for (final j in list) {
        try {
          parsed.add(ReasoningTraceEntry.fromJson(Map<String, dynamic>.from(j as Map)));
        } catch (_) {
          // Skip just this entry — one corrupted trace shouldn't lose the rest.
        }
      }
      entries.value = parsed;
    } catch (_) {
      entries.value = [];
    }
  }

  Future<void> _persist() async {
    final snapshot = entries.toList();
    _persistChain = _persistChain.then((_) async {
      try {
        await _indexFile.writeAsString(jsonEncode(snapshot.map((e) => e.toJson()).toList()));
      } catch (_) {
        // Best-effort — a failed write here must never surface as a chat error.
      }
    });
    await _persistChain;
  }

  /// Appends a new gist for [chatId]. [summary] is capped to
  /// [_fullSummaryMaxChars] here — the caller is expected to have already
  /// asked for something short, this is just a hard backstop.
  Future<void> add(String chatId, String summary) async {
    final trimmed = summary.trim();
    if (trimmed.isEmpty) return;
    entries.add(ReasoningTraceEntry(
      id: '${DateTime.now().microsecondsSinceEpoch}',
      createdAt: DateTime.now(),
      chatId: chatId,
      summary: trimmed.length > _fullSummaryMaxChars
          ? '${trimmed.substring(0, _fullSummaryMaxChars)}…'
          : trimmed,
    ));
    await _persist();
  }

  Future<({int compressed, int dropped})> runLaneMaintenance() async {
    if (entries.length <= _fullDetailCount) return (compressed: 0, dropped: 0);

    final byAge = entries.toList()..sort((a, b) => a.createdAt.compareTo(b.createdAt));
    final agingOutCount = byAge.length - _fullDetailCount;

    var compressedCount = 0;
    for (var i = 0; i < agingOutCount; i++) {
      final old = byAge[i];
      if (old.compressed) continue;
      final idx = entries.indexWhere((e) => e.id == old.id);
      if (idx == -1) continue;
      final s = entries[idx].summary;
      entries[idx] = ReasoningTraceEntry(
        id: old.id,
        createdAt: old.createdAt,
        chatId: old.chatId,
        summary: s.length > _compressedMaxChars ? '${s.substring(0, _compressedMaxChars)}…' : s,
        compressed: true,
      );
      compressedCount++;
    }

    var droppedCount = 0;
    while (entries.length > _maxEntries) {
      entries.sort((a, b) => a.createdAt.compareTo(b.createdAt));
      entries.removeAt(0);
      droppedCount++;
    }

    if (compressedCount > 0 || droppedCount > 0) await _persist();
    return (compressed: compressedCount, dropped: droppedCount);
  }

  /// The [count] most recent gists, newest first — plain recency, no
  /// semantic search. Used to let the model (via the `recall_reasoning`
  /// tool) or the self-critique pass look back at its own past reasoning.
  List<String> recent({int count = 8}) {
    final byRecency = entries.toList()..sort((a, b) => b.createdAt.compareTo(a.createdAt));
    return byRecency.take(count).map((e) => e.summary).toList();
  }
}
