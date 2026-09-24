import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:get/get.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../models/memory_entry.dart';

/// Persistent, cross-conversation memory — a real directory on disk
/// (`PortableAI/memory/`) holding distilled notes from past chats, each with
/// an embedding vector for semantic retrieval. This is deliberately global
/// (not per-chat): the point is that something learned in one conversation
/// can surface again in a completely different one.
class MemoryService extends GetxService {
  final entries = <MemoryEntry>[].obs;

  late String _memoryDir;
  late File _indexFile;

  Future<MemoryService> init() async {
    final appDir = await getApplicationDocumentsDirectory();
    _memoryDir = p.join(appDir.path, 'PortableAI', 'memory');
    await Directory(_memoryDir).create(recursive: true);
    _indexFile = File(p.join(_memoryDir, 'index.json'));
    await _load();
    return this;
  }

  /// The actual on-disk memory directory, surfaced so Settings can show the
  /// user exactly where this data lives.
  String get memoryDirPath => _memoryDir;

  Future<void> _load() async {
    try {
      if (!await _indexFile.exists()) {
        entries.value = [];
        return;
      }
      final raw = await _indexFile.readAsString();
      final list = jsonDecode(raw) as List;
      entries.value = list
          .map((j) => MemoryEntry.fromJson(Map<String, dynamic>.from(j as Map)))
          .toList();
    } catch (_) {
      // Corrupt or unreadable index — start fresh rather than crash. The
      // old file is left on disk in case it's worth investigating by hand.
      entries.value = [];
    }
  }

  Future<void> _persist() async {
    try {
      final list = entries.map((e) => e.toJson()).toList();
      await _indexFile.writeAsString(jsonEncode(list));
    } catch (_) {
      // Best-effort — memory persistence failing should never break chat.
    }
  }

  Future<void> add(
    String text,
    List<double> embedding, {
    String? sourceChatId,
  }) async {
    if (text.trim().isEmpty || embedding.isEmpty) return;
    entries.add(MemoryEntry(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      text: text.trim(),
      embedding: embedding,
      createdAt: DateTime.now(),
      sourceChatId: sourceChatId,
    ));
    await _persist();
  }

  /// Same as [add], but skipped entirely if a near-duplicate (cosine
  /// similarity above [dupThreshold]) is already stored — cheap insurance
  /// against the memory file growing by one entry every time the user
  /// repeats "my name is Alex" across ten different chats.
  Future<bool> addIfNotDuplicate(
    String text,
    List<double> embedding, {
    String? sourceChatId,
    double dupThreshold = 0.92,
  }) async {
    if (text.trim().isEmpty || embedding.isEmpty) return false;
    for (final entry in entries) {
      if (_cosineSimilarity(embedding, entry.embedding) >= dupThreshold) {
        return false;
      }
    }
    await add(text, embedding, sourceChatId: sourceChatId);
    return true;
  }

  Future<void> delete(String id) async {
    entries.removeWhere((e) => e.id == id);
    await _persist();
  }

  Future<void> clear() async {
    entries.clear();
    await _persist();
  }

  /// Top-[k] most relevant memories to [queryEmbedding] by cosine
  /// similarity, dropping anything below [minScore] so unrelated memories
  /// never get force-injected just to fill k slots.
  List<MemoryEntry> topK(
    List<double> queryEmbedding, {
    int k = 3,
    double minScore = 0.2,
  }) {
    if (entries.isEmpty || queryEmbedding.isEmpty) return const [];

    final scored = <MapEntry<MemoryEntry, double>>[];
    for (final entry in entries) {
      final score = _cosineSimilarity(queryEmbedding, entry.embedding);
      if (score >= minScore) scored.add(MapEntry(entry, score));
    }
    scored.sort((a, b) => b.value.compareTo(a.value));
    return scored.take(k).map((e) => e.key).toList();
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    if (a.isEmpty || b.isEmpty || a.length != b.length) return 0.0;
    var dot = 0.0, normA = 0.0, normB = 0.0;
    for (var i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    if (normA == 0 || normB == 0) return 0.0;
    return dot / (sqrt(normA) * sqrt(normB));
  }
}
