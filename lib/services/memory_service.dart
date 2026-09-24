import 'dart:async';
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
///
/// Beyond flat similarity search, entries form a small associative graph:
/// each new memory is linked to its most similar existing ones at write
/// time (reusing the embedding already computed — no extra inference cost),
/// and retrieval walks one hop along those links so recalling a memory can
/// surface related ones, not just the single closest match. Repeated or
/// recently-recalled memories rank higher via reinforcement; a memory a
/// later one contradicts gets marked superseded (never deleted) so it stops
/// competing with the current fact in ranking without erasing the record.
class MemoryService extends GetxService {
  final entries = <MemoryEntry>[].obs;

  /// How many nearest existing memories a new one gets linked to at write
  /// time — small on purpose, since this is a lightweight associative hint,
  /// not a dense graph.
  static const _maxLinksPerNode = 4;
  static const _linkThreshold = 0.5;

  /// Half-life, in days, for the recency component of retrieval ranking —
  /// not a deletion timer (nothing is ever auto-deleted), just how quickly a
  /// memory's "was this recent" contribution to ranking fades.
  static const _recencyHalfLifeDays = 21.0;

  late String _memoryDir;
  late File _indexFile;

  /// Serializes all disk writes through a single chain — without this, two
  /// overlapping `_persist()` calls (e.g. a synchronous reinforcement
  /// during `topK()` racing a background extraction task's own write) can
  /// have their file writes complete out of call order, since each
  /// `_persist()` snapshots `entries` synchronously but writes it
  /// asynchronously; whichever write actually lands on disk last wins and
  /// can silently overwrite a newer change with an older snapshot. Chaining
  /// through one Future guarantees writes land in the order they were
  /// requested.
  Future<void> _persistChain = Future.value();

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
      // Parsed one entry at a time — a single malformed entry (a corrupted
      // write, a manual edit, a future format change) throwing must only
      // drop that one entry, not the user's entire memory store via one
      // outer catch-all around the whole list.
      final parsed = <MemoryEntry>[];
      for (final j in list) {
        try {
          parsed.add(MemoryEntry.fromJson(Map<String, dynamic>.from(j as Map)));
        } catch (_) {
          // Skip just this entry.
        }
      }
      entries.value = parsed;
    } catch (_) {
      // File itself unreadable/not valid JSON at all — start fresh rather
      // than crash. The old file is left on disk in case it's worth
      // investigating by hand.
      entries.value = [];
    }
  }

  Future<void> _persist() {
    // Snapshot `entries` synchronously, right now — before this write's
    // turn in the chain — so what gets written reflects the state at the
    // time `_persist()` was called, not whatever `entries` happens to be
    // once this write's turn finally comes up.
    final snapshot = entries.map((e) => e.toJson()).toList();
    final next = _persistChain.then((_) async {
      try {
        await _indexFile.writeAsString(jsonEncode(snapshot));
      } catch (_) {
        // Best-effort — memory persistence failing should never break chat.
      }
    });
    _persistChain = next;
    return next;
  }

  /// Returns the new entry's id, or null if [text]/[embedding] was empty and
  /// nothing was stored.
  Future<String?> add(
    String text,
    List<double> embedding, {
    String? sourceChatId,
    String category = 'general',
    String valence = 'neutral',
    List<String> tags = const [],
  }) async {
    if (text.trim().isEmpty || embedding.isEmpty) return null;

    // Link the new node to its nearest existing active neighbors — reusing
    // the embedding just computed for storage, not a second pass. Bidirectional:
    // the new node remembers its neighbors, and each of those neighbors gets
    // the new node added to its own link list, so walking from an old memory
    // can surface a newer related one too.
    final scored = <MapEntry<MemoryEntry, double>>[];
    for (final entry in entries) {
      if (!entry.isActive) continue;
      final score = _cosineSimilarity(embedding, entry.embedding);
      if (score >= _linkThreshold) scored.add(MapEntry(entry, score));
    }
    scored.sort((a, b) => b.value.compareTo(a.value));
    final neighbors = scored.take(_maxLinksPerNode).map((e) => e.key).toList();

    final newEntry = MemoryEntry(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      text: text.trim(),
      embedding: embedding,
      createdAt: DateTime.now(),
      sourceChatId: sourceChatId,
      category: category,
      valence: valence,
      tags: tags,
      linkedIds: neighbors.map((n) => n.id).toList(),
    );

    for (final neighbor in neighbors) {
      final idx = entries.indexWhere((e) => e.id == neighbor.id);
      if (idx == -1) continue;
      // Cap the reverse edge too — without this, a frequently-referenced
      // memory (e.g. "user's name") accumulates an ever-growing link list
      // over the app's lifetime, contradicting the "small associative
      // graph" this is meant to be. Oldest link drops first (FIFO) to make
      // room, keeping the most recent associations.
      final updatedLinks = [...entries[idx].linkedIds, newEntry.id];
      final trimmed = updatedLinks.length > _maxLinksPerNode
          ? updatedLinks.sublist(updatedLinks.length - _maxLinksPerNode)
          : updatedLinks;
      entries[idx] = entries[idx].copyWith(linkedIds: trimmed);
    }

    entries.add(newEntry);
    await _persist();
    return newEntry.id;
  }

  /// Same as [add], but if a near-duplicate (cosine similarity above
  /// [dupThreshold]) already exists, that memory is *reinforced* instead —
  /// its access count bumped and last-accessed time refreshed — rather than
  /// silently discarded or stored as a second copy. Saying the same thing
  /// twice should strengthen one memory, not create two.
  ///
  /// Returns a record: `wasNew` is true only if a genuinely new entry was
  /// stored (false if an existing one was reinforced instead, or nothing
  /// happened), and `id` is that new entry's id, or the reinforced entry's
  /// id, letting a caller (e.g. contradiction handling) reference whichever
  /// entry actually now represents this fact.
  Future<({bool wasNew, String? id})> addIfNotDuplicate(
    String text,
    List<double> embedding, {
    String? sourceChatId,
    String category = 'general',
    String valence = 'neutral',
    List<String> tags = const [],
    double dupThreshold = 0.92,
  }) async {
    if (text.trim().isEmpty || embedding.isEmpty) return (wasNew: false, id: null);
    for (var i = 0; i < entries.length; i++) {
      if (!entries[i].isActive) continue;
      if (_cosineSimilarity(embedding, entries[i].embedding) >= dupThreshold) {
        entries[i] = entries[i].copyWith(
          accessCount: entries[i].accessCount + 1,
          lastAccessedAt: DateTime.now(),
        );
        await _persist();
        return (wasNew: false, id: entries[i].id);
      }
    }
    final newId = await add(
      text,
      embedding,
      sourceChatId: sourceChatId,
      category: category,
      valence: valence,
      tags: tags,
    );
    return (wasNew: true, id: newId);
  }

  /// Marks [oldId] as superseded by [newId] — [oldId] stays in storage and
  /// stays visible in the memory browser, but drops out of normal retrieval
  /// ranking so an outdated fact stops competing with the current one.
  /// Never deletes anything; this is the only "the model got something
  /// wrong or outdated" correction mechanism, deliberately not deletion.
  Future<void> markSuperseded(String oldId, String newId) async {
    final idx = entries.indexWhere((e) => e.id == oldId);
    if (idx == -1 || oldId == newId) return;
    entries[idx] = entries[idx].copyWith(supersededBy: newId);
    await _persist();
  }

  Future<void> delete(String id) async {
    // Clean up dangling references before removing the entry — otherwise
    // other entries keep pointing at an id that no longer exists: a
    // neighbor's linkedIds would silently miscount its associations, and a
    // memory this one superseded would stay hidden from retrieval forever
    // even though nothing contradicts it anymore.
    for (var i = 0; i < entries.length; i++) {
      final e = entries[i];
      final needsUnlink = e.linkedIds.contains(id);
      final needsUnsupersede = e.supersededBy == id;
      if (!needsUnlink && !needsUnsupersede) continue; // most entries, most deletes

      entries[i] = e.copyWith(
        linkedIds: needsUnlink ? e.linkedIds.where((l) => l != id).toList() : null,
        clearSupersededBy: needsUnsupersede,
      );
    }
    entries.removeWhere((e) => e.id == id);
    await _persist();
  }

  Future<void> clear() async {
    entries.clear();
    await _persist();
  }

  /// Top-[k] most relevant *active* memories to [queryEmbedding] — not a
  /// flat similarity search, but one hop of spreading activation: direct
  /// semantic matches are found first, then their linked neighbors are
  /// pulled in too (at a discount, since they matched associatively rather
  /// than directly), and the combined pool is ranked by a blend of semantic
  /// similarity, recency, and reinforcement (how often/recently recalled) —
  /// so a frequently-referenced memory can outrank a merely-similar one
  /// mentioned once and never touched again. Whatever's actually returned
  /// is itself reinforced (recall strengthens a memory), which is why this
  /// is not a `const`/pure function despite looking read-only.
  List<MemoryEntry> topK(
    List<double> queryEmbedding, {
    int k = 3,
    double minScore = 0.2,
  }) {
    if (entries.isEmpty || queryEmbedding.isEmpty) return const [];
    final now = DateTime.now();

    double rank(MemoryEntry e, double similarity) {
      // Recency measures "was this relevant recently" — a memory that's
      // frequently recalled should stay ranked as fresh even if it was
      // first created a while ago, so this uses whichever is more recent
      // of creation or last recall, not just creation.
      final mostRecentTouch =
          e.lastAccessedAt.isAfter(e.createdAt) ? e.lastAccessedAt : e.createdAt;
      final ageDays = now.difference(mostRecentTouch).inHours / 24.0;
      final recency = pow(0.5, ageDays / _recencyHalfLifeDays).toDouble();
      final reinforcement = (log(e.accessCount + 1) / log(10)).clamp(0.0, 1.0);
      return similarity * 0.7 + recency * 0.15 + reinforcement * 0.15;
    }

    // Pass 1: direct semantic matches.
    final direct = <String, double>{}; // id -> similarity
    for (final entry in entries) {
      if (!entry.isActive) continue;
      final score = _cosineSimilarity(queryEmbedding, entry.embedding);
      if (score >= minScore) direct[entry.id] = score;
    }

    // Pass 2: one hop along the graph from each direct match. Score each
    // neighbor by its own real similarity to the query (cheap — just more
    // dot products over vectors already in memory, no extra embedding
    // call), not a proxy derived from the direct match's score — a
    // neighbor linked to a strong direct hit only because it once scored
    // just over the write-time link threshold isn't necessarily itself
    // relevant to *this* query, and scoring it off the direct match's
    // similarity could let a loosely-related memory crowd out a genuinely
    // relevant one. A small discount still applies on top of the neighbor's
    // real similarity, since it only surfaced associatively, not directly.
    const associativeDiscount = 0.85;
    final byId = {for (final e in entries) e.id: e};
    final associative = <String, double>{};
    for (final id in direct.keys) {
      final node = byId[id];
      if (node == null) continue;
      for (final linkedId in node.linkedIds) {
        if (direct.containsKey(linkedId) || associative.containsKey(linkedId)) {
          continue;
        }
        final neighbor = byId[linkedId];
        if (neighbor == null || !neighbor.isActive) continue;
        final neighborSimilarity =
            _cosineSimilarity(queryEmbedding, neighbor.embedding) * associativeDiscount;
        // Same floor as direct matches — a memory linked to a real match
        // only because it once scored just over the write-time link
        // threshold isn't necessarily relevant to *this* query, and
        // without this check it could still fill a top-k slot ahead of a
        // genuinely relevant memory that simply wasn't linked to anything.
        if (neighborSimilarity < minScore) continue;
        associative[linkedId] = neighborSimilarity;
      }
    }

    final scored = <MapEntry<MemoryEntry, double>>[];
    for (final id in {...direct.keys, ...associative.keys}) {
      final entry = byId[id];
      if (entry == null) continue;
      final similarity = direct[id] ?? associative[id]!;
      scored.add(MapEntry(entry, rank(entry, similarity)));
    }
    scored.sort((a, b) => b.value.compareTo(a.value));
    final result = scored.take(k).map((e) => e.key).toList();

    if (result.isNotEmpty) _reinforce(result);
    return result;
  }

  Timer? _reinforcementPersistTimer;

  void _reinforce(List<MemoryEntry> recalled) {
    final now = DateTime.now();
    for (final e in recalled) {
      final idx = entries.indexWhere((x) => x.id == e.id);
      if (idx == -1) continue;
      entries[idx] = entries[idx].copyWith(
        accessCount: entries[idx].accessCount + 1,
        lastAccessedAt: now,
      );
    }
    // Debounced, not immediate: topK() (and therefore this) runs on nearly
    // every non-trivial chat turn, and _persist() rewrites the ENTIRE
    // store — every stored embedding vector included — to disk. Doing that
    // on every single message would put a full-store disk write on the hot
    // per-message path; coalescing rapid reinforcements into one write a
    // few seconds later keeps retrieval fast while still persisting well
    // before the app would plausibly close. The in-memory ranking effect
    // (which is what actually matters turn-to-turn) is immediate either way.
    _reinforcementPersistTimer?.cancel();
    _reinforcementPersistTimer = Timer(const Duration(seconds: 5), () {
      unawaited(_persist());
    });
  }

  /// Forces any pending debounced reinforcement write to disk right now,
  /// rather than waiting out the timer — used by the periodic memory-health
  /// sweep so "verify memory" also means "make sure it's actually durable",
  /// not just correct in memory.
  Future<void> flushPending() async {
    if (_reinforcementPersistTimer?.isActive ?? false) {
      _reinforcementPersistTimer!.cancel();
      await _persist();
    }
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

  @override
  void onClose() {
    // Flush a pending debounced reinforcement write rather than losing it —
    // this only fires if the service itself is torn down (rare for a
    // GetX service kept for the app's lifetime, but cheap to guard against).
    if (_reinforcementPersistTimer?.isActive ?? false) {
      _reinforcementPersistTimer!.cancel();
      unawaited(_persist());
    }
    super.onClose();
  }
}
