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

    final id = DateTime.now().microsecondsSinceEpoch.toString();
    // entryId isn't in `entries` yet at this point, so _relink only ever
    // adds reverse links to existing neighbors here (nothing to remove) —
    // same net effect as the original inline version, just shared with the
    // enrichment path in addIfNotDuplicate, which needs the remove-stale-
    // links half too when an entry's embedding actually changes.
    final linkedIds = _relink(id, embedding);

    entries.add(MemoryEntry(
      id: id,
      text: text.trim(),
      embedding: embedding,
      createdAt: DateTime.now(),
      sourceChatId: sourceChatId,
      category: category,
      valence: valence,
      tags: tags,
      linkedIds: linkedIds,
    ));
    await _persist();
    return id;
  }

  /// Recomputes [entryId]'s neighbor links against [embedding] and keeps
  /// the graph bidirectional: removes [entryId] from neighbors it's no
  /// longer close enough to, adds it to new ones (respecting the same
  /// FIFO reverse-edge cap as [add]). Returns the fresh forward link list
  /// for [entryId] itself — the caller is responsible for actually storing
  /// that on [entryId]'s own entry, since [entryId] may not exist in
  /// [entries] yet (called from [add] before the new entry is inserted).
  List<String> _relink(String entryId, List<double> embedding) {
    final scored = <MapEntry<MemoryEntry, double>>[];
    for (final entry in entries) {
      if (entry.id == entryId || !entry.isActive) continue;
      final score = _cosineSimilarity(embedding, entry.embedding);
      if (score >= _linkThreshold) scored.add(MapEntry(entry, score));
    }
    scored.sort((a, b) => b.value.compareTo(a.value));
    final newNeighborIds =
        scored.take(_maxLinksPerNode).map((e) => e.key.id).toSet();

    for (var i = 0; i < entries.length; i++) {
      final e = entries[i];
      if (e.id == entryId) continue;
      final hasLink = e.linkedIds.contains(entryId);
      final shouldLink = newNeighborIds.contains(e.id);
      if (hasLink && !shouldLink) {
        entries[i] =
            e.copyWith(linkedIds: e.linkedIds.where((l) => l != entryId).toList());
      } else if (!hasLink && shouldLink) {
        // Cap the reverse edge too — without this, a frequently-referenced
        // memory (e.g. "user's name") accumulates an ever-growing link
        // list over the app's lifetime, contradicting the "small
        // associative graph" this is meant to be. Oldest link drops first
        // (FIFO) to make room, keeping the most recent associations.
        final updated = [...e.linkedIds, entryId];
        final trimmed = updated.length > _maxLinksPerNode
            ? updated.sublist(updated.length - _maxLinksPerNode)
            : updated;
        entries[i] = e.copyWith(linkedIds: trimmed);
      }
    }
    return newNeighborIds.toList();
  }

  /// Same as [add], but if a near-duplicate (cosine similarity above
  /// [dupThreshold]) already exists, that memory is *reinforced* instead —
  /// its access count bumped and last-accessed time refreshed — rather than
  /// silently discarded or stored as a second copy. Saying the same thing
  /// twice should strengthen one memory, not create two.
  ///
  /// A near-duplicate isn't always a pure repeat, though: "Au" and "I am
  /// User and my name is AU" score as near-duplicates (same core fact,
  /// >0.92 cosine) but the second one carries real extra detail the first
  /// didn't. Discarding that detail just because it scored as a duplicate
  /// would silently lose information the user explicitly asked to be
  /// remembered. So when the new text is both meaningfully longer than
  /// what's stored AND a superset of it (contains the old text as a
  /// substring, case-insensitive — cheap and conservative: it guarantees
  /// nothing in the old note is lost by the replacement, without needing
  /// real diffing), the existing entry's text/embedding/tags are updated
  /// in place instead of just bumping its counters — still one memory, one
  /// id, one reinforcement, just now carrying the fuller version of the
  /// fact. A near-duplicate that ISN'T a superset (different phrasing of a
  /// similar-but-not-confirmed-identical fact) stays conservative and only
  /// reinforces, since blindly overwriting there risks losing distinct
  /// detail the old text had that the new one doesn't repeat.
  ///
  /// Returns a record: `wasNew` is true only if a genuinely new entry was
  /// stored; `wasEnriched` is true if an existing entry's text was instead
  /// updated in place per the enrichment case above (distinct from a pure
  /// reinforcement, so a caller can tell "content actually changed" from
  /// "same thing said again" instead of both looking identical); `id` is
  /// that new/updated entry's id either way, letting a caller (e.g.
  /// contradiction handling) reference whichever entry actually now
  /// represents this fact. Logging/toasting on the outcome is the caller's
  /// job, not this method's, so it stays in one place rather than split
  /// between here and every call site.
  /// [scope], when given, restricts which existing entries can even be
  /// considered a near-duplicate — used by memory consolidation, whose
  /// synthesized connecting sentence is expected to score similar to the
  /// very source memories it's summarizing (that's the point), which would
  /// otherwise make it get silently swallowed as a "duplicate" of one of
  /// them instead of stored as the new note it actually is. Restricting
  /// the check to other consolidation notes still dedupes repeated sweeps
  /// against each other without treating "summarizes X" as "is X".
  Future<({bool wasNew, bool wasEnriched, String? id})> addIfNotDuplicate(
    String text,
    List<double> embedding, {
    String? sourceChatId,
    String category = 'general',
    String valence = 'neutral',
    List<String> tags = const [],
    double dupThreshold = 0.92,
    bool Function(MemoryEntry)? scope,
  }) async {
    final trimmedNew = text.trim();
    if (trimmedNew.isEmpty || embedding.isEmpty) {
      return (wasNew: false, wasEnriched: false, id: null);
    }
    for (var i = 0; i < entries.length; i++) {
      final existing = entries[i];
      if (!existing.isActive) continue;
      if (scope != null && !scope(existing)) continue;
      if (_cosineSimilarity(embedding, existing.embedding) >= dupThreshold) {
        // Word-boundary match, not a raw substring check — a short/generic
        // existing memory (e.g. "Al") would otherwise false-positive as
        // "contained in" any longer new text that merely happens to embed
        // those same letters (e.g. inside "global"), silently overwriting
        // an unrelated memory just because it scored as embedding-similar.
        // Leading/trailing punctuation is stripped from the needle before
        // wrapping it in \b — these distilled notes routinely end in a
        // period, and \b can't match at a punctuation→space transition (no
        // word/non-word boundary there), which would otherwise make a
        // completely genuine superset match (e.g. "Name is Sarah." inside
        // "Name is Sarah. She works in Boston.") fail to be recognized.
        final needle =
            existing.text.trim().replaceAll(RegExp(r'^[^\w]+|[^\w]+$'), '');
        final isEnrichment = trimmedNew.length > existing.text.length + 8 &&
            needle.isNotEmpty &&
            RegExp(
              r'\b' + RegExp.escape(needle) + r'\b',
              caseSensitive: false,
            ).hasMatch(trimmedNew);
        entries[i] = isEnrichment
            // Only the text/embedding/tags/links actually change here —
            // category and valence are deliberately NOT overwritten from
            // the caller's arguments. An existing entry's classification
            // (e.g. 'instruction') is meaningful and shouldn't get
            // silently clobbered by whatever category a later, unrelated
            // caller (e.g. the consolidation sweep, which always passes
            // 'summary') happens to pass just because its text scored as
            // an enrichment of this one.
            ? existing.copyWith(
                text: trimmedNew,
                embedding: embedding,
                tags: {...existing.tags, ...tags}.toList(),
                // The embedding just changed, possibly substantially (a
                // bare name enriched into a full sentence) — recompute
                // this entry's place in the associative graph rather than
                // leaving it pointing at neighbors picked for the old,
                // narrower embedding.
                linkedIds: _relink(existing.id, embedding),
                accessCount: existing.accessCount + 1,
                lastAccessedAt: DateTime.now(),
              )
            : existing.copyWith(
                accessCount: existing.accessCount + 1,
                lastAccessedAt: DateTime.now(),
              );
        await _persist();
        return (wasNew: false, wasEnriched: isEnrichment, id: entries[i].id);
      }
    }
    final newId = await add(
      trimmedNew,
      embedding,
      sourceChatId: sourceChatId,
      category: category,
      valence: valence,
      tags: tags,
    );
    return (wasNew: true, wasEnriched: false, id: newId);
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
