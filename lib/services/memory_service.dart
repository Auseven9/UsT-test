import 'dart:math';

import 'package:get/get.dart';
import 'package:hive/hive.dart';

import '../models/memory_fact.dart';
import 'llm_service.dart';

/// Cross-chat durable memory: a small store of atomic facts, independent of
/// any single [ChatModel], recalled on every turn before generation.
///
/// This deliberately does NOT try to replay or summarize whole transcripts.
/// It holds short standalone statements ("user's name is Dylon") and answers
/// one question per turn: "do I know something relevant to this message?" —
/// if so, that (bounded) context is injected instead of the growing chat
/// history, and the caller can surface that a recall happened.
class MemoryService extends GetxService {
  static const _boxName = 'memory_facts';

  late Box<MemoryFact> _box;
  LlmService? _llm;

  /// Matches below this score are not considered a recall hit.
  static const double _matchThreshold = 0.32;

  Future<MemoryService> init() async {
    _box = Hive.box<MemoryFact>(_boxName);
    try {
      _llm = Get.find<LlmService>();
    } catch (_) {
      _llm = null;
    }
    return this;
  }

  List<MemoryFact> get allFacts => _box.values.toList()
    ..sort((a, b) => b.createdAt.compareTo(a.createdAt));

  Future<MemoryFact> addFact(String text, {String? sourceChatId}) async {
    final fact = MemoryFact(
      id: '${DateTime.now().microsecondsSinceEpoch}',
      text: text.trim(),
      sourceChatId: sourceChatId,
    );
    await _box.put(fact.id, fact);
    return fact;
  }

  Future<void> deleteFact(String id) => _box.delete(id);

  Future<void> clearAll() async {
    await _box.clear();
  }

  // ── Recall ───────────────────────────────────────────────────────

  /// Look up facts relevant to [query]. Tries embedding similarity first
  /// (when a model is loaded and supports it), falling back to lexical
  /// overlap otherwise — always returns, never throws.
  Future<List<MemoryFact>> recall(String query, {int topK = 3}) async {
    if (_box.isEmpty || query.trim().isEmpty) return const [];

    final queryEmbedding = await _llm?.embed(query);
    final scored = <_ScoredFact>[];

    if (queryEmbedding != null) {
      for (final fact in _box.values) {
        final factEmbedding = await _embeddingFor(fact);
        if (factEmbedding == null) continue;
        final score = _cosineSimilarity(queryEmbedding, factEmbedding);
        scored.add(_ScoredFact(fact, score));
      }
    }

    // Embedding path found nothing usable (no model loaded, unsupported
    // backend, or every fact still lacks a cached vector) — fall back.
    if (scored.isEmpty) {
      final queryTokens = _tokenize(query);
      if (queryTokens.isEmpty) return const [];
      for (final fact in _box.values) {
        final score = _lexicalOverlap(queryTokens, _tokenize(fact.text));
        scored.add(_ScoredFact(fact, score));
      }
    }

    scored.sort((a, b) => b.score.compareTo(a.score));
    final hits = scored
        .where((s) => s.score >= _matchThreshold)
        .take(topK)
        .toList();

    for (final hit in hits) {
      hit.fact.recallCount++;
      hit.fact.lastRecalledAt = DateTime.now();
      await hit.fact.save();
    }

    return hits.map((s) => s.fact).toList();
  }

  Future<List<double>?> _embeddingFor(MemoryFact fact) async {
    if (fact.embedding != null) return fact.embedding;
    final vec = await _llm?.embed(fact.text);
    if (vec != null) {
      fact.embedding = vec;
      await fact.save();
    }
    return vec;
  }

  // ── Cheap heuristic extraction ──────────────────────────────────

  /// Regex-based first pass over a finished exchange. Cheap enough to run
  /// after every assistant turn — no model call, no extra latency.
  /// A periodic LLM-based consolidation pass (batching several turns into
  /// higher-quality facts) is a good follow-up but isn't implemented here.
  /// Each entry pairs a pattern with a template for the fact text it
  /// produces — `{}` is replaced with the captured group.
  static final List<(RegExp, String)> _factPatterns = [
    (
      RegExp(r"\bmy name is ([A-Za-z][\w' -]{1,40})", caseSensitive: false),
      'Name: {}',
    ),
    (
      RegExp(r"\bcall me ([A-Za-z][\w' -]{1,40})", caseSensitive: false),
      'Prefers to be called: {}',
    ),
    (
      RegExp(
        r"\bi(?:'m| am) (?:working on|building) ([\w' .,-]{3,80})",
        caseSensitive: false,
      ),
      'Is working on: {}',
    ),
    (
      RegExp(r"\bi prefer ([\w' .,-]{3,80})", caseSensitive: false),
      'Prefers: {}',
    ),
    (
      RegExp(r"\bi(?:'m| am) (?:a|an) ([\w' .,-]{3,60})", caseSensitive: false),
      'Is: {}',
    ),
    (
      RegExp(r"\bi live in ([\w' .,-]{2,60})", caseSensitive: false),
      'Lives in: {}',
    ),
    (
      RegExp(r"\bremember that ([\w' .,-]{3,120})", caseSensitive: false),
      '{}',
    ),
  ];

  /// Extracts facts from [userText] and persists new ones, skipping near
  /// duplicates of what's already stored.
  Future<List<MemoryFact>> extractHeuristic(
    String userText, {
    String? sourceChatId,
  }) async {
    final found = <String>{};
    for (final (pattern, template) in _factPatterns) {
      for (final match in pattern.allMatches(userText)) {
        final captured = match.group(1)?.trim();
        if (captured == null || captured.isEmpty) continue;
        found.add(template.replaceFirst('{}', captured));
      }
    }

    if (found.isEmpty) return const [];

    final existingTexts = _box.values.map((f) => f.text.toLowerCase()).toSet();
    final added = <MemoryFact>[];
    for (final normalized in found) {
      if (normalized.length < 4) continue;
      final isDuplicate = existingTexts.any(
        (existing) =>
            existing.contains(normalized.toLowerCase()) ||
            normalized.toLowerCase().contains(existing),
      );
      if (isDuplicate) continue;
      added.add(await addFact(normalized, sourceChatId: sourceChatId));
    }
    return added;
  }

  // ── Scoring helpers ──────────────────────────────────────────────

  static final RegExp _wordPattern = RegExp(r"[a-z0-9']+");
  static const Set<String> _stopwords = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'in',
    'on', 'for', 'it', 'i', 'you', 'me', 'my', 'do', 'does', 'did', 'what',
    'this', 'that', 'with', 'at', 'be', 'or', 'as', 'so', 'can', 'will',
  };

  Set<String> _tokenize(String text) => _wordPattern
      .allMatches(text.toLowerCase())
      .map((m) => m.group(0)!)
      .where((w) => w.length > 1 && !_stopwords.contains(w))
      .toSet();

  double _lexicalOverlap(Set<String> a, Set<String> b) {
    if (a.isEmpty || b.isEmpty) return 0.0;
    final intersection = a.intersection(b).length;
    final union = a.union(b).length;
    if (union == 0) return 0.0;
    return intersection / union; // Jaccard
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length || a.isEmpty) return 0.0;
    double dot = 0, normA = 0, normB = 0;
    for (var i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    if (normA == 0 || normB == 0) return 0.0;
    return dot / (sqrt(normA) * sqrt(normB));
  }
}

class _ScoredFact {
  final MemoryFact fact;
  final double score;
  _ScoredFact(this.fact, this.score);
}
