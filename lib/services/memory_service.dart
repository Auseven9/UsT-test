import 'package:get/get.dart';
import 'package:hive/hive.dart';

import '../models/chat_model.dart';
import '../models/memory_entry.dart';
import '../models/message_model.dart';
import 'chat_storage_service.dart';
import 'embedding_service.dart';
import 'llm_service.dart';
import 'log_service.dart';

/// Long-term, cross-chat memory: durable personal facts distilled out of
/// conversations (not the conversations themselves) and recalled in any
/// future chat regardless of where they were said.
///
/// Two independent, best-effort halves:
///  - extraction: after a turn, ask the already-loaded chat model whether
///    anything in it is worth remembering long-term, and store what it finds.
///  - recall: given a new user message, semantically search stored memories
///    and return the most relevant ones to inject as context.
///
/// Both are gated on ChatStorageService.smartRecallEnabled and never throw
/// into the caller — a failure here should degrade to "no memory," not
/// break the chat.
class MemoryService extends GetxService {
  final LlmService _llm = Get.find<LlmService>();
  final EmbeddingService _embedder = Get.find<EmbeddingService>();
  final ChatStorageService _storage = Get.find<ChatStorageService>();

  late Box<MemoryEntry> _box;

  /// Hard cap on extraction output so a rambling model can't turn a
  /// background memory pass into an unbounded generation.
  static const int _maxExtractionChars = 600;

  static const int _defaultTopK = 5;
  static const double _minRecallScore = 0.55;

  /// Bound on how many un-embedded memories get backfilled per recall call,
  /// so an old backlog can't stall a single chat turn.
  static const int _backfillBatchSize = 20;

  LogService? get _log {
    try {
      return Get.find<LogService>();
    } catch (_) {
      return null;
    }
  }

  Future<MemoryService> init() async {
    _box = Hive.box<MemoryEntry>('memories');
    return this;
  }

  // ── Read access for the management screen ──────────────────────

  List<MemoryEntry> getAll() {
    final list = _box.values.toList();
    list.sort((a, b) => b.createdAt.compareTo(a.createdAt));
    return list;
  }

  Future<void> delete(String id) async {
    await _box.delete(id);
  }

  Future<void> clearAll() async {
    await _box.clear();
  }

  // ── Extraction ───────────────────────────────────────────────────

  /// Best-effort: reads the exchange, asks the loaded chat model to pull
  /// out any durable personal facts, embeds and stores each one. Silently
  /// no-ops if the feature is off, no model is loaded, or anything fails.
  Future<void> extractAndStore({
    required ChatModel chat,
    required MessageModel userMessage,
    required MessageModel assistantMessage,
  }) async {
    if (!_storage.smartRecallEnabled) return;
    if (!_llm.isLoaded.value) return;
    if (userMessage.content.trim().isEmpty) return;

    try {
      final prompt = _buildExtractionPrompt(userMessage, assistantMessage);
      final raw = await _runCappedExtraction(prompt);
      final facts = _parseFacts(raw);
      if (facts.isEmpty) return;

      for (var i = 0; i < facts.length; i++) {
        final fact = facts[i];
        final entry = MemoryEntry(
          id: '${DateTime.now().microsecondsSinceEpoch}-$i',
          text: fact,
          eventDate: userMessage.timestamp,
          sourceChatId: chat.id,
        );
        entry.embedding = await _embedder.embed(fact);
        await _box.put(entry.id, entry);
      }

      _log?.info(
        'Stored ${facts.length} new long-term memor${facts.length == 1 ? 'y' : 'ies'}.',
        source: 'Memory',
      );
    } catch (e) {
      _log?.warn('Memory extraction skipped: $e', source: 'Memory');
    }
  }

  String _buildExtractionPrompt(
    MessageModel userMessage,
    MessageModel assistantMessage,
  ) {
    return 'User said: "${userMessage.content.trim()}"\n'
        'Assistant replied: "${assistantMessage.content.trim()}"';
  }

  String _extractionSystemPrompt(DateTime anchor) {
    return 'You are a silent memory-extraction tool running inside a chat app. '
        "Today's date is ${_formatDate(anchor)}. "
        'Read the exchange and decide if it contains any durable, personal '
        "fact about the user worth remembering months from now — people, "
        'relationships, plans, events, health, milestones, preferences. '
        'Ignore small talk, opinions asked of the assistant, and anything not '
        "about the user's real life. Resolve relative dates like \"today\", "
        '"yesterday", or "next spring" into an actual date or season using '
        "today's date above, so the fact still makes sense read later. "
        'Output each fact as its own line starting with "- ". '
        'If nothing is worth remembering, output exactly: NONE';
  }

  Future<String> _runCappedExtraction(String userContent) async {
    final buffer = StringBuffer();
    final stream = _llm.generate(
      messages: [
        {'role': 'user', 'content': userContent},
      ],
      systemPrompt: _extractionSystemPrompt(DateTime.now()),
      temperature: 0.3,
    );

    await for (final token in stream) {
      buffer.write(token);
      if (buffer.length >= _maxExtractionChars) {
        await _llm.stopGeneration();
        break;
      }
    }
    return buffer.toString();
  }

  List<String> _parseFacts(String raw) {
    final trimmed = raw.trim();
    if (trimmed.isEmpty || trimmed.toUpperCase() == 'NONE') return [];

    final facts = <String>[];
    for (final line in trimmed.split('\n')) {
      final l = line.trim();
      if (!l.startsWith('-')) continue;
      final fact = l.substring(1).trim();
      if (fact.isEmpty || fact.toUpperCase() == 'NONE') continue;
      facts.add(fact);
    }
    return facts;
  }

  // ── Recall ───────────────────────────────────────────────────────

  /// Semantically searches stored memories for [query]. Returns the empty
  /// list (never throws) when the feature is off or no embedding model is
  /// loaded — recall is opt-in and degrades to nothing, not a partial
  /// keyword guess.
  Future<List<MemoryEntry>> recall(
    String query, {
    int topK = _defaultTopK,
  }) async {
    if (!_storage.smartRecallEnabled) return [];
    if (!_embedder.isLoaded.value) return [];
    if (_box.isEmpty) return [];

    final queryVector = await _embedder.embed(query);
    if (queryVector == null) return [];

    await _backfillMissingEmbeddings();

    final scored = <MapEntry<MemoryEntry, double>>[];
    for (final entry in _box.values) {
      if (!entry.hasEmbedding) continue;
      final score = EmbeddingService.cosineSimilarity(
        queryVector,
        entry.embedding!,
      );
      if (score >= _minRecallScore) {
        scored.add(MapEntry(entry, score));
      }
    }

    scored.sort((a, b) => b.value.compareTo(a.value));
    final top = scored.take(topK).map((e) => e.key).toList();

    if (top.isNotEmpty) {
      _log?.info(
        'Recalled ${top.length} relevant memor${top.length == 1 ? 'y' : 'ies'} for this turn.',
        source: 'Memory',
      );
    }
    return top;
  }

  Future<void> _backfillMissingEmbeddings() async {
    final missing = _box.values.where((m) => !m.hasEmbedding).take(
      _backfillBatchSize,
    );
    for (final entry in missing) {
      final vector = await _embedder.embed(entry.text);
      if (vector != null) {
        entry.embedding = vector;
        await _box.put(entry.id, entry);
      }
    }
  }

  static const _months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
  ];

  String _formatDate(DateTime dt) => '${_months[dt.month - 1]} ${dt.day}, ${dt.year}';
}
