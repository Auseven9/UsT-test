import 'package:get/get.dart';

import '../models/message_model.dart';
import 'llm_service.dart';
import 'log_service.dart';

/// Fits persisted chat history into the active model's fixed context window.
///
/// Chat history itself lives forever in Hive via [ChatStorageService] — this
/// service does not touch persistence. Its only job is deciding, on each
/// generation call, how much of that history actually fits the model's
/// token budget: newest turns are always kept, oldest turns are dropped
/// first, and the decision is logged so a degraded conversation is visible
/// rather than a silent context overflow inside the native engine.
class ConversationMemoryService extends GetxService {
  final LlmService _llm = Get.find<LlmService>();

  /// Tokens reserved for the model's own reply.
  static const int _responseReserve = 256;

  /// Extra headroom so the prompt never lands exactly on the boundary.
  static const int _safetyMargin = 32;

  /// Fallback estimate (chars/token) used only when native token counting
  /// is unavailable or fails for non-empty text — keeps trimming safe
  /// instead of treating uncounted text as free.
  static const int _charsPerTokenEstimate = 4;

  LogService? get _log {
    try {
      return Get.find<LogService>();
    } catch (_) {
      return null;
    }
  }

  /// Returns the newest-first-selected, chronologically-ordered subset of
  /// [messages] that fits alongside [systemPrompt] in the active model's
  /// context window. Returns [messages] unchanged if no model is loaded
  /// (contextSize unknown) or the list is empty.
  Future<List<MessageModel>> fitToContext(
    List<MessageModel> messages, {
    required String systemPrompt,
  }) async {
    final contextSize = _llm.contextSize.value;
    if (contextSize <= 0 || messages.isEmpty) return messages;

    final budget = contextSize - _responseReserve - _safetyMargin;
    if (budget <= 0) return [messages.last];

    var remaining = budget - await _tokenCount(systemPrompt);

    final kept = <MessageModel>[];
    for (final msg in messages.reversed) {
      final cost = await _tokenCount(msg.content);
      if (kept.isEmpty) {
        // Always keep at least the most recent message — dropping what the
        // user just sent is worse than letting a single oversized turn
        // exceed the budget on its own.
        kept.add(msg);
        remaining -= cost;
        continue;
      }
      if (cost > remaining) break;
      kept.add(msg);
      remaining -= cost;
    }

    final trimmed = kept.reversed.toList();
    final droppedCount = messages.length - trimmed.length;

    if (droppedCount > 0) {
      _log?.warn(
        'Context window full: dropped $droppedCount oldest message(s) '
        '(kept ${trimmed.length}/${messages.length}, ctx=$contextSize tokens) '
        'to stay under budget.',
        source: 'Memory',
      );
    } else {
      _log?.info(
        'Context fit: ${trimmed.length}/${trimmed.length} messages, '
        '~${budget - remaining}/$budget prompt tokens used.',
        source: 'Memory',
      );
    }

    return trimmed;
  }

  Future<int> _tokenCount(String text) async {
    if (text.isEmpty) return 0;
    final counted = await _llm.countTokens(text);
    if (counted > 0) return counted;

    // Native counting returned 0 for non-empty text — either it failed or
    // no engine is loaded. Fall back to a conservative character-based
    // estimate so trimming still degrades safely instead of assuming free.
    final estimate = (text.length / _charsPerTokenEstimate).ceil();
    if (estimate > 0) {
      _log?.warn(
        'Native token count unavailable for a message; using estimate '
        '(~$estimate tokens).',
        source: 'Memory',
      );
    }
    return estimate;
  }
}
