import 'package:hive/hive.dart';

part 'memory_entry.g.dart';

/// A single durable, cross-chat fact about the user — extracted from a
/// conversation and recalled in any future chat, independent of which
/// chat it originated in. This is separate from a ChatModel's own message
/// history: chats are the transcript, memories are what got distilled
/// out of them for the long term.
@HiveType(typeId: 3)
class MemoryEntry extends HiveObject {
  @HiveField(0)
  final String id;

  @HiveField(1)
  final String text;

  @HiveField(2)
  final DateTime createdAt;

  /// Best-effort date the fact actually refers to (usually the timestamp
  /// of the message it was extracted from). Kept separate from [createdAt]
  /// so recall months later can say "on Sept 22" instead of implying the
  /// fact was learned today.
  @HiveField(3)
  final DateTime? eventDate;

  /// Chat the fact was extracted from, for traceability only — recall is
  /// intentionally cross-chat and never filters by this.
  @HiveField(4)
  final String sourceChatId;

  /// Embedding vector for semantic recall. Null until an embedding model
  /// is available to compute it; MemoryService lazily backfills these.
  @HiveField(5)
  List<double>? embedding;

  MemoryEntry({
    required this.id,
    required this.text,
    DateTime? createdAt,
    this.eventDate,
    this.sourceChatId = '',
    this.embedding,
  }) : createdAt = createdAt ?? DateTime.now();

  bool get hasEmbedding => embedding != null && embedding!.isNotEmpty;
}
