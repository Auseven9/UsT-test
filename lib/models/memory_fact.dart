import 'package:hive/hive.dart';

part 'memory_fact.g.dart';

/// A single durable, atomic fact recalled across chats — not raw transcript.
///
/// Facts are short, standalone statements ("user's name is Dylon", "prefers
/// dark mode") extracted from conversation, persisted independently of any
/// one [ChatModel], and retrieved on later turns via [MemoryService.recall].
@HiveType(typeId: 3)
class MemoryFact extends HiveObject {
  @HiveField(0)
  final String id;

  @HiveField(1)
  String text;

  @HiveField(2)
  final DateTime createdAt;

  @HiveField(3)
  DateTime? lastRecalledAt;

  @HiveField(4)
  int recallCount;

  @HiveField(5)
  String? sourceChatId;

  /// Cached embedding vector for [text], computed lazily on first recall
  /// attempt while a model is loaded. Null until then, or if the loaded
  /// model/backend doesn't support embeddings — recall falls back to a
  /// lexical match in that case.
  @HiveField(6)
  List<double>? embedding;

  MemoryFact({
    required this.id,
    required this.text,
    DateTime? createdAt,
    this.lastRecalledAt,
    this.recallCount = 0,
    this.sourceChatId,
    this.embedding,
  }) : createdAt = createdAt ?? DateTime.now();
}
