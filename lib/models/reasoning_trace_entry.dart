/// One turn's chain-of-thought, distilled down to a short gist and stored
/// in its own lane — separate from fact memory (`MemoryEntry`), since a
/// reasoning trace is "how I arrived at something" rather than "something
/// durable that's true". Plain, JSON-serializable (no Hive adapter, no
/// embedding) — retrieval here is recency-based, not semantic search.
class ReasoningTraceEntry {
  final String id;
  final DateTime createdAt;
  final String chatId;
  String summary;

  /// Whether [summary] has already been shortened once by
  /// `ReasoningTraceService.runLaneMaintenance` — prevents a trace that
  /// ages past the full-detail window from being compressed over and over
  /// on every later maintenance pass.
  bool compressed;

  ReasoningTraceEntry({
    required this.id,
    required this.createdAt,
    required this.chatId,
    required this.summary,
    this.compressed = false,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'createdAt': createdAt.toIso8601String(),
        'chatId': chatId,
        'summary': summary,
        'compressed': compressed,
      };

  factory ReasoningTraceEntry.fromJson(Map<String, dynamic> json) => ReasoningTraceEntry(
        id: json['id'] as String? ?? DateTime.now().microsecondsSinceEpoch.toString(),
        createdAt: DateTime.tryParse(json['createdAt'] as String? ?? '') ?? DateTime.now(),
        chatId: json['chatId'] as String? ?? '',
        summary: json['summary'] as String? ?? '',
        compressed: json['compressed'] as bool? ?? false,
      );
}
