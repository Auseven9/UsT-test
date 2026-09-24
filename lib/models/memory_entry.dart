/// A single persisted memory note — a distilled fact or preference the app
/// chose to remember, plus the embedding vector used to retrieve it later.
class MemoryEntry {
  final String id;
  final String text;
  final List<double> embedding;
  final DateTime createdAt;
  final String? sourceChatId;

  const MemoryEntry({
    required this.id,
    required this.text,
    required this.embedding,
    required this.createdAt,
    this.sourceChatId,
  });

  factory MemoryEntry.fromJson(Map<String, dynamic> json) {
    return MemoryEntry(
      id: json['id'] as String,
      text: json['text'] as String,
      embedding: (json['embedding'] as List)
          .map((e) => (e as num).toDouble())
          .toList(),
      createdAt: DateTime.parse(json['createdAt'] as String),
      sourceChatId: json['sourceChatId'] as String?,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'text': text,
        'embedding': embedding,
        'createdAt': createdAt.toIso8601String(),
        'sourceChatId': sourceChatId,
      };
}
