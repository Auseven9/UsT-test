/// A single persisted memory node — a distilled fact or preference the app
/// chose to remember, its embedding vector for retrieval, and the metadata
/// that makes recall behave less like a flat search and more like
/// association: links to related memories, how often and recently it's
/// been recalled, and whether a later, contradicting memory has superseded
/// it (memories are never deleted by the model — see MemoryService — but a
/// superseded one is deprioritized in recall so an outdated fact stops
/// competing equally with the current one).
class MemoryEntry {
  final String id;
  final String text;
  final List<double> embedding;
  final DateTime createdAt;
  final String? sourceChatId;

  /// Coarse kind of memory — 'fact' (durable, context-free), 'preference',
  /// 'event' (tied to a specific conversation/moment), 'instruction'
  /// (an explicit "remember this"), or 'general' when unclassified.
  final String category;

  /// Estimated tone of the memory's content — 'positive', 'negative', or
  /// 'neutral'. A label on the content, not a claim that anything was felt.
  final String valence;

  final List<String> tags;

  /// IDs of the most semantically similar other memories at the time this
  /// one was stored — a small associative graph built from embeddings
  /// already being computed, at zero extra inference cost. Retrieval walks
  /// one hop along these links so recalling a memory can surface related
  /// ones, not just the single closest match.
  final List<String> linkedIds;

  /// When this memory was last surfaced in a retrieval — recall itself
  /// reinforces a memory, same as [accessCount].
  final DateTime lastAccessedAt;

  /// How many times this memory has been recalled or re-stated. A memory
  /// mentioned repeatedly ranks higher in retrieval than one mentioned once
  /// and never touched again — reinforcement, not just raw similarity.
  final int accessCount;

  /// If a later memory contradicted this one (e.g. a changed fact), the
  /// contradicting memory's ID goes here. Never deleted, never hidden from
  /// the memory browser — just excluded from normal retrieval ranking so an
  /// outdated fact doesn't compete with the current one.
  final String? supersededBy;

  const MemoryEntry({
    required this.id,
    required this.text,
    required this.embedding,
    required this.createdAt,
    this.sourceChatId,
    this.category = 'general',
    this.valence = 'neutral',
    this.tags = const [],
    this.linkedIds = const [],
    DateTime? lastAccessedAt,
    this.accessCount = 0,
    this.supersededBy,
  }) : lastAccessedAt = lastAccessedAt ?? createdAt;

  bool get isActive => supersededBy == null;

  MemoryEntry copyWith({
    List<String>? linkedIds,
    DateTime? lastAccessedAt,
    int? accessCount,
    String? supersededBy,
    // MemoryService.delete() un-supersedes an entry when the memory that
    // superseded it is itself deleted (nothing contradicts this one
    // anymore) — that needs to explicitly set supersededBy back to null,
    // which a plain `supersededBy: null` argument can't distinguish from
    // "caller didn't pass anything, keep the old value" under `??`.
    bool clearSupersededBy = false,
  }) {
    return MemoryEntry(
      id: id,
      text: text,
      embedding: embedding,
      createdAt: createdAt,
      sourceChatId: sourceChatId,
      category: category,
      valence: valence,
      tags: tags,
      linkedIds: linkedIds ?? this.linkedIds,
      lastAccessedAt: lastAccessedAt ?? this.lastAccessedAt,
      accessCount: accessCount ?? this.accessCount,
      supersededBy: clearSupersededBy ? null : (supersededBy ?? this.supersededBy),
    );
  }

  factory MemoryEntry.fromJson(Map<String, dynamic> json) {
    final createdAt = DateTime.parse(json['createdAt'] as String);
    return MemoryEntry(
      id: json['id'] as String,
      text: json['text'] as String,
      embedding: (json['embedding'] as List)
          .map((e) => (e as num).toDouble())
          .toList(),
      createdAt: createdAt,
      sourceChatId: json['sourceChatId'] as String?,
      // Older stored entries won't have these keys at all — default to the
      // same neutral values new entries start with, not null/crash.
      category: json['category'] as String? ?? 'general',
      valence: json['valence'] as String? ?? 'neutral',
      tags: (json['tags'] as List?)?.map((e) => e as String).toList() ?? const [],
      linkedIds:
          (json['linkedIds'] as List?)?.map((e) => e as String).toList() ?? const [],
      lastAccessedAt: json['lastAccessedAt'] != null
          ? DateTime.parse(json['lastAccessedAt'] as String)
          : createdAt,
      accessCount: (json['accessCount'] as num?)?.toInt() ?? 0,
      supersededBy: json['supersededBy'] as String?,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'text': text,
        'embedding': embedding,
        'createdAt': createdAt.toIso8601String(),
        'sourceChatId': sourceChatId,
        'category': category,
        'valence': valence,
        'tags': tags,
        'linkedIds': linkedIds,
        'lastAccessedAt': lastAccessedAt.toIso8601String(),
        'accessCount': accessCount,
        'supersededBy': supersededBy,
      };
}
