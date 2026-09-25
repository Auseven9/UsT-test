/// A single persisted memory node — a distilled fact or preference the app
/// chose to remember, its embedding vector for retrieval, and the metadata
/// that makes recall behave less like a flat search and more like
/// association: links to related memories, how often and recently it's
/// been recalled, and whether a later, contradicting memory has superseded
/// it (memories are never deleted — see MemoryService — but a superseded
/// one is deprioritized in recall so an outdated fact stops competing
/// equally with the current one).
class MemoryEntry {
  final String id;
  final String text;
  final List<double> embedding;
  final DateTime createdAt;
  final String? sourceChatId;

  /// Coarse kind of memory — 'fact' (durable, context-free), 'preference',
  /// 'event' (tied to a specific conversation/moment), 'instruction'
  /// (an explicit "remember this"), 'summary' (written by the memory-
  /// consolidation sweep, not a single exchange), or 'general' when
  /// unclassified.
  final String category;

  /// Estimated tone of the memory's content — 'positive', 'negative', or
  /// 'neutral'. A label on the content, not a claim that anything was felt.
  final String valence;

  final List<String> tags;

  /// IDs of the most semantically similar other memories at the time this
  /// one was stored — a small associative graph built from embeddings
  /// already being computed, at zero extra inference cost. Retrieval walks
  /// outward along these links (see MemoryService.topK's hop limit) so
  /// recalling a memory can surface related ones, not just the single
  /// closest match.
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

  /// How confident the extraction was that this is actually true/durable,
  /// 0.0–1.0. Not shown as false precision to the user — used internally to
  /// decide contradiction handling (see MemoryService.resolveContradiction):
  /// a high-confidence correction supersedes the old fact outright; a
  /// low-confidence one is kept alongside it instead, both active, until
  /// reinforcement or a further exchange resolves which one is right.
  /// Defaults to 1.0 for entries written before this field existed, or
  /// written through a path that doesn't estimate confidence (manual entry,
  /// consolidation) — treated as certain rather than penalized for missing
  /// data it was never asked to produce.
  final double confidence;

  /// 'episodic' (tied to a specific conversation/moment — "asked about X
  /// on Tuesday") or 'semantic' (a durable, context-free fact — "prefers
  /// Y"). Extraction defaults new entries to 'episodic'. Nothing promotes
  /// this automatically yet — MemoryService's working-memory maintenance
  /// only flips [isWorkingMemory] (probation to permanent), not this field;
  /// the only way an entry becomes 'semantic' today is a manual edit in the
  /// memory browser. An episodic memory that keeps getting reinforced is
  /// the right eventual candidate for that graduation, but the automatic
  /// version of it doesn't exist yet.
  final String memoryType;

  /// Whose fact this is — 'user' (the default: something about the person
  /// talking to the model) or 'assistant' (something about the model's own
  /// persona, instructions, or corrected behavior — kept in the same store
  /// but distinguishable, so "what am I supposed to be" and "what do I know
  /// about the user" don't silently blend into one undifferentiated list).
  final String subject;

  /// Earlier versions of [text], oldest first, kept when an enrichment
  /// (see MemoryService.addIfNotDuplicate) updates this entry's text in
  /// place rather than superseding it outright — a superseded correction
  /// already keeps its full old entry untouched, but an in-place enrichment
  /// previously discarded the prior wording entirely. This is that history.
  final List<String> priorTexts;

  /// True while this memory is in the short-lived working-memory tier —
  /// captured but not yet promoted to permanent storage. See
  /// MemoryService's decay sweep: a working-memory entry that gets
  /// reinforced (recalled again, referenced again) gets promoted
  /// (`isWorkingMemory` flips false); one that never gets touched again
  /// within the decay window gets deleted, same as short-term memory that
  /// was never worth keeping. A permanent entry (`false`) is never demoted
  /// back to working memory.
  final bool isWorkingMemory;

  /// How many times idle-time rehearsal (the consolidation sweep) has
  /// deliberately re-touched this memory to keep it fresh, distinct from
  /// [accessCount] (which counts actual recall during a real
  /// conversation). Purely a rehearsal counter — it does not by itself
  /// change ranking beyond the recency boost [lastAccessedAt] already
  /// gives everything rehearsal also updates.
  final int rehearsalCount;

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
    this.confidence = 1.0,
    this.memoryType = 'episodic',
    this.subject = 'user',
    this.priorTexts = const [],
    this.isWorkingMemory = false,
    this.rehearsalCount = 0,
  }) : lastAccessedAt = lastAccessedAt ?? createdAt;

  bool get isActive => supersededBy == null;

  MemoryEntry copyWith({
    String? text,
    List<double>? embedding,
    String? category,
    String? valence,
    List<String>? tags,
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
    double? confidence,
    String? memoryType,
    String? subject,
    List<String>? priorTexts,
    bool? isWorkingMemory,
    int? rehearsalCount,
  }) {
    return MemoryEntry(
      id: id,
      text: text ?? this.text,
      embedding: embedding ?? this.embedding,
      createdAt: createdAt,
      sourceChatId: sourceChatId,
      category: category ?? this.category,
      valence: valence ?? this.valence,
      tags: tags ?? this.tags,
      linkedIds: linkedIds ?? this.linkedIds,
      lastAccessedAt: lastAccessedAt ?? this.lastAccessedAt,
      accessCount: accessCount ?? this.accessCount,
      supersededBy: clearSupersededBy ? null : (supersededBy ?? this.supersededBy),
      confidence: confidence ?? this.confidence,
      memoryType: memoryType ?? this.memoryType,
      subject: subject ?? this.subject,
      priorTexts: priorTexts ?? this.priorTexts,
      isWorkingMemory: isWorkingMemory ?? this.isWorkingMemory,
      rehearsalCount: rehearsalCount ?? this.rehearsalCount,
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
      confidence: (json['confidence'] as num?)?.toDouble() ?? 1.0,
      memoryType: json['memoryType'] as String? ?? 'episodic',
      subject: json['subject'] as String? ?? 'user',
      priorTexts:
          (json['priorTexts'] as List?)?.map((e) => e as String).toList() ?? const [],
      isWorkingMemory: json['isWorkingMemory'] as bool? ?? false,
      rehearsalCount: (json['rehearsalCount'] as num?)?.toInt() ?? 0,
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
        'confidence': confidence,
        'memoryType': memoryType,
        'subject': subject,
        'priorTexts': priorTexts,
        'isWorkingMemory': isWorkingMemory,
        'rehearsalCount': rehearsalCount,
      };
}
