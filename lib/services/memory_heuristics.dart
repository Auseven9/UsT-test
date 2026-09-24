/// Cheap, zero-model-call pattern matching that decides whether a message
/// is worth embedding into persistent memory or worth searching memory for
/// at all — the whole point being that neither decision costs a chat-model
/// generation. An embedding call is milliseconds on-device; a chat
/// generation is seconds. Every "should we bother" question here is
/// answered by these patterns, not by asking the model.
class MemoryHeuristics {
  MemoryHeuristics._();

  static final List<RegExp> _memorablePatterns = [
    RegExp(r'\bmy name is\b', caseSensitive: false),
    RegExp(r"\bi'?m (?:called|named)\b", caseSensitive: false),
    RegExp(r'\bcall me\b', caseSensitive: false),
    RegExp(r'\bi (?:am|work as|work at|study at)\b', caseSensitive: false),
    RegExp(r'\bi (?:live|reside) in\b', caseSensitive: false),
    RegExp(r'\bi (?:like|love|prefer|hate|dislike|enjoy)\b', caseSensitive: false),
    RegExp(r"\bmy (?:favorite|birthday|age|job|goal|name)\b", caseSensitive: false),
    RegExp(r"\bi'?ve (?:always|never)\b", caseSensitive: false),
    RegExp(r'\bremember (?:that|this|it)\b', caseSensitive: false),
    RegExp(r"\bdon'?t forget\b", caseSensitive: false),
    RegExp(r'\bi (?:have|own|drive|use)\b.{0,40}\b(?:allerg|car|dog|cat|kids?|children)\b',
        caseSensitive: false),
  ];

  static const _trivialExact = {
    'ok', 'okay', 'k', 'kk', 'thanks', 'thank you', 'thx', 'ty', 'lol',
    'lmao', 'yes', 'no', 'yep', 'yeah', 'nope', 'sure', 'hi', 'hello',
    'hey', 'cool', 'nice', 'great', 'bye', 'goodbye', 'good', 'fine',
  };

  /// Whether [text] looks like a durable, memorable statement worth
  /// storing — a first-person declarative about identity, preferences, or
  /// something explicitly flagged with "remember"/"don't forget". A single
  /// regex pass, no model involved.
  static bool looksMemorable(String text) {
    final trimmed = text.trim();
    if (trimmed.length < 8 || trimmed.length > 500) return false;
    return _memorablePatterns.any((p) => p.hasMatch(trimmed));
  }

  /// Whether [text] is too trivial to bother embedding for retrieval at
  /// all — short acknowledgements, greetings, single-word replies. Skipping
  /// these avoids an embedding call and a memory search on the majority of
  /// turns in a normal chat.
  static bool looksTrivial(String text) {
    final trimmed = text.trim().toLowerCase();
    if (trimmed.isEmpty) return true;
    if (trimmed.length <= 3) return true;
    return _trivialExact.contains(trimmed);
  }

  static final RegExp _noneWord = RegExp(r'\b(none|nothing|n/a|nil)\b', caseSensitive: false);

  /// Whether an LLM's extraction response amounts to "nothing worth
  /// remembering" — deliberately more forgiving than an exact `== 'NONE'`
  /// check, since a small on-device model asked to "respond with exactly
  /// NONE" often instead hedges with a full sentence around it, e.g. "I
  /// don't think there's anything notable to remember here, so I'll say
  /// NONE." Gated on word count (at most 20 — generous enough to cover that
  /// kind of hedge, but a real extracted memory fact is essentially never
  /// this wordy) precisely so a genuine short memory that happens to
  /// contain the word "none" — e.g. "User has none of the common
  /// allergies" (7 words) — is still comfortably under the cap and kept;
  /// only responses that are ALSO mostly about there being nothing to
  /// remember, not just short, get discarded.
  static bool isNoMemorySentinel(String text) {
    final trimmed = text.trim();
    if (trimmed.isEmpty) return true;
    final words = trimmed.split(RegExp(r'\s+'));
    if (words.length > 20) return false;
    return _noneWord.hasMatch(trimmed);
  }
}
