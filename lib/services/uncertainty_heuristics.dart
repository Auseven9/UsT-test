/// Cheap, zero-model-call estimate of how hedged a piece of text reads —
/// the closest thing to a confidence signal available here. llamadart
/// exposes no per-token logprobs (checked directly in its engine API), so
/// there's no real entropy measurement to draw on; this is surface language
/// only, same limitation as any human guessing at a writer's confidence
/// from word choice alone. Mirrors `MemoryHeuristics`' style: a fixed
/// pattern list, no inference, cheap enough to run on every turn.
class UncertaintyHeuristics {
  UncertaintyHeuristics._();

  static final List<RegExp> _hedgePatterns = [
    RegExp(r"\bi(?:'m| am) not (?:entirely |completely |100% )?sure\b", caseSensitive: false),
    RegExp(r'\bi (?:think|believe|guess|suspect)\b', caseSensitive: false),
    RegExp(r'\b(?:probably|possibly|perhaps|maybe|presumably)\b', caseSensitive: false),
    RegExp(r'\bmight (?:be|have|not)\b', caseSensitive: false),
    RegExp(r'\bcould (?:be|have been)\b', caseSensitive: false),
    RegExp(r"\b(?:i'?m|it'?s) (?:uncertain|unclear|unsure)\b", caseSensitive: false),
    RegExp(r'\bas far as i (?:know|can tell)\b', caseSensitive: false),
    RegExp(r"\bi (?:don'?t|do not) (?:know|recall) for (?:certain|sure)\b", caseSensitive: false),
    RegExp(r'\b(?:it depends|hard to say|difficult to say)\b', caseSensitive: false),
    RegExp(r"\bi (?:could|may) be wrong\b", caseSensitive: false),
    RegExp(r"\bdon'?t (?:quote|hold) me on\b", caseSensitive: false),
  ];

  /// A rough 0.0 (reads confident) to 1.0 (reads heavily hedged) density
  /// score. One hedge barely moves it; several in a short answer push it
  /// high. Not a calibrated probability — a relative signal for the UI and
  /// telemetry, nothing more.
  static double hedgeScore(String text) {
    if (text.trim().isEmpty) return 0.0;
    final matchCount =
        _hedgePatterns.fold<int>(0, (sum, pattern) => sum + pattern.allMatches(text).length);
    if (matchCount == 0) return 0.0;
    // 4+ hedges in one answer saturates the score — beyond that it's not
    // "somewhat cautious" anymore, it's an answer that's mostly hedging.
    return (matchCount / 4.0).clamp(0.0, 1.0);
  }
}
