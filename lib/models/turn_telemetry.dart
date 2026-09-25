import 'package:flutter/foundation.dart';

/// One real sample of the token stream as it arrived during generation —
/// actual wall-clock spacing between chunks, tagged by which channel
/// (thinking vs answer) it belonged to. Every field here comes from
/// something the app was already computing to run generation; nothing is
/// synthesized or fetched via extra inference just to have something to
/// show.
class CadenceSample {
  final bool isThinking;
  final double intervalMs;
  const CadenceSample({required this.isThinking, required this.intervalMs});
}

/// A real tool call firing during this turn, with its actual timestamp and
/// outcome.
class ToolCallEvent {
  final String name;
  final DateTime time;
  final bool succeeded;
  const ToolCallEvent({
    required this.name,
    required this.time,
    required this.succeeded,
  });
}

/// Live visualization data collected while a single assistant turn
/// generates — real per-token cadence, real context-window usage, real
/// tool-call events, real per-round token counts, and (once background
/// extraction finishes) the real category/valence of whatever memory it
/// captured, if anything. Attached to a [MessageModel] as a deliberately
/// transient field, never a `@HiveField` — this is ephemeral, in-session
/// telemetry for the chat UI, not part of the persisted chat record, so it
/// is simply absent after a reload rather than bloating chat storage with
/// per-message instrumentation data.
///
/// Extends [ChangeNotifier] so the widget showing it can update live as
/// generation streams in, without needing a GetX controller of its own for
/// what's fundamentally per-message, short-lived state.
class TurnTelemetry extends ChangeNotifier {
  static const _maxCadenceSamples = 240;

  final List<CadenceSample> cadence = [];
  final List<ToolCallEvent> toolCalls = [];
  final List<int> roundTokenCounts = [];
  int contextUsedTokens = 0;
  int contextTotalTokens = 0;
  String? extractedCategory;
  String? extractedValence;

  /// 0.0 (confident) to 1.0 (uncertain), set once generation finishes.
  /// When [uncertaintyIsHeuristic] is false, this is a real measurement —
  /// 1.0 minus the engine's own average per-token confidence (how much
  /// probability mass the model put on the tokens it actually sampled).
  /// When true, it's a fallback: surface-language hedge-word density from
  /// [UncertaintyHeuristics.hedgeScore], used only when the engine can't
  /// report the real signal (older llamadart, no model loaded).
  double? uncertainty;

  /// Whether [uncertainty] is the surface-language fallback rather than a
  /// real per-token confidence measurement — see [uncertainty]'s own doc.
  bool uncertaintyIsHeuristic = false;

  /// Whether the self-critique pass (a bounded second generation — see
  /// ChatController._runSelfCritique) flagged a possible issue with this
  /// answer, and its note if so. Null means the pass never ran (no helper
  /// or main model available, or the feature is toggled off), not "ran and
  /// found nothing" — see [critiqueRan] for that distinction.
  bool critiqueRan = false;
  bool? critiqueFlagged;
  String? critiqueNote;

  void addCadenceSample(bool isThinking, double intervalMs) {
    cadence.add(CadenceSample(isThinking: isThinking, intervalMs: intervalMs));
    if (cadence.length > _maxCadenceSamples) cadence.removeAt(0);
    notifyListeners();
  }

  void addToolCall(String name, {required bool succeeded}) {
    toolCalls.add(ToolCallEvent(name: name, time: DateTime.now(), succeeded: succeeded));
    notifyListeners();
  }

  void addRoundTokens(int count) {
    roundTokenCounts.add(count);
    notifyListeners();
  }

  void setContextUsage(int used, int total) {
    contextUsedTokens = used;
    contextTotalTokens = total;
    notifyListeners();
  }

  void setExtractedValence(String category, String valence) {
    extractedCategory = category;
    extractedValence = valence;
    notifyListeners();
  }

  void setUncertainty(double value, {required bool isHeuristic}) {
    uncertainty = value;
    uncertaintyIsHeuristic = isHeuristic;
    notifyListeners();
  }

  void setCritique({required bool flagged, String? note}) {
    critiqueRan = true;
    critiqueFlagged = flagged;
    critiqueNote = note;
    notifyListeners();
  }

  /// Whether there's anything at all worth rendering a panel for — an old
  /// message reloaded from disk has no telemetry object at all (handled by
  /// the caller checking for null), but a freshly-created one could still
  /// be empty for a beat before the first chunk arrives.
  bool get hasAnyData =>
      cadence.isNotEmpty ||
      toolCalls.isNotEmpty ||
      roundTokenCounts.isNotEmpty ||
      contextTotalTokens > 0;
}
