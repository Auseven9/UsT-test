/// One stop in the guided tour — spotlights whatever widget registered
/// itself under [targetId] (via `TutorialTarget`) and shows [title]/
/// [description] next to it. [injectsIntoModel] marks a step as covering a
/// "model injection area" — a control whose value becomes part of what the
/// model actually sees, not just an app-behavior toggle — so the overlay
/// can flag that distinction explicitly instead of leaving it implicit.
class TutorialStep {
  final String id;
  final String targetId;
  final String title;
  final String description;

  /// Which mobile bottom-nav tab (0=Chat, 1=Models, 2=Settings) must be
  /// showing for this step's target to exist on screen. Null means don't
  /// change tabs — the target is already visible, or this is a desktop
  /// layout where all three panes can be visible at once.
  final int? requiredTab;

  final bool injectsIntoModel;

  const TutorialStep({
    required this.id,
    required this.targetId,
    required this.title,
    required this.description,
    this.requiredTab,
    this.injectsIntoModel = false,
  });
}
