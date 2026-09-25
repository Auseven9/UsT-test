import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../models/tutorial_step.dart';
import 'chat_storage_service.dart';

/// Drives the guided tour: which step is active, which mobile tab it needs,
/// and a registry mapping each step's `targetId` to the real `GlobalKey` of
/// whatever widget is currently showing it on screen. Widgets register
/// themselves (via `TutorialTarget`) rather than the tour holding keys
/// directly, since the actual widget for a given id can be unmounted and
/// remounted as the user switches tabs during the tour.
class TutorialService extends GetxService {
  final ChatStorageService _storage = Get.find<ChatStorageService>();

  final isActive = false.obs;
  final stepIndex = 0.obs;

  /// Set when a step needs a different mobile tab active than whatever is
  /// currently showing — HomeScreen listens to this and switches tabs,
  /// then this is cleared back to null so the same tab can be re-requested
  /// later without relying on Rx's "didn't change" no-op behavior.
  final Rxn<int> requestedTab = Rxn<int>();

  final Map<String, GlobalKey> _targets = {};
  List<TutorialStep> steps = const [];

  bool get hasCompletedOnboarding => _storage.tutorialCompleted;

  TutorialStep? get currentStep =>
      steps.isEmpty ? null : steps[stepIndex.value];

  void registerTarget(String id, GlobalKey key) {
    _targets[id] = key;
  }

  /// Only clears the entry if [key] is still the one registered — a widget
  /// that got rebuilt with a fresh key before disposing the old one must
  /// not clobber the new registration.
  void unregisterTarget(String id, GlobalKey key) {
    if (_targets[id] == key) _targets.remove(id);
  }

  GlobalKey? targetKey(String id) => _targets[id];

  void start(List<TutorialStep> withSteps) {
    if (withSteps.isEmpty) return;
    steps = withSteps;
    stepIndex.value = 0;
    isActive.value = true;
    _applyStepTab();
  }

  /// Starts the tour again regardless of whether it was already completed
  /// or skipped — used by the "Replay Tutorial" entry in Settings. Doesn't
  /// touch the completed flag until this run itself finishes or is
  /// skipped, same as any other run.
  void replay(List<TutorialStep> withSteps) => start(withSteps);

  void _applyStepTab() {
    final tab = currentStep?.requiredTab;
    if (tab != null) requestedTab.value = tab;
  }

  void next() {
    if (stepIndex.value < steps.length - 1) {
      stepIndex.value++;
      _applyStepTab();
    } else {
      complete();
    }
  }

  void back() {
    if (stepIndex.value > 0) {
      stepIndex.value--;
      _applyStepTab();
    }
  }

  void skip() => complete();

  void complete() {
    isActive.value = false;
    _storage.tutorialCompleted = true;
  }
}
