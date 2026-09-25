import 'dart:async';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../theme/app_colors.dart';
import '../models/tutorial_step.dart';
import '../services/tutorial_service.dart';

/// Wraps any widget so the guided tour can find and spotlight it later.
/// Registers its `GlobalKey` under [id] on mount, unregisters on dispose —
/// `TutorialOverlay` looks keys up by id rather than holding them directly,
/// since the actual widget instance for a given id can be unmounted and
/// remounted (switching tabs, scrolling a list back into view) while the
/// tour is mid-run.
class TutorialTarget extends StatefulWidget {
  final String id;
  final Widget child;
  const TutorialTarget({super.key, required this.id, required this.child});

  @override
  State<TutorialTarget> createState() => _TutorialTargetState();
}

class _TutorialTargetState extends State<TutorialTarget> {
  final GlobalKey _key = GlobalKey();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) Get.find<TutorialService>().registerTarget(widget.id, _key);
    });
  }

  @override
  void dispose() {
    Get.find<TutorialService>().unregisterTarget(widget.id, _key);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => KeyedSubtree(key: _key, child: widget.child);
}

/// A small "feeds the model" marker for controls whose value becomes part
/// of what the model actually sees — the system prompt, the chat input,
/// tool schemas, memory context, reasoning recall — as opposed to a plain
/// app-behavior toggle that never reaches the model itself.
class ModelInjectionBadge extends StatelessWidget {
  const ModelInjectionBadge({super.key});

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: 'This becomes part of what the model actually sees, not '
          'just an app setting.',
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
        decoration: BoxDecoration(
          color: AppColors.accentHi.withValues(alpha: 0.14),
          borderRadius: BorderRadius.circular(5),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.arrow_upward_rounded, size: 10, color: AppColors.accentHi),
            const SizedBox(width: 2),
            Text(
              'feeds the model',
              style: TextStyle(
                fontSize: 9,
                color: AppColors.accentHi,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Full-screen coach-mark overlay for the guided tour — dims everything
/// except a spotlighted cutout around the current step's registered
/// target, with a description card and Back/Next/Skip controls. Read-only
/// by design (it absorbs all touches itself rather than passing taps
/// through to the spotlighted widget) — steps are informational, advanced
/// only via the card's own buttons, never by interacting with the app
/// underneath while the tour is active.
class TutorialOverlay extends StatefulWidget {
  const TutorialOverlay({super.key});

  @override
  State<TutorialOverlay> createState() => _TutorialOverlayState();
}

class _TutorialOverlayState extends State<TutorialOverlay> {
  final _tutorial = Get.find<TutorialService>();
  Timer? _retryTimer;

  @override
  void dispose() {
    _retryTimer?.cancel();
    super.dispose();
  }

  Rect? _highlightRect() {
    final step = _tutorial.currentStep;
    if (step == null) return null;
    final key = _tutorial.targetKey(step.targetId);
    final renderObject = key?.currentContext?.findRenderObject();
    if (renderObject is! RenderBox || !renderObject.attached) return null;
    final topLeft = renderObject.localToGlobal(Offset.zero);
    const pad = 6.0;
    return Rect.fromLTWH(
      topLeft.dx - pad,
      topLeft.dy - pad,
      renderObject.size.width + pad * 2,
      renderObject.size.height + pad * 2,
    );
  }

  /// The tab switch a step requests re-renders a different screen, which
  /// takes a frame (or a few, on a slow device) before the new target is
  /// actually mounted and measurable — this keeps retrying briefly rather
  /// than giving up on the first miss and falling back to an unspotlit
  /// centered card for a step that really does have a target.
  void _scheduleRetryIfMissing() {
    _retryTimer?.cancel();
    if (_highlightRect() != null) return;
    _retryTimer = Timer(const Duration(milliseconds: 120), () {
      if (mounted) setState(() {});
    });
  }

  @override
  Widget build(BuildContext context) {
    return Obx(() {
      if (!_tutorial.isActive.value) return const SizedBox.shrink();
      final step = _tutorial.currentStep;
      if (step == null) return const SizedBox.shrink();

      final highlight = _highlightRect();
      if (highlight == null) _scheduleRetryIfMissing();

      final screenSize = MediaQuery.of(context).size;
      final safeTop = MediaQuery.of(context).padding.top;
      final safeBottom = MediaQuery.of(context).padding.bottom;

      return Positioned.fill(
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTap: () {}, // absorb taps — see class doc
          child: Stack(
            children: [
              Positioned.fill(
                child: CustomPaint(painter: _SpotlightPainter(highlight)),
              ),
              _CardPlacement(
                highlight: highlight,
                screenSize: screenSize,
                safeTop: safeTop,
                safeBottom: safeBottom,
                child: _TutorialCard(
                  step: step,
                  index: _tutorial.stepIndex.value,
                  total: _tutorial.steps.length,
                  onNext: _tutorial.next,
                  onBack: _tutorial.stepIndex.value > 0 ? _tutorial.back : null,
                  onSkip: _tutorial.skip,
                ),
              ),
            ],
          ),
        ),
      );
    });
  }
}

class _SpotlightPainter extends CustomPainter {
  final Rect? highlight;
  const _SpotlightPainter(this.highlight);

  @override
  void paint(Canvas canvas, Size size) {
    final scrimPaint = Paint()..color = Colors.black.withValues(alpha: 0.72);
    final fullPath = Path()..addRect(Offset.zero & size);
    if (highlight == null) {
      canvas.drawPath(fullPath, scrimPaint);
      return;
    }
    final cutoutRRect = RRect.fromRectAndRadius(highlight!, const Radius.circular(12));
    final cutoutPath = Path()..addRRect(cutoutRRect);
    final combined = Path.combine(PathOperation.difference, fullPath, cutoutPath);
    canvas.drawPath(combined, scrimPaint);
    canvas.drawRRect(
      cutoutRRect,
      Paint()
        ..color = AppColors.accent
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5,
    );
  }

  @override
  bool shouldRepaint(covariant _SpotlightPainter oldDelegate) =>
      oldDelegate.highlight != highlight;
}

/// Positions [child] below the highlight if there's room, else above it,
/// else centered on screen when there's no highlight at all.
class _CardPlacement extends StatelessWidget {
  final Rect? highlight;
  final Size screenSize;
  final double safeTop;
  final double safeBottom;
  final Widget child;

  const _CardPlacement({
    required this.highlight,
    required this.screenSize,
    required this.safeTop,
    required this.safeBottom,
    required this.child,
  });

  static const _cardMargin = 16.0;
  static const _cardGap = 12.0;
  static const _estimatedCardHeight = 220.0;

  @override
  Widget build(BuildContext context) {
    if (highlight == null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: _cardMargin),
          child: child,
        ),
      );
    }

    final spaceBelow = screenSize.height - safeBottom - highlight!.bottom;
    final spaceAbove = highlight!.top - safeTop;
    final placeBelow = spaceBelow >= _estimatedCardHeight || spaceBelow >= spaceAbove;

    return Positioned(
      left: _cardMargin,
      right: _cardMargin,
      top: placeBelow ? highlight!.bottom + _cardGap : null,
      bottom: placeBelow ? null : (screenSize.height - highlight!.top + _cardGap),
      child: child,
    );
  }
}

class _TutorialCard extends StatelessWidget {
  final TutorialStep step;
  final int index;
  final int total;
  final VoidCallback onNext;
  final VoidCallback? onBack;
  final VoidCallback onSkip;

  const _TutorialCard({
    required this.step,
    required this.index,
    required this.total,
    required this.onNext,
    required this.onBack,
    required this.onSkip,
  });

  @override
  Widget build(BuildContext context) {
    final isLast = index >= total - 1;
    return Material(
      color: Colors.transparent,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: context.bgPanel,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.accent.withValues(alpha: 0.4)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.35),
              blurRadius: 20,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    step.title,
                    style: TextStyle(
                      color: context.text,
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
                Text(
                  '${index + 1} / $total',
                  style: TextStyle(color: context.textD, fontSize: 11),
                ),
              ],
            ),
            if (step.injectsIntoModel) ...[
              const SizedBox(height: 6),
              const ModelInjectionBadge(),
            ],
            const SizedBox(height: 8),
            Text(
              step.description,
              style: TextStyle(color: context.textM, fontSize: 13, height: 1.4),
            ),
            const SizedBox(height: 14),
            Row(
              children: [
                TextButton(
                  onPressed: onSkip,
                  child: Text('Skip Tutorial', style: TextStyle(color: context.textD)),
                ),
                const Spacer(),
                if (onBack != null)
                  TextButton(
                    onPressed: onBack,
                    child: const Text('Back'),
                  ),
                const SizedBox(width: 4),
                ElevatedButton(
                  onPressed: onNext,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.accent,
                    foregroundColor: Colors.white,
                    elevation: 0,
                  ),
                  child: Text(isLast ? 'Done' : 'Next'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
