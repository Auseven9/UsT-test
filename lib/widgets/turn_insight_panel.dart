import 'package:flutter/material.dart';

import '../theme/app_colors.dart';
import '../models/turn_telemetry.dart';

/// Collapsible panel showing what actually happened while a turn generated
/// — real per-token cadence, real context-window usage, real tool-call
/// events, real per-round token counts, and (once background extraction
/// finishes) the real category/valence of whatever memory it captured.
/// Every number here traces back to something the app was already
/// computing to run generation; nothing is synthesized to look
/// impressive. Updates live via [telemetry] (a [ChangeNotifier]) while a
/// message streams in, same pattern as the Thoughts panel next to it.
class TurnInsightPanel extends StatefulWidget {
  final TurnTelemetry telemetry;
  const TurnInsightPanel({super.key, required this.telemetry});

  @override
  State<TurnInsightPanel> createState() => _TurnInsightPanelState();
}

class _TurnInsightPanelState extends State<TurnInsightPanel> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: widget.telemetry,
      builder: (context, _) {
        if (!widget.telemetry.hasAnyData) return const SizedBox.shrink();
        return Container(
          margin: const EdgeInsets.only(bottom: 10),
          decoration: BoxDecoration(
            color: context.bgHover.withValues(alpha: 0.5),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: context.borderFaint),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              InkWell(
                onTap: () => setState(() => _expanded = !_expanded),
                borderRadius: BorderRadius.circular(8),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                  child: Row(
                    children: [
                      Icon(
                        _expanded ? Icons.expand_less : Icons.expand_more,
                        size: 16,
                        color: context.textD,
                      ),
                      const SizedBox(width: 6),
                      Icon(Icons.insights_rounded, size: 14, color: context.textD),
                      const SizedBox(width: 6),
                      Text(
                        'Insight',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: context.textD,
                        ),
                      ),
                      if (widget.telemetry.extractedValence != null) ...[
                        const SizedBox(width: 8),
                        _ValenceChip(
                          category: widget.telemetry.extractedCategory!,
                          valence: widget.telemetry.extractedValence!,
                        ),
                      ],
                      // 0.5 means over half diverged from the model's top
                      // choice (real signal) or two-plus hedge phrases
                      // (heuristic fallback — hedgeScore is matchCount/4.0,
                      // so a single hedge alone scores 0.25 and shouldn't
                      // flag on its own).
                      if ((widget.telemetry.uncertainty ?? 0) >= 0.5) ...[
                        const SizedBox(width: 6),
                        _UncertaintyChip(
                          score: widget.telemetry.uncertainty!,
                          isHeuristic: widget.telemetry.uncertaintyIsHeuristic,
                        ),
                      ],
                      if (widget.telemetry.critiqueFlagged == true) ...[
                        const SizedBox(width: 6),
                        const _CritiqueChip(),
                      ],
                    ],
                  ),
                ),
              ),
              if (_expanded)
                Padding(
                  padding: const EdgeInsets.fromLTRB(10, 0, 10, 10),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (widget.telemetry.contextTotalTokens > 0) ...[
                        _ContextGauge(
                          used: widget.telemetry.contextUsedTokens,
                          total: widget.telemetry.contextTotalTokens,
                        ),
                        const SizedBox(height: 10),
                      ],
                      if (widget.telemetry.cadence.isNotEmpty) ...[
                        Text(
                          'Generation cadence',
                          style: TextStyle(fontSize: 10, color: context.textD),
                        ),
                        const SizedBox(height: 4),
                        SizedBox(
                          height: 36,
                          width: double.infinity,
                          child: CustomPaint(
                            painter: _CadencePainter(
                              samples: widget.telemetry.cadence,
                              thinkingColor: AppColors.accentHi,
                              contentColor: AppColors.green,
                            ),
                          ),
                        ),
                        const SizedBox(height: 10),
                      ],
                      if (widget.telemetry.roundTokenCounts.isNotEmpty) ...[
                        Text(
                          '${widget.telemetry.roundTokenCounts.length} '
                          '${widget.telemetry.roundTokenCounts.length == 1 ? "round" : "rounds"}',
                          style: TextStyle(fontSize: 10, color: context.textD),
                        ),
                        const SizedBox(height: 4),
                        SizedBox(
                          height: 28,
                          child: CustomPaint(
                            painter: _RoundStepPainter(
                              rounds: widget.telemetry.roundTokenCounts,
                              color: AppColors.accent,
                            ),
                          ),
                        ),
                        const SizedBox(height: 10),
                      ],
                      if (widget.telemetry.toolCalls.isNotEmpty) ...[
                        Wrap(
                          spacing: 6,
                          runSpacing: 6,
                          children: widget.telemetry.toolCalls
                              .map((t) => _ToolChip(event: t))
                              .toList(),
                        ),
                        const SizedBox(height: 10),
                      ],
                      if (widget.telemetry.critiqueRan &&
                          widget.telemetry.critiqueNote != null) ...[
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Icon(
                              widget.telemetry.critiqueFlagged == true
                                  ? Icons.flag_rounded
                                  : Icons.check_circle_outline_rounded,
                              size: 12,
                              color: widget.telemetry.critiqueFlagged == true
                                  ? AppColors.orange
                                  : context.textD,
                            ),
                            const SizedBox(width: 4),
                            Expanded(
                              child: Text(
                                'Self-check: ${widget.telemetry.critiqueNote}',
                                style: TextStyle(fontSize: 10, color: context.textD),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ],
                  ),
                ),
            ],
          ),
        );
      },
    );
  }
}

/// Thin horizontal fill bar — real fraction of the context window actually
/// used by history for this turn, out of the model's real n_ctx.
class _ContextGauge extends StatelessWidget {
  final int used;
  final int total;
  const _ContextGauge({required this.used, required this.total});

  @override
  Widget build(BuildContext context) {
    final fraction = total > 0 ? (used / total).clamp(0.0, 1.0) : 0.0;
    final overHalf = fraction > 0.5;
    final color = fraction > 0.85
        ? AppColors.red
        : overHalf
            ? AppColors.orange
            : AppColors.green;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text('Context used', style: TextStyle(fontSize: 10, color: context.textD)),
            Text(
              '$used / $total tokens',
              style: TextStyle(fontSize: 10, color: context.textD),
            ),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(3),
          child: LayoutBuilder(
            builder: (context, constraints) => Stack(
              children: [
                Container(height: 6, color: context.borderFaint),
                Container(
                  height: 6,
                  width: constraints.maxWidth * fraction,
                  color: color,
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

/// Vertical-bar strip, one bar per real chunk-arrival sample — height
/// inversely tracks the real interval since the previous chunk (faster
/// arrival = taller bar, reading as more "activity"), color splits
/// thinking-channel samples from answer-channel ones. Not a literal audio
/// spectrogram — there's no frequency-domain analysis here — just an
/// honest scrolling strip of real generation timing, styled the same way.
class _CadencePainter extends CustomPainter {
  final List<CadenceSample> samples;
  final Color thinkingColor;
  final Color contentColor;
  const _CadencePainter({
    required this.samples,
    required this.thinkingColor,
    required this.contentColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (samples.isEmpty) return;
    final barWidth = size.width / samples.length;
    // Clamp so one long pause doesn't flatten every other bar to nothing,
    // and one instant back-to-back pair doesn't blow past the strip.
    const minIntervalMs = 5.0;
    const maxIntervalMs = 400.0;
    for (var i = 0; i < samples.length; i++) {
      final s = samples[i];
      final clamped = s.intervalMs.clamp(minIntervalMs, maxIntervalMs);
      final normalized = 1.0 - ((clamped - minIntervalMs) / (maxIntervalMs - minIntervalMs));
      final barHeight = (size.height * 0.15) + (size.height * 0.85 * normalized);
      final paint = Paint()
        ..color = (s.isThinking ? thinkingColor : contentColor).withValues(alpha: 0.85);
      final x = i * barWidth;
      canvas.drawRect(
        Rect.fromLTWH(x, size.height - barHeight, (barWidth - 1).clamp(0.5, barWidth), barHeight),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _CadencePainter oldDelegate) =>
      oldDelegate.samples.length != samples.length;
}

/// One bar per real tool-calling round, height proportional to that
/// round's real token count (from the same counter the engine itself
/// reports generation speed from).
class _RoundStepPainter extends CustomPainter {
  final List<int> rounds;
  final Color color;
  const _RoundStepPainter({required this.rounds, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    if (rounds.isEmpty) return;
    final maxCount = rounds.reduce((a, b) => a > b ? a : b).clamp(1, 1 << 30);
    final gap = 4.0;
    final barWidth = (size.width - gap * (rounds.length - 1)) / rounds.length;
    for (var i = 0; i < rounds.length; i++) {
      final fraction = rounds[i] / maxCount;
      final barHeight = (size.height * 0.15) + (size.height * 0.85 * fraction);
      final x = i * (barWidth + gap);
      final rect = RRect.fromRectAndRadius(
        Rect.fromLTWH(x, size.height - barHeight, barWidth, barHeight),
        const Radius.circular(2),
      );
      canvas.drawRRect(rect, Paint()..color = color.withValues(alpha: 0.8));
    }
  }

  @override
  bool shouldRepaint(covariant _RoundStepPainter oldDelegate) =>
      oldDelegate.rounds.length != rounds.length ||
      (rounds.isNotEmpty && oldDelegate.rounds.last != rounds.last);
}

class _ToolChip extends StatelessWidget {
  final ToolCallEvent event;
  const _ToolChip({required this.event});

  @override
  Widget build(BuildContext context) {
    final color = event.succeeded ? AppColors.green : AppColors.red;
    return Tooltip(
      message: '${event.name} · ${event.time.hour.toString().padLeft(2, '0')}:'
          '${event.time.minute.toString().padLeft(2, '0')}:'
          '${event.time.second.toString().padLeft(2, '0')}',
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              event.succeeded ? Icons.bolt_rounded : Icons.error_outline_rounded,
              size: 11,
              color: color,
            ),
            const SizedBox(width: 4),
            Text(event.name, style: TextStyle(fontSize: 10, color: color)),
          ],
        ),
      ),
    );
  }
}

/// Surfaces `TurnTelemetry.uncertainty` — a heuristic hedge-language
/// density score, not a calibrated probability (see
/// UncertaintyHeuristics' own doc for why nothing more rigorous is
/// available here). Only shown once it clears a small threshold, so an
/// answer with zero or one hedge word doesn't get visually flagged.
class _UncertaintyChip extends StatelessWidget {
  final double score;
  final bool isHeuristic;
  const _UncertaintyChip({required this.score, required this.isHeuristic});

  @override
  Widget build(BuildContext context) {
    final color = score > 0.6 ? AppColors.orange : context.textM;
    final message = isHeuristic
        ? 'Hedge-language density in this answer: '
            '${(score * 100).round()}% (heuristic — the engine couldn\'t '
            'report real per-token confidence for this turn)'
        : 'Model uncertainty: ${(score * 100).round()}% (measured from '
            'real per-token confidence — how far sampling diverged from '
            'the model\'s own top choice, not a correctness score)';
    return Tooltip(
      message: message,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.12),
          borderRadius: BorderRadius.circular(5),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.help_outline_rounded, size: 10, color: color),
            const SizedBox(width: 2),
            Text(
              isHeuristic ? 'hedged' : 'uncertain',
              style: TextStyle(fontSize: 9, color: color, fontWeight: FontWeight.w600),
            ),
          ],
        ),
      ),
    );
  }
}

/// Shown when the self-critique background pass flagged a possible issue
/// with this answer (a contradiction with memory, or an unsupported
/// confident claim) — see ChatController._runSelfCritique.
class _CritiqueChip extends StatelessWidget {
  const _CritiqueChip();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: AppColors.orange.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(5),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.flag_rounded, size: 10, color: AppColors.orange),
          const SizedBox(width: 2),
          Text('self-check', style: TextStyle(fontSize: 9, color: AppColors.orange, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}

/// Discrete 3-state marker — positive/neutral/negative is a categorical
/// judgment from the extraction model, not a continuous measurement, so
/// this snaps between three fixed states rather than implying a smooth
/// gradient of intensity the app doesn't actually have.
class _ValenceChip extends StatelessWidget {
  final String category;
  final String valence;
  const _ValenceChip({required this.category, required this.valence});

  Color _valenceColor(BuildContext context) {
    switch (valence) {
      case 'positive':
        return AppColors.green;
      case 'negative':
        return AppColors.red;
      default:
        return context.textM;
    }
  }

  IconData get _valenceIcon {
    switch (valence) {
      case 'positive':
        return Icons.arrow_upward_rounded;
      case 'negative':
        return Icons.arrow_downward_rounded;
      default:
        return Icons.remove_rounded;
    }
  }

  @override
  Widget build(BuildContext context) {
    final color = _valenceColor(context);
    return Tooltip(
      message: 'Memory captured: $category / $valence',
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.12),
          borderRadius: BorderRadius.circular(5),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(_valenceIcon, size: 10, color: color),
            const SizedBox(width: 2),
            Text(category, style: TextStyle(fontSize: 9, color: color, fontWeight: FontWeight.w600)),
          ],
        ),
      ),
    );
  }
}
