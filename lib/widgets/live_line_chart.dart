import 'package:flutter/material.dart';

import '../theme/app_colors.dart';

/// A small, single-series live line chart — a rolling window of recent
/// values with a direct-labeled current reading. Deliberately never a
/// dual-axis chart: each metric gets its own instance of this widget rather
/// than sharing a scale with another series.
class LiveLineChart extends StatelessWidget {
  final String title;
  final List<double?> values; // oldest first; null = no reading that sample
  final Color color;
  final String Function(double) formatValue;
  final double? fixedMax; // e.g. 100 for a percentage; null = auto-scale
  final double height;

  const LiveLineChart({
    super.key,
    required this.title,
    required this.values,
    required this.color,
    required this.formatValue,
    this.fixedMax,
    this.height = 90,
  });

  @override
  Widget build(BuildContext context) {
    final latest = values.isNotEmpty ? values.last : null;

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: context.bgPanel,
        border: Border.all(color: context.border),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 8,
                height: 8,
                decoration: BoxDecoration(color: color, shape: BoxShape.circle),
              ),
              const SizedBox(width: 8),
              Text(
                title,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: context.text,
                ),
              ),
              const Spacer(),
              Text(
                latest != null ? formatValue(latest) : '—',
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: latest != null ? color : context.textD,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          SizedBox(
            height: height,
            width: double.infinity,
            child: CustomPaint(
              painter: _LineChartPainter(
                values: values,
                color: color,
                fixedMax: fixedMax,
                gridColor: context.borderFaint,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _LineChartPainter extends CustomPainter {
  final List<double?> values;
  final Color color;
  final double? fixedMax;
  final Color gridColor;

  _LineChartPainter({
    required this.values,
    required this.color,
    required this.fixedMax,
    required this.gridColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Baseline gridline — recessive, per the "recessive grid/axes" guidance.
    final gridPaint = Paint()
      ..color = gridColor
      ..strokeWidth = 1;
    canvas.drawLine(
      Offset(0, size.height - 1),
      Offset(size.width, size.height - 1),
      gridPaint,
    );

    final known = values.whereType<double>().toList();
    if (known.isEmpty) return;

    final max = fixedMax ?? (known.reduce((a, b) => a > b ? a : b) * 1.2 + 0.001);
    final min = 0.0;
    final range = (max - min) == 0 ? 1.0 : (max - min);

    final path = Path();
    final stepX = values.length > 1 ? size.width / (values.length - 1) : 0.0;
    var started = false;
    Offset? lastPoint;

    for (var i = 0; i < values.length; i++) {
      final v = values[i];
      if (v == null) {
        started = false; // gap in data — don't connect across it
        continue;
      }
      final x = stepX * i;
      final normalized = ((v - min) / range).clamp(0.0, 1.0);
      final y = size.height - (normalized * (size.height - 4)) - 2;

      if (!started) {
        path.moveTo(x, y);
        started = true;
      } else {
        path.lineTo(x, y);
      }
      lastPoint = Offset(x, y);
    }

    final linePaint = Paint()
      ..color = color
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    canvas.drawPath(path, linePaint);

    // Direct-label the current point with a small marker (>= 8px per spec).
    if (lastPoint != null) {
      canvas.drawCircle(lastPoint, 4, Paint()..color = color);
    }
  }

  @override
  bool shouldRepaint(covariant _LineChartPainter oldDelegate) {
    return oldDelegate.values != values || oldDelegate.color != color;
  }
}
