import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../theme/app_colors.dart';
import '../models/memory_entry.dart';
import '../services/memory_service.dart';

const _kGraphCanvasSize = 1200.0;

/// Standard force-directed placement (Fruchterman-Reingold-style: nodes
/// repel each other, real edges pull linked nodes together), run over the
/// real link structure purely so it's legible on screen. Top-level (not a
/// method) so it can run via [compute] on a background isolate — the
/// 220-iteration O(n²) repulsion pass would otherwise block the UI thread
/// during build() as the memory store grows.
Map<String, Offset> _computeGraphLayout(List<MemoryEntry> entries) {
  final rnd = Random(7);
  final positions = <String, Offset>{
    for (final e in entries)
      e.id: Offset(
        rnd.nextDouble() * _kGraphCanvasSize,
        rnd.nextDouble() * _kGraphCanvasSize,
      ),
  };
  if (entries.length < 2) return positions;

  final byId = {for (final e in entries) e.id: e};
  const k = 90.0; // ideal edge length
  var temperature = _kGraphCanvasSize / 10;

  for (var iter = 0; iter < 220; iter++) {
    final disp = <String, Offset>{for (final e in entries) e.id: Offset.zero};

    // Repulsion between every pair — keeps unrelated nodes from piling up.
    for (var i = 0; i < entries.length; i++) {
      for (var j = i + 1; j < entries.length; j++) {
        final a = entries[i].id, b = entries[j].id;
        var delta = positions[a]! - positions[b]!;
        var dist = delta.distance;
        if (dist < 0.01) {
          delta = Offset(rnd.nextDouble() - 0.5, rnd.nextDouble() - 0.5);
          dist = 0.01;
        }
        final force = (k * k) / dist;
        final dir = delta / dist;
        disp[a] = disp[a]! + dir * force;
        disp[b] = disp[b]! - dir * force;
      }
    }

    // Attraction along real links only.
    for (final e in entries) {
      for (final linkedId in e.linkedIds) {
        if (!byId.containsKey(linkedId)) continue;
        final a = e.id, b = linkedId;
        var delta = positions[a]! - positions[b]!;
        final dist = delta.distance.clamp(0.01, double.infinity);
        final force = (dist * dist) / k;
        final dir = delta / dist;
        disp[a] = disp[a]! - dir * force;
        disp[b] = disp[b]! + dir * force;
      }
    }

    for (final e in entries) {
      final d = disp[e.id]!;
      final dist = d.distance.clamp(0.01, double.infinity);
      final capped = d / dist * min(dist, temperature);
      final next = positions[e.id]! + capped;
      positions[e.id] = Offset(
        next.dx.clamp(40.0, _kGraphCanvasSize - 40.0),
        next.dy.clamp(40.0, _kGraphCanvasSize - 40.0),
      );
    }
    temperature *= 0.97;
  }
  return positions;
}

/// A node-and-edge rendering of the actual associative memory graph — every
/// node is a real stored memory, every edge a real link MemoryService wrote
/// at write time, node size a real access count, color a real category,
/// opacity a real active/superseded state. The layout (where each node
/// sits) is the one thing that isn't literally stored data — it's a
/// standard force-directed placement (Fruchterman-Reingold-style: nodes
/// repel each other, real edges pull linked nodes together) computed once
/// over the real link structure, purely so the real structure is legible
/// on screen rather than a list of ids.
class MemoryGraphScreen extends StatefulWidget {
  const MemoryGraphScreen({super.key});

  @override
  State<MemoryGraphScreen> createState() => _MemoryGraphScreenState();
}

class _MemoryGraphScreenState extends State<MemoryGraphScreen> {
  Map<String, Offset> _positions = {};
  int _lastLayoutSignature = -1;
  bool _computingLayout = false;
  String? _selectedId;

  /// Fire-and-forget: kicks off the layout computation on a background
  /// isolate (see [_computeGraphLayout]) if the entry set actually
  /// changed and nothing is already computing. Safe to call from inside
  /// `build()` — it only mutates state synchronously (readable in this
  /// same build pass, for the loading indicator) and the actual `setState`
  /// happens later, in the compute() callback, never during build itself.
  void _maybeStartLayout(List<MemoryEntry> entries) {
    // Recompute only when the underlying set of entries actually changed —
    // a signature over ids+link-counts is cheap and avoids re-running the
    // layout (and losing the user's current pan/zoom orientation) on every
    // unrelated rebuild.
    var signature = entries.length;
    for (final e in entries) {
      signature = signature * 31 + e.id.hashCode + e.linkedIds.length;
    }
    if (signature == _lastLayoutSignature || _computingLayout) return;
    _lastLayoutSignature = signature;
    _computingLayout = true;
    // Runs on a separate isolate — the 220-iteration O(n²) force-directed
    // layout would otherwise block the UI thread synchronously during
    // build(), visibly freezing the app as the memory store grows (memories
    // are never deleted, only superseded, so this only ever grows over the
    // app's lifetime).
    compute(_computeGraphLayout, entries).then((result) {
      if (!mounted) return;
      setState(() {
        _positions = result;
        _computingLayout = false;
      });
    });
  }

  Color _categoryColor(String category) {
    switch (category) {
      case 'fact':
        return AppColors.accent;
      case 'preference':
        return AppColors.green;
      case 'event':
        return AppColors.orange;
      case 'instruction':
        return AppColors.accentHi;
      case 'summary':
        return AppColors.accentDim;
      default:
        return AppColors.accent;
    }
  }

  void _handleTap(Offset localPosition, List<MemoryEntry> entries) {
    String? closestId;
    var closestDist = double.infinity;
    for (final e in entries) {
      final pos = _positions[e.id];
      if (pos == null) continue;
      final radius = _nodeRadius(e);
      final dist = (pos - localPosition).distance;
      if (dist <= radius + 12 && dist < closestDist) {
        closestDist = dist;
        closestId = e.id;
      }
    }
    setState(() => _selectedId = closestId);
    if (closestId != null) {
      final entry = entries.firstWhere((e) => e.id == closestId);
      _showEntrySheet(entry);
    }
  }

  double _nodeRadius(MemoryEntry e) => 7.0 + sqrt(e.accessCount.toDouble()) * 3.0;

  void _showEntrySheet(MemoryEntry entry) {
    showModalBottomSheet(
      context: context,
      backgroundColor: context.bgPanel,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (sheetContext) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Wrap(
                  spacing: 6,
                  children: [
                    _chip(sheetContext, entry.category, _categoryColor(entry.category)),
                    _chip(
                      sheetContext,
                      entry.valence,
                      entry.valence == 'positive'
                          ? AppColors.green
                          : entry.valence == 'negative'
                              ? AppColors.red
                              : sheetContext.textM,
                    ),
                    if (!entry.isActive) _chip(sheetContext, 'superseded', AppColors.red),
                    for (final tag in entry.tags) _chip(sheetContext, tag, sheetContext.textD),
                  ],
                ),
                const SizedBox(height: 12),
                Text(
                  entry.text,
                  style: TextStyle(
                    fontSize: 15,
                    color: sheetContext.text,
                    height: 1.5,
                    decoration: entry.isActive ? null : TextDecoration.lineThrough,
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  'Recalled ${entry.accessCount}× · ${entry.linkedIds.length} linked · '
                  '${entry.createdAt.toString().split('.').first}',
                  style: TextStyle(fontSize: 11, color: sheetContext.textD),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _chip(BuildContext context, String label, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        label,
        style: TextStyle(fontSize: 10, color: color, fontWeight: FontWeight.w600),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final memory = Get.find<MemoryService>();

    return Scaffold(
      backgroundColor: context.bg,
      body: Column(
        children: [
          Container(
            padding: EdgeInsets.only(top: MediaQuery.of(context).padding.top, left: 4, right: 4),
            decoration: BoxDecoration(
              color: context.bg,
              border: Border(bottom: BorderSide(color: context.border, width: 0.5)),
            ),
            child: SizedBox(
              height: 52,
              child: Row(
                children: [
                  IconButton(
                    icon: Icon(Icons.arrow_back_rounded, color: context.text),
                    onPressed: () => Get.back(),
                  ),
                  Text(
                    'Memory Graph',
                    style: TextStyle(fontSize: 17, fontWeight: FontWeight.w600, color: context.text),
                  ),
                ],
              ),
            ),
          ),
          Expanded(
            child: Obx(() {
              final entries = memory.entries.toList();
              if (entries.isEmpty) {
                return Center(
                  child: Text('Nothing remembered yet.', style: TextStyle(color: context.textD)),
                );
              }
              _maybeStartLayout(entries);
              if (_computingLayout && _positions.isEmpty) {
                return const Center(child: CircularProgressIndicator());
              }
              return InteractiveViewer(
                maxScale: 4,
                minScale: 0.2,
                boundaryMargin: const EdgeInsets.all(200),
                child: GestureDetector(
                  onTapUp: (details) => _handleTap(details.localPosition, entries),
                  child: SizedBox(
                    width: _kGraphCanvasSize,
                    height: _kGraphCanvasSize,
                    child: CustomPaint(
                      painter: _GraphPainter(
                        entries: entries,
                        positions: _positions,
                        selectedId: _selectedId,
                        categoryColor: _categoryColor,
                        nodeRadius: _nodeRadius,
                        isDark: context.isDark,
                      ),
                    ),
                  ),
                ),
              );
            }),
          ),
        ],
      ),
    );
  }
}

class _GraphPainter extends CustomPainter {
  final List<MemoryEntry> entries;
  final Map<String, Offset> positions;
  final String? selectedId;
  final Color Function(String) categoryColor;
  final double Function(MemoryEntry) nodeRadius;
  final bool isDark;

  const _GraphPainter({
    required this.entries,
    required this.positions,
    required this.selectedId,
    required this.categoryColor,
    required this.nodeRadius,
    required this.isDark,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final byId = {for (final e in entries) e.id: e};
    final edgePaint = Paint()
      ..strokeWidth = 1
      ..color = (isDark ? Colors.white : Colors.black).withValues(alpha: 0.15);
    final drawnPairs = <String>{};
    for (final e in entries) {
      final from = positions[e.id];
      if (from == null) continue;
      for (final linkedId in e.linkedIds) {
        final to = positions[linkedId];
        if (to == null || !byId.containsKey(linkedId)) continue;
        final key = ([e.id, linkedId]..sort()).join('|');
        if (drawnPairs.contains(key)) continue;
        drawnPairs.add(key);
        canvas.drawLine(from, to, edgePaint);
      }
    }

    for (final e in entries) {
      final pos = positions[e.id];
      if (pos == null) continue;
      final radius = nodeRadius(e);
      final color = categoryColor(e.category);
      canvas.drawCircle(
        pos,
        radius,
        Paint()..color = color.withValues(alpha: e.isActive ? 0.9 : 0.3),
      );
      if (e.id == selectedId) {
        canvas.drawCircle(
          pos,
          radius + 4,
          Paint()
            ..style = PaintingStyle.stroke
            ..strokeWidth = 2
            ..color = color,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant _GraphPainter oldDelegate) =>
      oldDelegate.entries.length != entries.length || oldDelegate.selectedId != selectedId;
}
