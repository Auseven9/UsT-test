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
  final TransformationController _viewController = TransformationController();
  // Only the very first layout auto-centers the view. Recentering on every
  // later recompute (e.g. a new memory arriving mid-session) would snap the
  // view back to centered/1.0 scale out from under a user actively zoomed
  // or panned into a node cluster, discarding their navigation.
  bool _hasCenteredView = false;

  @override
  void dispose() {
    _viewController.dispose();
    super.dispose();
  }

  /// When true, tapping nodes links/unlinks them (see [_handleTap]) instead
  /// of opening the detail sheet. [_linkAnchorId] holds the first node
  /// tapped while in this mode, waiting for a second.
  bool _linkMode = false;
  String? _linkAnchorId;

  /// Fire-and-forget: kicks off the layout computation on a background
  /// isolate (see [_computeGraphLayout]) if the entry set actually
  /// changed and nothing is already computing. Safe to call from inside
  /// `build()` — it only mutates state synchronously (readable in this
  /// same build pass, for the loading indicator) and the actual `setState`
  /// happens later, in the compute() callback, never during build itself.
  void _maybeStartLayout(List<MemoryEntry> entries, Size viewportSize) {
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
      // Without this, InteractiveViewer starts showing the canvas's raw
      // (0,0) corner — since the force-directed layout settles the node
      // cluster somewhere near the 1200x1200 canvas's own center rather
      // than near (0,0), that left the whole graph sitting off to the
      // lower right of a phone-sized viewport, needing a manual pan to
      // even find it (and every tap landing on nothing, since there was
      // nothing under the visible area). Centering the viewport on the
      // actual node cluster's bounding-box centroid right after the FIRST
      // layout fixes both, without later snapping a since-zoomed/panned
      // view back out from under the user on every subsequent recompute.
      if (!_hasCenteredView) {
        // Only latch once centering actually ran — if the viewport still
        // had zero size right at this moment (e.g. mid route-transition),
        // _centerViewOn is a no-op, and setting the flag anyway would skip
        // centering forever, permanently leaving the graph off-screen
        // instead of catching it on the next layout recompute.
        _hasCenteredView = result.isNotEmpty && viewportSize.width > 0 && viewportSize.height > 0;
        _centerViewOn(result, viewportSize);
      }
    });
  }

  void _centerViewOn(Map<String, Offset> positions, Size viewportSize) {
    if (positions.isEmpty || viewportSize.width <= 0 || viewportSize.height <= 0) return;
    var minX = double.infinity, minY = double.infinity;
    var maxX = double.negativeInfinity, maxY = double.negativeInfinity;
    for (final p in positions.values) {
      minX = min(minX, p.dx);
      maxX = max(maxX, p.dx);
      minY = min(minY, p.dy);
      maxY = max(maxY, p.dy);
    }
    final centroid = Offset((minX + maxX) / 2, (minY + maxY) / 2);
    _viewController.value = Matrix4.identity()
      ..translateByDouble(
        viewportSize.width / 2 - centroid.dx,
        viewportSize.height / 2 - centroid.dy,
        0,
        1,
      );
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
    if (closestId == null) {
      // Tapping empty canvas outside link mode clears the current
      // highlight — restored here since link mode's early-return for a
      // miss shouldn't also swallow this for ordinary node selection.
      if (!_linkMode && _selectedId != null) setState(() => _selectedId = null);
      return;
    }

    if (_linkMode) {
      if (_linkAnchorId == null) {
        setState(() => _linkAnchorId = closestId);
        return;
      }
      if (_linkAnchorId == closestId) {
        setState(() => _linkAnchorId = null); // tapped anchor again — cancel
        return;
      }
      MemoryEntry? anchor;
      for (final e in entries) {
        if (e.id == _linkAnchorId) {
          anchor = e;
          break;
        }
      }
      if (anchor == null) {
        // The anchor was deleted (e.g. by working-memory maintenance)
        // between the first tap and this second one — nothing to link.
        setState(() => _linkAnchorId = null);
        return;
      }
      final alreadyLinked = anchor.linkedIds.contains(closestId);
      final memory = Get.find<MemoryService>();
      if (alreadyLinked) {
        memory.unlinkManually(anchor.id, closestId);
        Get.snackbar('Unlinked', 'Removed the manual link.',
            snackPosition: SnackPosition.BOTTOM, duration: const Duration(seconds: 2));
      } else {
        memory.linkManually(anchor.id, closestId);
        Get.snackbar('Linked', 'Connected the two memories.',
            snackPosition: SnackPosition.BOTTOM, duration: const Duration(seconds: 2));
      }
      setState(() => _linkAnchorId = null);
      return;
    }

    setState(() => _selectedId = closestId);
    final entry = entries.firstWhere((e) => e.id == closestId);
    _showEntrySheet(entry);
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
                if (entry.entityName.isNotEmpty ||
                    entry.location.isNotEmpty ||
                    entry.participants.isNotEmpty ||
                    entry.connection.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  Text(
                    [
                      if (entry.entityName.isNotEmpty)
                        '${entry.entityType == 'none' ? '' : '${entry.entityType}: '}${entry.entityName}',
                      if (entry.location.isNotEmpty) 'at ${entry.location}',
                      if (entry.participants.isNotEmpty)
                        'with ${entry.participants.join(', ')}',
                      if (entry.connection.isNotEmpty) entry.connection,
                    ].join(' · '),
                    style: TextStyle(
                      fontSize: 12,
                      fontStyle: FontStyle.italic,
                      color: sheetContext.textM,
                    ),
                  ),
                ],
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
                  const Spacer(),
                  IconButton(
                    icon: Icon(
                      _linkMode ? Icons.link_rounded : Icons.link_outlined,
                      size: 20,
                      color: _linkMode ? AppColors.accent : context.textD,
                    ),
                    tooltip: _linkMode ? 'Exit link mode' : 'Manually link/unlink memories',
                    onPressed: () => setState(() {
                      _linkMode = !_linkMode;
                      _linkAnchorId = null;
                    }),
                  ),
                  const SizedBox(width: 4),
                ],
              ),
            ),
          ),
          if (_linkMode)
            Container(
              width: double.infinity,
              color: AppColors.accent.withValues(alpha: 0.12),
              padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
              child: Text(
                _linkAnchorId == null
                    ? 'Tap a memory, then tap a second one to link or unlink them.'
                    : 'Now tap the second memory. Tap the first one again to cancel.',
                style: TextStyle(fontSize: 12, color: context.text),
                textAlign: TextAlign.center,
              ),
            ),
          Expanded(
            child: LayoutBuilder(builder: (context, constraints) {
              final viewportSize = Size(constraints.maxWidth, constraints.maxHeight);
              return Obx(() {
                final entries = memory.entries.toList();
                if (entries.isEmpty) {
                  return Center(
                    child: Text('Nothing remembered yet.', style: TextStyle(color: context.textD)),
                  );
                }
                _maybeStartLayout(entries, viewportSize);
                if (_computingLayout && _positions.isEmpty) {
                  return const Center(child: CircularProgressIndicator());
                }
                return InteractiveViewer(
                  transformationController: _viewController,
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
                          // The highlight ring doubles as the link-mode
                          // anchor indicator — same visual, different
                          // meaning depending on mode, so no extra painter
                          // logic.
                          selectedId: _linkMode ? _linkAnchorId : _selectedId,
                          categoryColor: _categoryColor,
                          nodeRadius: _nodeRadius,
                          isDark: context.isDark,
                        ),
                      ),
                    ),
                  ),
                );
              });
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
