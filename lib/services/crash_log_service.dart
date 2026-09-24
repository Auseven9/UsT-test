import 'dart:convert';
import 'dart:io';

import 'package:get/get.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../models/crash_log_entry.dart';

/// Persistent crash/fatal-error log — deliberately independent of the normal
/// in-memory [LogService].
///
/// This is a plain singleton (not constructed through GetX) because it must
/// be usable from `main()`'s top-level error handlers *before* Flutter
/// bindings, Hive, or GetX are necessarily ready. [record] is always safe to
/// call: entries are buffered in memory until [init] resolves the on-disk
/// file, then flushed and written immediately from then on. The same
/// instance is also registered with GetX in `AppBindings` so the Settings UI
/// can read it reactively like any other service.
class CrashLogService extends GetxService {
  CrashLogService._internal();
  static final CrashLogService instance = CrashLogService._internal();

  final entries = <CrashLogEntry>[].obs;

  File? _file;
  final _pending = <CrashLogEntry>[];
  bool _initializing = false;

  Future<CrashLogService> init() async {
    if (_file != null || _initializing) return this;
    _initializing = true;
    try {
      final appDir = await getApplicationDocumentsDirectory();
      final dir = Directory(p.join(appDir.path, 'PortableAI', 'logs'));
      await dir.create(recursive: true);
      _file = File(p.join(dir.path, 'crash_log.jsonl'));

      await _loadFromDisk();

      // Flush anything recorded before init() finished.
      if (_pending.isNotEmpty) {
        for (final entry in _pending) {
          entries.add(entry);
        }
        await _appendToDisk(_pending);
        _pending.clear();
      }
    } catch (_) {
      // If we can't even set up the crash log, there's nothing safe to do
      // about it — it must never itself throw and break app startup.
    } finally {
      _initializing = false;
    }
    return this;
  }

  Future<void> _loadFromDisk() async {
    final file = _file;
    if (file == null || !await file.exists()) return;
    try {
      final lines = await file.readAsLines();
      final loaded = <CrashLogEntry>[];
      for (final line in lines) {
        if (line.trim().isEmpty) continue;
        try {
          loaded.add(CrashLogEntry.fromJson(
            Map<String, dynamic>.from(jsonDecode(line) as Map),
          ));
        } catch (_) {
          // Skip a corrupted line rather than losing the rest of the log.
        }
      }
      entries.value = loaded;
    } catch (_) {
      // Unreadable file — start fresh rather than crash reading the crash log.
    }
  }

  Future<void> _appendToDisk(List<CrashLogEntry> newEntries) async {
    final file = _file;
    if (file == null) return;
    try {
      final buffer = StringBuffer();
      for (final entry in newEntries) {
        buffer.writeln(jsonEncode(entry.toJson()));
      }
      await file.writeAsString(buffer.toString(), mode: FileMode.append);
    } catch (_) {
      // Best-effort by design — a crash log that fails to write must never
      // itself throw during error handling.
    }
  }

  /// Records a crash/fatal-error entry. Safe to call at any point in the
  /// app's lifecycle, including before [init] has resolved.
  void record(
    String type,
    String message, {
    String? stackTrace,
    String? context,
  }) {
    final entry = CrashLogEntry(
      timestamp: DateTime.now(),
      type: type,
      message: message,
      // Cap stack traces — some native/FFI traces are enormous and this is
      // meant to be human-readable in-app, not a full dump.
      stackTrace: stackTrace != null && stackTrace.length > 4000
          ? '${stackTrace.substring(0, 4000)}\n… (truncated)'
          : stackTrace,
      context: context,
    );

    if (_file == null) {
      _pending.add(entry);
      entries.add(entry);
      return;
    }
    entries.add(entry);
    // Fire-and-forget: recording a crash must never itself become an
    // unhandled async error.
    _appendToDisk([entry]).catchError((_) {});
  }

  Future<void> clear() async {
    entries.clear();
    try {
      await _file?.writeAsString('');
    } catch (_) {}
  }

  Future<void> deleteAt(int index) async {
    if (index < 0 || index >= entries.length) return;
    entries.removeAt(index);
    try {
      final buffer = StringBuffer();
      for (final entry in entries) {
        buffer.writeln(jsonEncode(entry.toJson()));
      }
      await _file?.writeAsString(buffer.toString());
    } catch (_) {}
  }

  /// Full log as plain text, for sharing/export — same pattern as
  /// [LogService.exportAll].
  String exportAll() {
    final buf = StringBuffer();
    buf.writeln('=== Portable AI Crash Log ===');
    buf.writeln('Exported: ${DateTime.now().toIso8601String()}');
    buf.writeln('Total entries: ${entries.length}');
    buf.writeln('');
    for (final entry in entries) {
      buf.writeln('[${entry.timestamp.toIso8601String()}] [${entry.type}] ${entry.message}');
      if (entry.context != null) buf.writeln('  context: ${entry.context}');
      if (entry.stackTrace != null) buf.writeln(entry.stackTrace);
      buf.writeln('');
    }
    return buf.toString();
  }
}
