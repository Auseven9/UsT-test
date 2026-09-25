import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:get/get.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../models/reminder.dart';

/// In-app reminders — surfaced when the app is open and a due date has
/// passed (see [checkDue]), not real OS push notifications. This app has no
/// notification channel/permission set up, and adding one is a real native
/// config change (AndroidManifest, runtime permission on Android 13+) that
/// deserves its own explicit decision, not something bundled silently into
/// a tool. This is the honest, buildable version: reminders persist and
/// surface reliably any time the app is foregrounded, just not while it's
/// fully closed.
class ReminderService extends GetxService {
  final reminders = <Reminder>[].obs;

  late File _file;
  Future<void> _persistChain = Future.value();

  Future<ReminderService> init() async {
    final appDir = await getApplicationDocumentsDirectory();
    final dir = Directory(p.join(appDir.path, 'PortableAI'));
    await dir.create(recursive: true);
    _file = File(p.join(dir.path, 'reminders.json'));
    await _load();
    return this;
  }

  Future<void> _load() async {
    try {
      if (!await _file.exists()) {
        reminders.value = [];
        return;
      }
      final raw = await _file.readAsString();
      final list = jsonDecode(raw) as List;
      final parsed = <Reminder>[];
      for (final j in list) {
        try {
          parsed.add(Reminder.fromJson(Map<String, dynamic>.from(j as Map)));
        } catch (_) {
          // Skip just this one.
        }
      }
      reminders.value = parsed;
    } catch (_) {
      reminders.value = [];
    }
  }

  Future<void> _persist() async {
    final snapshot = reminders.toList();
    _persistChain = _persistChain.then((_) async {
      try {
        await _file.writeAsString(jsonEncode(snapshot.map((r) => r.toJson()).toList()));
      } catch (_) {}
    });
    await _persistChain;
  }

  Future<Reminder> add(String text, DateTime dueAt) async {
    final reminder = Reminder(
      id: '${DateTime.now().microsecondsSinceEpoch}',
      text: text,
      dueAt: dueAt,
      createdAt: DateTime.now(),
    );
    reminders.add(reminder);
    await _persist();
    return reminder;
  }

  Future<void> dismiss(String id) async {
    reminders.removeWhere((r) => r.id == id);
    await _persist();
  }

  /// Reminders whose due time has passed and haven't been reported yet.
  /// Marks them fired (so the same reminder isn't reported twice) but
  /// leaves them in the list — a missed reminder stays visible/dismissible
  /// rather than silently vanishing the moment it's read.
  Future<List<Reminder>> checkDue() async {
    final now = DateTime.now();
    final due = reminders.where((r) => !r.fired && !r.dueAt.isAfter(now)).toList();
    if (due.isEmpty) return due;
    for (final r in due) {
      r.fired = true;
    }
    reminders.refresh();
    await _persist();
    return due;
  }
}
