/// A reminder the model (via the `set_reminder` tool) or the user asked to
/// be surfaced at a future time. Checked in-app (see `ReminderService.
/// checkDue`) rather than as a real OS notification — this app has no
/// notification permission/channel set up, so a due reminder surfaces the
/// next time the app is actually open, not as a background push.
class Reminder {
  final String id;
  final String text;
  final DateTime dueAt;
  final DateTime createdAt;
  bool fired;

  Reminder({
    required this.id,
    required this.text,
    required this.dueAt,
    required this.createdAt,
    this.fired = false,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'text': text,
        'dueAt': dueAt.toIso8601String(),
        'createdAt': createdAt.toIso8601String(),
        'fired': fired,
      };

  factory Reminder.fromJson(Map<String, dynamic> json) => Reminder(
        id: json['id'] as String? ?? DateTime.now().microsecondsSinceEpoch.toString(),
        text: json['text'] as String? ?? '',
        dueAt: DateTime.tryParse(json['dueAt'] as String? ?? '') ?? DateTime.now(),
        createdAt: DateTime.tryParse(json['createdAt'] as String? ?? '') ?? DateTime.now(),
        fired: json['fired'] as bool? ?? false,
      );
}
