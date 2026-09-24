/// A single crash/fatal-error record — deliberately separate from the
/// regular [LogEntry] stream (which is memory-only and capped) because a
/// crash log's entire purpose is to survive the crash that created it.
class CrashLogEntry {
  final DateTime timestamp;
  final String type; // flutter_error | platform_error | zone_error | model_load
  final String message;
  final String? stackTrace;
  final String? context; // free-form extra detail (file, backend, etc.)

  const CrashLogEntry({
    required this.timestamp,
    required this.type,
    required this.message,
    this.stackTrace,
    this.context,
  });

  factory CrashLogEntry.fromJson(Map<String, dynamic> json) => CrashLogEntry(
        timestamp: DateTime.parse(json['timestamp'] as String),
        type: json['type'] as String,
        message: json['message'] as String,
        stackTrace: json['stackTrace'] as String?,
        context: json['context'] as String?,
      );

  Map<String, dynamic> toJson() => {
        'timestamp': timestamp.toIso8601String(),
        'type': type,
        'message': message,
        'stackTrace': stackTrace,
        'context': context,
      };
}
