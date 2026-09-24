import 'package:hive/hive.dart';

part 'message_model.g.dart';

@HiveType(typeId: 1)
enum MessageRole {
  @HiveField(0)
  user,
  @HiveField(1)
  assistant,
  @HiveField(2)
  system,
}

@HiveType(typeId: 2)
class MessageModel extends HiveObject {
  @HiveField(0)
  final MessageRole role;

  @HiveField(1)
  String content;

  @HiveField(2)
  final DateTime timestamp;

  @HiveField(3)
  String? imageBase64;

  @HiveField(4)
  String? imageMimeType;

  /// The model's chain-of-thought/reasoning for this message, kept separate
  /// from [content] so it can be shown collapsed instead of dumped inline —
  /// populated either from llamadart's native `thinking` delta (correctly
  /// detected reasoning-capable templates) or from the defensive fallback
  /// split in LlmService for models whose template isn't recognized.
  @HiveField(5)
  String? reasoning;

  MessageModel({
    required this.role,
    required this.content,
    DateTime? timestamp,
    this.imageBase64,
    this.imageMimeType,
    this.reasoning,
  }) : timestamp = timestamp ?? DateTime.now();

  bool get isUser => role == MessageRole.user;
  bool get isAssistant => role == MessageRole.assistant;
  bool get isSystem => role == MessageRole.system;

  Map<String, String> toLlamaMessage() {
    return {
      'role': role.name,
      'content': content,
    };
  }
}
