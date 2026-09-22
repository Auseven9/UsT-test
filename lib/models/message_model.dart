import 'dart:convert';

import 'package:hive/hive.dart';
import 'package:llamadart/llamadart.dart';

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

  MessageModel({
    required this.role,
    required this.content,
    DateTime? timestamp,
    this.imageBase64,
    this.imageMimeType,
  }) : timestamp = timestamp ?? DateTime.now();

  bool get isUser => role == MessageRole.user;
  bool get isAssistant => role == MessageRole.assistant;
  bool get isSystem => role == MessageRole.system;

  LlamaChatMessage toLlamaChatMessage() {
    final chatRole = switch (role) {
      MessageRole.user => LlamaChatRole.user,
      MessageRole.assistant => LlamaChatRole.assistant,
      MessageRole.system => LlamaChatRole.system,
    };

    if (imageBase64 != null && imageBase64!.isNotEmpty) {
      return LlamaChatMessage.withContent(
        role: chatRole,
        content: [
          if (content.isNotEmpty) LlamaTextContent(content),
          LlamaImageContent(bytes: base64Decode(imageBase64!)),
        ],
      );
    }

    return LlamaChatMessage.fromText(role: chatRole, text: content);
  }
}
