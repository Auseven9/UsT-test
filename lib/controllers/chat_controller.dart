import 'dart:async';
import 'package:get/get.dart';

import '../models/chat_model.dart';
import '../models/message_model.dart';
import '../services/llm_service.dart';
import '../services/chat_storage_service.dart';
import '../services/conversation_memory_service.dart';
import '../services/memory_service.dart';

class ChatController extends GetxController {
  final LlmService _llm = Get.find<LlmService>();
  final ChatStorageService _storage = Get.find<ChatStorageService>();
  final ConversationMemoryService _contextWindow =
      Get.find<ConversationMemoryService>();
  final MemoryService _longTermMemory = Get.find<MemoryService>();

  final chats = <ChatModel>[].obs;
  final activeChatId = RxnString();
  final isGenerating = false.obs;
  final streamedResponse = ''.obs;
  final temperature = 0.7.obs;
  final systemPrompt = ''.obs;

  StreamSubscription<String>? _genSub;

  @override
  void onInit() {
    super.onInit();
    _loadChats();
    temperature.value = _storage.defaultTemperature;
    systemPrompt.value = _storage.globalSystemPrompt;
  }

  void _loadChats() {
    chats.value = _storage.getAllChats();
  }

  ChatModel? get activeChat {
    if (activeChatId.value == null) return null;
    try {
      return chats.firstWhere((c) => c.id == activeChatId.value);
    } catch (_) {
      return null;
    }
  }

  /// Create a new chat and switch to it.
  void newChat() {
    final chat = ChatModel(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      systemPrompt: systemPrompt.value,
    );
    chats.insert(0, chat);
    _storage.saveChat(chat);
    activeChatId.value = chat.id;
  }

  /// Switch to an existing chat.
  void switchChat(String id) {
    activeChatId.value = id;
    final chat = activeChat;
    if (chat != null) {
      systemPrompt.value = chat.systemPrompt;
    }
  }

  /// Delete a chat.
  void deleteChat(String id) {
    chats.removeWhere((c) => c.id == id);
    _storage.deleteChat(id);
    if (activeChatId.value == id) {
      activeChatId.value = chats.isNotEmpty ? chats.first.id : null;
    }
  }

  /// Send a user message and stream AI response.
  Future<void> sendMessage(String text, {String? modelFilename}) async {
    if (text.trim().isEmpty) return;
    final chat = activeChat;
    if (chat == null) return;

    // Add user message
    final userMsg = MessageModel(role: MessageRole.user, content: text.trim());
    chat.messages.add(userMsg);
    chat.autoTitle();
    chat.updatedAt = DateTime.now();

    // Lock model to this chat on first message
    if (chat.modelId.isEmpty && modelFilename != null) {
      chat.modelId = modelFilename;
    }

    _storage.saveChat(chat);
    chats.refresh();

    var effectiveSystemPrompt = chat.systemPrompt.isNotEmpty
        ? chat.systemPrompt
        : systemPrompt.value;

    // Cross-chat long-term memory recall (opt-in; no-op unless Smart Recall
    // is enabled and an embedding model is loaded). Injected as a labeled
    // block in the system prompt so it flows through the same context-window
    // budget as everything else below.
    final recalled = await _longTermMemory.recall(text.trim());
    if (recalled.isNotEmpty) {
      final block = StringBuffer(
        'Long-term memory — things you remember about the user from '
        'earlier conversations (use only if relevant):\n',
      );
      for (final m in recalled) {
        final date = m.eventDate ?? m.createdAt;
        block.writeln('- (${date.year}-${date.month.toString().padLeft(2, '0')}-'
            '${date.day.toString().padLeft(2, '0')}) ${m.text}');
      }
      effectiveSystemPrompt = '$effectiveSystemPrompt\n\n${block.toString().trim()}';
    }

    // Build message history for the LLM, trimmed to fit the active model's
    // context window (newest turns kept, oldest dropped first — see
    // ConversationMemoryService). The full history stays persisted in Hive
    // regardless of what gets sent to the model.
    final candidateMessages =
        chat.messages.where((m) => !m.isSystem).toList();
    final fittedMessages = await _contextWindow.fitToContext(
      candidateMessages,
      systemPrompt: effectiveSystemPrompt,
    );
    final history = fittedMessages.map((m) => m.toLlamaMessage()).toList();

    // Start generation
    isGenerating.value = true;
    streamedResponse.value = '';

    final aiMsg = MessageModel(role: MessageRole.assistant, content: '');
    chat.messages.add(aiMsg);
    chats.refresh();

    try {
      final stream = _llm.generate(
        messages: history,
        systemPrompt: effectiveSystemPrompt,
        temperature: temperature.value,
      );

      await for (final token in stream) {
        streamedResponse.value += token;
        aiMsg.content = streamedResponse.value;
        // Throttle UI refreshes
        chats.refresh();
      }
    } catch (e) {
      if (aiMsg.content.isEmpty) {
        aiMsg.content = '⚠ Error: ${e.toString()}';
      }
    } finally {
      // Clean up any trailing stop tokens or whitespace
      aiMsg.content = aiMsg.content
          .replaceAll(RegExp(
            r'<\|end\|>|<\|eot_id\|>|<\|endoftext\|>|<\|im_end\|>|<\|im_start\|>'
            r'|<end_of_turn>|<start_of_turn>|<\|assistant\|>|<\|user\|>|<\|system\|>'
            r'|<\|pad\|>|</s>|<s>|\[INST\]|\[/INST\]|\[end\]'
          ), '')
          .trim();
      streamedResponse.value = '';
      chat.updatedAt = DateTime.now();
      _storage.saveChat(chat);
      chats.refresh();
    }

    // Best-effort long-term memory extraction, run while isGenerating is
    // still true: it makes its own call into LlmService.generate(), and
    // that call rejects if one is already in flight, so this has to finish
    // before the UI can trigger another send.
    await _longTermMemory.extractAndStore(
      chat: chat,
      userMessage: userMsg,
      assistantMessage: aiMsg,
    );
    isGenerating.value = false;
  }

  /// Stop current generation.
  void stopGeneration() {
    _llm.stopGeneration();
    isGenerating.value = false;
  }

  /// Update the system prompt for the active chat.
  void updateSystemPrompt(String prompt) {
    systemPrompt.value = prompt;
    final chat = activeChat;
    if (chat != null) {
      chat.systemPrompt = prompt;
      _storage.saveChat(chat);
    }
  }

  /// Set and persist the global system prompt.
  void setGlobalSystemPrompt(String prompt) {
    systemPrompt.value = prompt;
    _storage.globalSystemPrompt = prompt;
  }

  /// Clear global system prompt.
  void clearGlobalSystemPrompt() {
    systemPrompt.value = '';
    _storage.globalSystemPrompt = '';
  }

  void updateTemperature(double temp) {
    temperature.value = temp;
    _storage.defaultTemperature = temp;
  }

  @override
  void onClose() {
    _genSub?.cancel();
    super.onClose();
  }
}
