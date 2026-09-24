import 'dart:async';
import 'package:get/get.dart';
import 'package:llamadart/llamadart.dart';

import '../models/chat_model.dart';
import '../models/message_model.dart';
import '../services/llm_service.dart';
import '../services/chat_storage_service.dart';
import '../services/embedding_service.dart';
import '../services/memory_service.dart';

/// Hard safety cap on a single response, independent of context size —
/// bounds worst-case generation time/battery drain if a model never
/// produces a natural stop token, without limiting normal replies (well
/// above what a typical chat turn needs).
const _maxResponseTokens = 1024;

/// How many recent turns worth of tokens the drop-summarization prompt
/// itself is allowed to use for its own (short) output.
const _summaryMaxTokens = 200;

class ChatController extends GetxController {
  final LlmService _llm = Get.find<LlmService>();
  final ChatStorageService _storage = Get.find<ChatStorageService>();
  final EmbeddingService _embedding = Get.find<EmbeddingService>();
  final MemoryService _memory = Get.find<MemoryService>();

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

    // Build message history as real chat messages — the effective system
    // prompt (per-chat override, else global) goes first so the model's own
    // Jinja chat template (read from the GGUF's tokenizer.chat_template) can
    // render it correctly for whatever family is loaded (Gemma, Qwen, Phi, …).
    var effectiveSystemPrompt =
        chat.systemPrompt.isNotEmpty ? chat.systemPrompt : systemPrompt.value;

    // Persistent memory: pull in anything relevant this app has remembered
    // from *any* past conversation, not just this one.
    if (_storage.persistentMemoryEnabled &&
        _embedding.isLoaded.value &&
        _memory.entries.isNotEmpty) {
      final queryVector = await _embedding.embed(text.trim());
      if (queryVector != null) {
        final relevant = _memory.topK(queryVector, k: 3);
        if (relevant.isNotEmpty) {
          final memoryBlock = relevant.map((e) => '- ${e.text}').join('\n');
          effectiveSystemPrompt =
              '$effectiveSystemPrompt\n\nRelevant memory from earlier conversations:\n$memoryBlock';
        }
      }
    }

    final history = <LlamaChatMessage>[
      if (effectiveSystemPrompt.isNotEmpty)
        LlamaChatMessage.fromText(
          role: LlamaChatRole.system,
          text: effectiveSystemPrompt,
        ),
      ...chat.messages.where((m) => !m.isSystem).map(
            (m) => LlamaChatMessage.fromText(
              role: m.isUser ? LlamaChatRole.user : LlamaChatRole.assistant,
              text: m.content,
            ),
          ),
    ];

    // Sliding window: drop the oldest turns (never the system message) so
    // history fits the model's real context size, leaving room to respond.
    // Must reserve exactly what generation is actually capped at below
    // (_maxResponseTokens) — a smaller default here would let history pack
    // in more than the response headroom it's supposed to leave, and the
    // two silently drifting apart is exactly how "leaves room to respond"
    // stops being true.
    final fitted =
        await _llm.fitToContext(history, reserveForResponse: _maxResponseTokens);
    final hasSystemMsg =
        history.isNotEmpty && history.first.role == LlamaChatRole.system;
    final droppedCount = history.length - fitted.length;
    final droppedTurns = droppedCount > 0
        ? history.sublist(
            hasSystemMsg ? 1 : 0,
            (hasSystemMsg ? 1 : 0) + droppedCount,
          )
        : const <LlamaChatMessage>[];

    // A background memory-summarization call (fired after a previous
    // message, see _summarizeAndRemember) may still be occupying the
    // engine's single generation slot. It's deliberately short, so wait
    // briefly for it to clear instead of letting generateChatCompletion's
    // StateError surface as a confusing "Another generation is already in
    // progress" chat error for something the user never saw start.
    var waitedMs = 0;
    while (_llm.isGenerating.value && waitedMs < 5000) {
      await Future.delayed(const Duration(milliseconds: 100));
      waitedMs += 100;
    }

    // Start generation
    isGenerating.value = true;
    streamedResponse.value = '';

    final aiMsg = MessageModel(role: MessageRole.assistant, content: '');
    chat.messages.add(aiMsg);
    chats.refresh();

    try {
      final stream = _llm.generateChatCompletion(
        messages: fitted,
        params: GenerationParams(
          temp: temperature.value,
          maxTokens: _maxResponseTokens,
        ),
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
      aiMsg.content = aiMsg.content.trim();
      isGenerating.value = false;
      streamedResponse.value = '';
      chat.updatedAt = DateTime.now();
      _storage.saveChat(chat);
      chats.refresh();
    }

    // Fire-and-forget: if the sliding window just pushed history out of the
    // model's context, distill it into a memory note before it's gone for
    // good, so it can still surface in a later conversation. Never blocks
    // or fails the visible response above.
    if (_storage.persistentMemoryEnabled &&
        droppedTurns.isNotEmpty &&
        _embedding.isLoaded.value) {
      unawaited(_summarizeAndRemember(droppedTurns, chat.id));
    }
  }

  /// Summarizes a block of about-to-be-dropped conversation turns into a
  /// short memory note, embeds it, and persists it — best-effort throughout;
  /// any failure here is silent since it must never surface as a chat error.
  Future<void> _summarizeAndRemember(
    List<LlamaChatMessage> droppedTurns,
    String chatId,
  ) async {
    try {
      if (!_llm.isLoaded.value || _llm.isGenerating.value) return;

      final summaryRequest = <LlamaChatMessage>[
        LlamaChatMessage.fromText(
          role: LlamaChatRole.system,
          text: 'You distill conversation excerpts into short-term memory '
              'notes. Given the excerpt below, write 2-4 concise bullet '
              'points capturing only facts, preferences, or context worth '
              'remembering long-term. If there is nothing worth '
              'remembering, respond with exactly: NONE',
        ),
        ...droppedTurns,
        LlamaChatMessage.fromText(
          role: LlamaChatRole.user,
          text: 'Summarize the excerpt above for long-term memory.',
        ),
      ];

      final buffer = StringBuffer();
      await for (final token in _llm.generateChatCompletion(
        messages: summaryRequest,
        params: const GenerationParams(temp: 0.3, maxTokens: _summaryMaxTokens),
      )) {
        buffer.write(token);
      }

      final summary = buffer.toString().trim();
      if (summary.isEmpty || summary.toUpperCase() == 'NONE') return;

      final vector = await _embedding.embed(summary);
      if (vector == null) return;

      await _memory.add(summary, vector, sourceChatId: chatId);
    } catch (_) {
      // Best-effort background task — never let this affect the chat UI.
    }
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
