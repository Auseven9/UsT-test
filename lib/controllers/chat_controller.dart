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
/// produces a natural stop token. Reasoning models can easily burn several
/// hundred tokens of chain-of-thought before ever reaching an answer, so
/// this leaves real headroom rather than the flat 1024 it used to be.
const _maxResponseTokens = 2048;

/// How many tokens the per-turn memory-extraction prompt is allowed for its
/// own (short) output.
const _extractionMaxTokens = 200;

class ChatController extends GetxController {
  final LlmService _llm = Get.find<LlmService>();
  final ChatStorageService _storage = Get.find<ChatStorageService>();
  final EmbeddingService _embedding = Get.find<EmbeddingService>();
  final MemoryService _memory = Get.find<MemoryService>();

  final chats = <ChatModel>[].obs;
  final activeChatId = RxnString();
  final isGenerating = false.obs;
  final streamedResponse = ''.obs;
  final streamedReasoning = ''.obs;
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

    // A background memory-extraction call (see _rememberFromTurn, fired
    // after every previous message now, not just on context eviction) may
    // still be occupying the engine's single generation slot. It's
    // deliberately short, so wait briefly for it to clear instead of
    // letting generateChatCompletion's StateError surface as a confusing
    // "Another generation is already in progress" chat error for something
    // the user never saw start.
    var waitedMs = 0;
    while (_llm.isGenerating.value && waitedMs < 5000) {
      await Future.delayed(const Duration(milliseconds: 100));
      waitedMs += 100;
    }

    // Start generation
    isGenerating.value = true;
    streamedResponse.value = '';
    streamedReasoning.value = '';

    final aiMsg = MessageModel(role: MessageRole.assistant, content: '');
    chat.messages.add(aiMsg);
    chats.refresh();

    try {
      final stream = _llm.generateChatCompletion(
        messages: fitted,
        params: GenerationParams(
          temp: temperature.value,
          maxTokens: _maxResponseTokens,
          topP: _storage.topP,
          topK: _storage.topK,
          minP: _storage.minP,
        ),
        enableThinking: _storage.enableModelThinking,
      );

      await for (final chunk in stream) {
        if (chunk.thinking.isNotEmpty) {
          streamedReasoning.value += chunk.thinking;
          aiMsg.reasoning = streamedReasoning.value;
        }
        if (chunk.content.isNotEmpty) {
          streamedResponse.value += chunk.content;
          aiMsg.content = streamedResponse.value;
        }
        // Throttle UI refreshes
        chats.refresh();
      }
    } catch (e) {
      if (aiMsg.content.isEmpty) {
        aiMsg.content = '⚠ Error: ${e.toString()}';
      }
    } finally {
      // Defensive fallback for a model whose chat template llamadart didn't
      // recognize (native `thinking` stayed empty even though this specific
      // reply clearly reasoned before answering): split any leaked
      // control-token-looking text out of the final content rather than
      // showing it raw. A no-op when native separation already worked.
      if ((aiMsg.reasoning == null || aiMsg.reasoning!.isEmpty) &&
          aiMsg.content.isNotEmpty) {
        final split = LlmService.splitLeakedControlTokens(aiMsg.content);
        if (split.reasoning.isNotEmpty) {
          aiMsg.reasoning = split.reasoning;
          aiMsg.content = split.answer;
        }
      }
      aiMsg.content = aiMsg.content.trim();
      isGenerating.value = false;
      streamedResponse.value = '';
      streamedReasoning.value = '';
      chat.updatedAt = DateTime.now();
      _storage.saveChat(chat);
      chats.refresh();
    }

    // Fire-and-forget: distill *this* exchange into a memory note as soon as
    // it happens, rather than waiting for the sliding window to evict it
    // later — memory should reflect what the user just said, not just what
    // eventually got old. Never blocks or fails the visible response above.
    if (_storage.persistentMemoryEnabled &&
        aiMsg.content.isNotEmpty &&
        _embedding.isLoaded.value) {
      unawaited(_rememberFromTurn(userMsg, aiMsg, chat.id));
    }
  }

  /// Looks at the latest user+assistant exchange and, if it contains
  /// anything worth remembering long-term, distills it into a memory note,
  /// embeds it, and persists it — best-effort throughout; any failure here
  /// is silent since it must never surface as a chat error. Runs after
  /// every turn (not just when the sliding window evicts one) so memory
  /// reflects what was just said, not just what's about to be forgotten.
  Future<void> _rememberFromTurn(
    MessageModel userMsg,
    MessageModel aiMsg,
    String chatId,
  ) async {
    try {
      if (!_llm.isLoaded.value || _llm.isGenerating.value) return;

      final extractionRequest = <LlamaChatMessage>[
        LlamaChatMessage.fromText(
          role: LlamaChatRole.system,
          text: 'You watch a conversation and extract durable facts or '
              'preferences worth remembering in future, unrelated '
              'conversations — things like the user\'s name, stated '
              'preferences, ongoing projects, or personal details. Given the '
              'exchange below, write ONE short sentence capturing anything '
              'worth remembering, or respond with exactly NONE if there is '
              'nothing durable (small talk, one-off questions, and the '
              'model\'s own replies don\'t count).',
        ),
        LlamaChatMessage.fromText(role: LlamaChatRole.user, text: userMsg.content),
        LlamaChatMessage.fromText(
          role: LlamaChatRole.assistant,
          text: aiMsg.content,
        ),
        LlamaChatMessage.fromText(
          role: LlamaChatRole.user,
          text: 'Extract anything worth remembering from the exchange above, '
              'in one sentence, or respond NONE.',
        ),
      ];

      final buffer = StringBuffer();
      await for (final chunk in _llm.generateChatCompletion(
        messages: extractionRequest,
        params: const GenerationParams(
          temp: 0.2,
          maxTokens: _extractionMaxTokens,
        ),
        enableThinking: false,
      )) {
        buffer.write(chunk.content);
      }

      final extracted = buffer.toString().trim();
      // Exact match on the sentinel only — a real extracted fact can
      // legitimately contain the word "none" (e.g. "has none of the common
      // allergies"), and .contains('NONE') would wrongly discard that.
      if (extracted.isEmpty || extracted.toUpperCase() == 'NONE') {
        return;
      }

      final vector = await _embedding.embed(extracted);
      if (vector == null) return;

      await _memory.add(extracted, vector, sourceChatId: chatId);
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
