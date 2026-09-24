import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:get/get.dart';
import 'package:llamadart/llamadart.dart';

import '../models/chat_model.dart';
import '../models/message_model.dart';
import '../models/memory_entry.dart';
import '../services/llm_service.dart';
import '../services/chat_storage_service.dart';
import '../services/embedding_service.dart';
import '../services/helper_llm_service.dart';
import '../services/memory_service.dart';
import '../services/memory_heuristics.dart';
import '../services/tool_definitions.dart';

/// Hard safety cap on a single response, independent of context size —
/// bounds worst-case generation time/battery drain if a model never
/// produces a natural stop token. Reasoning models can easily burn several
/// hundred tokens of chain-of-thought before ever reaching an answer, so
/// this leaves real headroom rather than the flat 1024 it used to be.
const _maxResponseTokens = 2048;

/// Conservative flat reserve for the vision tokens an attached image adds to
/// the prompt once encoded by the mmproj/CLIP pipeline — llamadart's
/// LlamaChatMessage.content getter (which fitToContext's token counting
/// relies on) only sees the text caption, not the image, so without this the
/// sliding window would undercount an image turn's real size and could pack
/// in more history than actually fits once the image is encoded.
const _imageTokenReserve = 1024;

/// Bounds how many tool-call ↔ tool-result round trips a single message can
/// trigger before the app just gives up and returns whatever's happened so
/// far — a model that keeps calling tools instead of answering shouldn't be
/// able to turn one message into an unbounded battery/time sink on a phone.
const _maxToolRounds = 3;

class ChatController extends GetxController {
  final LlmService _llm = Get.find<LlmService>();
  final ChatStorageService _storage = Get.find<ChatStorageService>();
  final EmbeddingService _embedding = Get.find<EmbeddingService>();
  final MemoryService _memory = Get.find<MemoryService>();
  final HelperLlmService _helper = Get.find<HelperLlmService>();

  // Rebuilt on every call rather than cached — `includeMemorySearch` must
  // track the user's live Persistent Memory setting, since a model still
  // holding a `search_memory` tool from before the user turned that setting
  // off would keep reading memories the toggle was meant to stop exposing.
  List<ToolDefinition> get _tools => buildToolDefinitions(
        memory: _memory,
        includeMemorySearch: _storage.persistentMemoryEnabled,
      );

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

  /// Send a user message and stream AI response. [imageBytes] (already
  /// downscaled by the caller — see ImageUtils.downscaleImageBytes) attaches
  /// an image to this turn, only meaningful when a vision projector is
  /// paired with the loaded model.
  Future<void> sendMessage(
    String text, {
    String? modelFilename,
    Uint8List? imageBytes,
    String imageMimeType = 'image/png',
  }) async {
    if (text.trim().isEmpty && imageBytes == null) return;
    final chat = activeChat;
    if (chat == null) return;

    // Add user message
    final userMsg = MessageModel(
      role: MessageRole.user,
      content: text.trim(),
      imageBase64: imageBytes != null ? base64Encode(imageBytes) : null,
      imageMimeType: imageBytes != null ? imageMimeType : null,
    );
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

    // Persistent memory retrieval — heuristic-gated so it costs nothing on
    // the common case. A cheap regex pass decides whether the message is
    // worth an embedding call at all; when it's not (small talk, short
    // acknowledgements), retrieval is skipped for free. When it is, the
    // *same* embedding vector is reused below if the text also looks worth
    // storing. (Storing a memorable turn does invoke a model for real
    // extraction — see _extractAndRememberFromTurn — but only on turns this
    // gate already flagged, and ideally on a dedicated small helper model
    // rather than the main chat model; see HelperLlmService.)
    List<double>? queryVector;
    // Captured here so the extraction step later can reuse the exact same
    // retrieval — needed to let it judge whether the new turn *contradicts*
    // one of these (e.g. an updated fact) without a second retrieval pass.
    List<MemoryEntry> relevantMemoriesForTurn = const [];
    final trimmedText = text.trim();
    if (_storage.persistentMemoryEnabled && _embedding.isLoaded.value) {
      // The model has no memory-write tool (search_memory is read-only, by
      // design — see tool_definitions.dart) because storage isn't its job:
      // it happens automatically in the background after the turn. Without
      // being told that, a model asked to "remember X" — finding no save
      // tool available — tends to narrate as if it saved something anyway
      // ("I'm filing that away"), which is a little dishonest since it did
      // nothing of the kind. One short line heads that off.
      effectiveSystemPrompt =
          '$effectiveSystemPrompt\n\nDurable facts and preferences worth '
          'remembering are captured automatically in the background after '
          'each message — you have no tool for this and don\'t need one; '
          'just answer naturally.';

      if (!MemoryHeuristics.looksTrivial(trimmedText) && _memory.entries.isNotEmpty) {
        queryVector = await _embedding.embed(trimmedText);
        if (queryVector != null) {
          final relevant = _memory.topK(queryVector, k: 2, minScore: 0.35);
          relevantMemoriesForTurn = relevant;
          if (relevant.isNotEmpty) {
            final memoryBlock = relevant.map((e) => '- ${e.text}').join('\n');
            effectiveSystemPrompt =
                '$effectiveSystemPrompt\n\nRelevant memory from earlier conversations:\n$memoryBlock';
          }
        }
      }
    }

    // Only the just-attached image (if any) is sent as real multimodal
    // content — older messages that happened to carry an image are
    // represented as text only, with a short marker in their place. Vision
    // encoding is expensive per the same on-device efficiency constraint
    // that shaped the memory system: re-encoding every image in history on
    // every single turn would make a multi-turn image conversation get
    // slower with each reply instead of staying flat.
    final history = <LlamaChatMessage>[
      if (effectiveSystemPrompt.isNotEmpty)
        LlamaChatMessage.fromText(
          role: LlamaChatRole.system,
          text: effectiveSystemPrompt,
        ),
      ...chat.messages.where((m) => !m.isSystem).map((m) {
        final role = m.isUser ? LlamaChatRole.user : LlamaChatRole.assistant;
        if (identical(m, userMsg) && imageBytes != null) {
          return LlamaChatMessage.withContent(
            role: role,
            content: [
              LlamaTextContent(m.content),
              LlamaImageContent(bytes: imageBytes),
            ],
          );
        }
        final text = m.imageBase64 != null
            ? '${m.content}\n[Image attached]'.trim()
            : m.content;
        return LlamaChatMessage.fromText(role: role, text: text);
      }),
    ];

    // Sliding window: drop the oldest turns (never the system message) so
    // history fits the model's real context size, leaving room to respond.
    // Must reserve exactly what generation is actually capped at below
    // (_maxResponseTokens) — a smaller default here would let history pack
    // in more than the response headroom it's supposed to leave, and the
    // two silently drifting apart is exactly how "leaves room to respond"
    // stops being true.
    final fitResult = await _llm.fitToContext(
      history,
      reserveForResponse:
          _maxResponseTokens + (imageBytes != null ? _imageTokenReserve : 0),
    );

    // The engine only allows one generation at a time — the Local API
    // Server feature (if enabled) can independently be mid-request right
    // now, and the background memory-extraction call from a previous turn
    // may also still be finishing — wait for that slot to free up instead
    // of surfacing a confusing "Another generation is already in progress"
    // chat error for something the user never saw start. Longer than the
    // old 5s: extraction is a real (if short) generation now, not just an
    // embedding call, and can take a while on a slow device.
    var waitedMs = 0;
    // Must exceed the background extraction call's own 20s timeout below —
    // otherwise this gives up first and throws while extraction is still
    // legitimately (if slowly) running, which is the exact error this wait
    // loop exists to avoid.
    while (_llm.isGenerating.value && waitedMs < 21000) {
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

    final activeTools = _storage.toolsEnabled ? _tools : null;
    var roundMessages = fitResult.messages;
    // Actual usable response budget for this turn — starts at whatever
    // fitToContext determined fit alongside the trimmed history, and gets
    // refreshed after each re-fit below. Always clamped to at most
    // _maxResponseTokens; only ever smaller, never larger, so a generation
    // call can never be asked to produce more than the context has room for.
    var responseBudget = fitResult.responseBudget.clamp(1, _maxResponseTokens);

    // Set once a round's tool calls have been consumed and fed back, so a
    // following round's text/reasoning doesn't run straight into the
    // previous round's with no separator (e.g. "Let me check.It's 22°C.",
    // or two rounds' reasoning reading as one duplicated block).
    // [contentPending, reasoningPending] — see _streamGeneration's doc.
    final pendingRoundSeparator = [false, false];
    var exhaustedRounds = false;

    try {
      for (var round = 0; round < _maxToolRounds; round++) {
        if (round > 0) {
          // Tool results were just appended to roundMessages — re-fit
          // against the context budget again, since those extra messages
          // (the tool call plus each tool's result) count against it too,
          // and the very first fitToContext call above never saw them.
          final refit = await _llm.fitToContext(
            roundMessages,
            reserveForResponse:
                _maxResponseTokens + (imageBytes != null ? _imageTokenReserve : 0),
          );
          roundMessages = refit.messages;
          responseBudget = refit.responseBudget.clamp(1, _maxResponseTokens);
        }
        exhaustedRounds = round == _maxToolRounds - 1;

        final toolCalls = await _streamGeneration(
          messages: roundMessages,
          aiMsg: aiMsg,
          tools: activeTools,
          maxTokens: responseBudget,
          separatorBox: pendingRoundSeparator,
        );

        if (toolCalls == null || toolCalls.isEmpty || activeTools == null) {
          exhaustedRounds = false; // Model answered directly — done.
          break;
        }
        if (streamedResponse.value.isNotEmpty) pendingRoundSeparator[0] = true;
        if (streamedReasoning.value.isNotEmpty) pendingRoundSeparator[1] = true;

        // Run each requested tool (independent calls run concurrently) and
        // feed the results back for a follow-up generation. The assistant's
        // tool-call message and each tool's result both need to go into
        // history for the next round to make sense of what happened,
        // mirroring the OpenAI-style tool protocol llamadart's chat
        // templates expect.
        final parsedCalls = toolCalls.map((tc) {
          final name = tc.function?.name ?? '';
          final rawArgs = tc.function?.arguments ?? '{}';
          Map<String, dynamic> args;
          try {
            args = (jsonDecode(rawArgs) as Map).cast<String, dynamic>();
          } catch (_) {
            args = const {};
          }
          return (tc: tc, name: name, rawArgs: rawArgs, args: args);
        }).toList();

        final callContents = parsedCalls
            .map((c) => LlamaToolCallContent(
                  id: c.tc.id,
                  name: c.name,
                  arguments: c.args,
                  rawJson: c.rawArgs,
                ))
            .toList();

        final resultMessages = await Future.wait(parsedCalls.map((c) async {
          ToolDefinition? tool;
          for (final t in activeTools) {
            if (t.name == c.name) {
              tool = t;
              break;
            }
          }
          Object? result;
          try {
            result = tool == null
                ? {'error': 'Unknown tool "${c.name}"'}
                : await tool.invoke(c.args);
          } catch (e) {
            result = {'error': e.toString()};
          }
          return LlamaChatMessage.withContent(
            role: LlamaChatRole.tool,
            content: [
              LlamaToolResultContent(id: c.tc.id, name: c.name, result: result),
            ],
          );
        }));

        roundMessages = [
          ...roundMessages,
          LlamaChatMessage.withContent(
            role: LlamaChatRole.assistant,
            content: callContents,
          ),
          ...resultMessages,
        ];
      }

      // Hit the round cap with a tool result that was never followed up on
      // (the model kept calling tools instead of answering) — force one
      // last tools-disabled generation so the user gets a real answer built
      // from whatever the tools returned, instead of a blank bubble.
      if (exhaustedRounds) {
        final refit = await _llm.fitToContext(
          roundMessages,
          reserveForResponse:
              _maxResponseTokens + (imageBytes != null ? _imageTokenReserve : 0),
        );
        roundMessages = refit.messages;
        responseBudget = refit.responseBudget.clamp(1, _maxResponseTokens);
        if (streamedResponse.value.isNotEmpty) pendingRoundSeparator[0] = true;
        if (streamedReasoning.value.isNotEmpty) pendingRoundSeparator[1] = true;
        await _streamGeneration(
          messages: roundMessages,
          aiMsg: aiMsg,
          tools: null,
          maxTokens: responseBudget,
          separatorBox: pendingRoundSeparator,
        );
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
        final split = LlmService.splitLeakedControlTokens(
          aiMsg.content,
          wasTruncated: _llm.lastGenerationHitTokenCap.value,
        );
        if (split.reasoning.isNotEmpty) {
          aiMsg.reasoning = split.reasoning;
          // An empty answer here means generation was cut off mid-thought,
          // before the model ever produced a real answer (see the doc on
          // splitLeakedControlTokens) — leaving the bubble truly blank
          // would just look broken, so say plainly what happened instead.
          aiMsg.content = split.answer.isNotEmpty
              ? split.answer
              : '_(cut off before finishing — see Thoughts above)_';
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

    // `looksMemorable` (a regex pass — first-person identity/preference
    // statements, explicit "remember this", etc.) decides WHETHER a turn is
    // worth capturing at all, so most turns (small talk, one-off questions)
    // never touch the chat model. But storing the raw *trigger* text
    // verbatim was wrong: "remember that." said right after the model
    // explained its own persona would get stored as literally the three
    // words "remember that." — the thing worth remembering was the
    // preceding content, not the command pointing at it. So once the gate
    // fires, a short bounded extraction call (not run on every turn — only
    // on turns the heuristic already flagged) reads the actual exchange and
    // distills what's durable, the same judgment call a human would need to
    // make when "that" doesn't name itself.
    if (_storage.persistentMemoryEnabled &&
        _embedding.isLoaded.value &&
        MemoryHeuristics.looksMemorable(trimmedText)) {
      unawaited(_extractAndRememberFromTurn(
        userMsg,
        aiMsg,
        chat.id,
        relevantMemoriesForTurn,
      ));
    }
  }

  /// Runs one generation call, streaming content/thinking into [aiMsg] and
  /// the `streamedResponse`/`streamedReasoning` Rx values as it arrives —
  /// shared by both the main tool-round loop and the round-cap fallback
  /// generation in [sendMessage] so the streaming/separator logic exists in
  /// exactly one place. [separatorBox] is a 2-element mutable box —
  /// `[contentPending, reasoningPending]` — tracked independently because
  /// thinking and content don't necessarily both appear in a given round
  /// (a pure tool-call round can be thinking-only). If the relevant flag is
  /// true when this round's first chunk of that kind arrives, a separator
  /// is inserted before it (so a follow-up round's reasoning/answer doesn't
  /// run straight into the previous round's with no boundary — a model
  /// re-deriving similar reasoning across rounds would otherwise look like
  /// its thoughts got flatly duplicated) and the flag is cleared. Returns
  /// the tool calls the model requested, if any.
  Future<List<LlamaCompletionChunkToolCall>?> _streamGeneration({
    required List<LlamaChatMessage> messages,
    required MessageModel aiMsg,
    required List<ToolDefinition>? tools,
    required int maxTokens,
    required List<bool> separatorBox,
  }) async {
    final stream = _llm.generateChatCompletion(
      messages: messages,
      params: GenerationParams(
        temp: temperature.value,
        maxTokens: maxTokens,
        topP: _storage.topP,
        topK: _storage.topK,
        minP: _storage.minP,
        penalty: _storage.repeatPenalty,
      ),
      enableThinking: _storage.enableModelThinking,
      tools: tools,
    );

    List<LlamaCompletionChunkToolCall>? toolCalls;
    await for (final chunk in stream) {
      if (chunk.thinking.isNotEmpty) {
        if (separatorBox[1]) {
          streamedReasoning.value += '\n\n---\n\n';
          separatorBox[1] = false;
        }
        streamedReasoning.value += chunk.thinking;
        aiMsg.reasoning = streamedReasoning.value;
      }
      if (chunk.content.isNotEmpty) {
        if (separatorBox[0]) {
          streamedResponse.value += '\n\n';
          separatorBox[0] = false;
        }
        streamedResponse.value += chunk.content;
        aiMsg.content = streamedResponse.value;
      }
      if (chunk.toolCalls != null && chunk.toolCalls!.isNotEmpty) {
        toolCalls = chunk.toolCalls;
      }
      chats.refresh();
    }
    return toolCalls;
  }

  /// Looks at the latest user+assistant exchange — already flagged by
  /// [MemoryHeuristics.looksMemorable] as worth capturing — and asks the
  /// model to distill anything durable, tagged with category/valence/tags,
  /// so something like "remember that" (which names nothing on its own)
  /// correctly captures whatever it was pointing at instead of being stored
  /// as its own three words. Also given [relevantMemories] — the same
  /// memories already retrieved for this turn's system prompt, at no extra
  /// retrieval cost — so it can flag when the new fact contradicts one of
  /// them (an updated fact), in which case the old one is marked superseded
  /// rather than left to compete equally with the current one. Bounded and
  /// cheap (short output, low temp), and only ever runs on a turn the
  /// heuristic gate already flagged — not every turn. Best-effort
  /// throughout; any failure here is silent since it must never surface as
  /// a chat error, and a small model producing malformed output degrades to
  /// the plain-text behavior this replaced rather than losing the memory.
  Future<void> _extractAndRememberFromTurn(
    MessageModel userMsg,
    MessageModel aiMsg,
    String chatId,
    List<MemoryEntry> relevantMemories,
  ) async {
    try {
      final useHelper = _helper.isLoaded.value;
      if (!useHelper && !_llm.isLoaded.value) return;

      // A bit more room than a one-sentence extraction needed, to fit the
      // small JSON structure (text/category/valence/tags) below.
      const extractionMaxTokens = 130;

      // A long exchange (a big assistant reply is exactly the case most
      // likely to actually contain something worth remembering) can exceed
      // the model's real context window on its own. fitToContext's general
      // sliding window isn't the right tool here: with only 4 messages,
      // walking newest-first unconditionally keeps the trailing "Extract
      // anything..." instruction (it's always the first one kept), and if
      // the assistant's reply alone doesn't fit the remaining budget, BOTH
      // the assistant reply and the user message before it get dropped —
      // leaving the extraction model nothing but a bare instruction to
      // extract from. Capping each message's own length directly instead
      // guarantees real content from both sides always survives, just
      // possibly truncated.
      //
      // The context estimate has to match whichever engine will actually
      // run this: the helper model's own context is fixed at 2048, but the
      // fallback path reuses the main model, whose context is user-
      // adjustable down to 512 via the Settings slider — assuming 2048
      // there regardless would overflow a smaller configured context.
      final extractionContextEstimate =
          useHelper ? 2048 : (await _llm.getContextSize()).clamp(256, 2048);
      const approxCharsPerToken = 4;
      // System prompt + JSON-format instructions + trailing instruction —
      // the JSON-structured extraction prompt roughly doubled this versus
      // the old one-sentence version, so the old flat 150 badly
      // undercounted real overhead once the "already remembered" context
      // block is added below too.
      const overheadTokens = 350;
      final availableTokens =
          (extractionContextEstimate - extractionMaxTokens - overheadTokens)
              .clamp(40, extractionContextEstimate);
      final maxCharsEach =
          (availableTokens * approxCharsPerToken / 2).round().clamp(100, 4000);
      String cap(String s) =>
          s.length > maxCharsEach ? '${s.substring(0, maxCharsEach)}…' : s;

      // Memory notes are themselves distilled to one short sentence, but
      // capped defensively anyway — this block adds to the same token
      // budget being carefully bounded above, and nothing guarantees an
      // old note (especially one recovered via the truncated-JSON fallback
      // elsewhere in this file) stayed short.
      String capMemoryText(String s) => s.length > 150 ? '${s.substring(0, 150)}…' : s;
      // Numbered, not bulleted — with more than one memory shown, the model
      // needs to say WHICH one a new fact contradicts (contradicts_index
      // below), not just "yes something was contradicted". A bare boolean
      // would leave no way to tell #1 from #2, and guessing "the first one"
      // is exactly the kind of wrong-target correction this feature exists
      // to prevent.
      final existingMemoryContext = relevantMemories.isEmpty
          ? ''
          : '\n\nAlready remembered, for reference — numbered so you can say '
              'which one (if any) the new exchange contradicts/updates (e.g. '
              'a changed fact):\n${relevantMemories.indexed.map((e) => '${e.$1 + 1}. ${capMemoryText(e.$2.text)}').join('\n')}';

      final extractionRequest = <LlamaChatMessage>[
        LlamaChatMessage.fromText(
          role: LlamaChatRole.system,
          text: 'You watch one exchange from a conversation and extract '
              'durable facts or preferences worth remembering in future, '
              'unrelated conversations — things like the user\'s name, '
              'stated preferences, ongoing projects, personal details, or '
              'something they explicitly asked to be remembered (if they '
              'said "remember that" or similar, figure out from the '
              'exchange what "that" refers to and capture the actual '
              'content, not the instruction itself).'
              '$existingMemoryContext\n\n'
              'Respond with ONLY a single-line JSON object, no other text: '
              '{"text": "<one short sentence, or empty string if nothing '
              'durable>", "category": "<fact|preference|event|instruction|'
              'general>", "valence": "<positive|negative|neutral>", '
              '"tags": ["<a few short keywords>"], "contradicts_index": '
              '<the number of the already-remembered item above that this '
              'replaces/updates, or null if none>}',
        ),
        LlamaChatMessage.fromText(role: LlamaChatRole.user, text: cap(userMsg.content)),
        LlamaChatMessage.fromText(role: LlamaChatRole.assistant, text: cap(aiMsg.content)),
        LlamaChatMessage.fromText(
          role: LlamaChatRole.user,
          text: 'Extract anything worth remembering from the exchange above '
              'as that JSON object, and nothing else.',
        ),
      ];

      String? extracted;
      if (useHelper) {
        // A dedicated, separate engine — no generation-slot contention with
        // the main model or the Local API Server, so no wait loop needed.
        // Still hard-timed out on general principle (any engine hanging on
        // a background task should never run forever).
        try {
          extracted = await _helper
              .complete(extractionRequest, maxTokens: extractionMaxTokens)
              .timeout(const Duration(seconds: 20));
        } on TimeoutException {
          _helper.stopGeneration();
          return;
        }
      } else {
        // No helper configured — fall back to the main model, which means
        // waiting for its single generation slot and hard-capping how long
        // this background task can occupy it.
        var waitedMs = 0;
        while (_llm.isGenerating.value && waitedMs < 3000) {
          await Future.delayed(const Duration(milliseconds: 100));
          waitedMs += 100;
        }
        if (_llm.isGenerating.value) return; // Still busy — skip this turn.

        // This reuses the main engine, which means its shared stats fields
        // (read directly by chat_bubble.dart/home_screen.dart for the "t/s"
        // readout on the user's actual last reply) would otherwise get
        // silently overwritten with this tiny background call's numbers —
        // snapshot and restore them so the visible reply stats stay put.
        final savedSpeed = _llm.lastGenerationSpeed.value;
        final savedTokens = _llm.lastGenerationTokens.value;
        final savedHitCap = _llm.lastGenerationHitTokenCap.value;

        final buffer = StringBuffer();
        try {
          final stream = _llm
              .generateChatCompletion(
                messages: extractionRequest,
                params: const GenerationParams(temp: 0.2, maxTokens: extractionMaxTokens),
                enableThinking: false,
              )
              .timeout(const Duration(seconds: 20));
          await for (final chunk in stream) {
            buffer.write(chunk.content);
          }
          extracted = buffer.toString();
        } on TimeoutException {
          await _llm.stopGeneration();
          return;
        } finally {
          _llm.lastGenerationSpeed.value = savedSpeed;
          _llm.lastGenerationTokens.value = savedTokens;
          _llm.lastGenerationHitTokenCap.value = savedHitCap;
        }
      }

      if (extracted == null) return;
      extracted = extracted.trim();

      final parsed = _parseExtractionJson(extracted);
      // A response that starts with '{' was clearly attempting JSON (the
      // prompt asks for nothing else) — if full parsing still failed, it's
      // a broken fragment, not a natural sentence, so it must never be
      // stored verbatim (braces, quotes, field names and all) as if it were
      // the memory itself. _parseExtractionJson already tries a regex-based
      // partial recovery of just the "text" field for exactly this
      // truncated-mid-object case; if even that comes back empty, there's
      // nothing safe to save from this turn.
      final looksLikeBrokenJson = parsed == null && extracted.trimLeft().startsWith('{');
      if (looksLikeBrokenJson) return;

      final noteText = parsed?.text ?? extracted;
      final category = parsed?.category ?? 'general';
      final valence = parsed?.valence ?? 'neutral';
      final tags = parsed?.tags ?? const <String>[];
      // 1-based index into relevantMemories, as shown to the model in the
      // numbered "already remembered" list — null/out-of-range means no
      // contradiction (or the model named something that isn't there).
      final contradictsIndex = parsed?.contradictsIndex;
      final contradictedMemory =
          (contradictsIndex != null && contradictsIndex >= 1 && contradictsIndex <= relevantMemories.length)
              ? relevantMemories[contradictsIndex - 1]
              : null;

      // Whether JSON parsed or not, the "nothing durable" check still runs
      // on the actual candidate text — a malformed-JSON model that still
      // wrote "NONE" as its `text` field (or the whole raw response, if
      // parsing failed outright) is still correctly recognized as empty.
      if (MemoryHeuristics.isNoMemorySentinel(noteText)) return;

      final vector = await _embedding.embed(noteText);
      if (vector == null) return;

      if (contradictedMemory != null) {
        // A correction must always create a fresh, superseding memory —
        // never go through addIfNotDuplicate's near-duplicate check here.
        // A corrected fact (e.g. "favorite color is green" replacing
        // "...is blue") can easily stay above the duplicate-similarity
        // threshold against the OLD entry it's meant to replace, since the
        // two sentences are structurally almost identical — that would
        // silently reinforce the wrong (outdated) memory instead of
        // recording the correction at all.
        final newId = await _memory.add(
          noteText,
          vector,
          sourceChatId: chatId,
          category: category,
          valence: valence,
          tags: tags,
        );
        if (newId != null) {
          await _memory.markSuperseded(contradictedMemory.id, newId);
        }
      } else {
        await _memory.addIfNotDuplicate(
          noteText,
          vector,
          sourceChatId: chatId,
          category: category,
          valence: valence,
          tags: tags,
        );
      }
    } catch (_) {
      // Best-effort background task — never let this affect the chat UI.
    }
  }

  static final RegExp _extractedTextFieldPattern =
      RegExp(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"');

  /// Best-effort parse of the extraction model's JSON response. Returns
  /// null (never throws) if nothing usable could be recovered at all, so
  /// the caller can fall back to treating the raw text as a plain memory
  /// note — a small quantized model failing to format valid JSON should
  /// degrade gracefully, not lose the memory entirely.
  ({String text, String category, String valence, List<String> tags, int? contradictsIndex})?
      _parseExtractionJson(String raw) {
    const validCategories = {'fact', 'preference', 'event', 'instruction', 'general'};
    const validValences = {'positive', 'negative', 'neutral'};

    try {
      final start = raw.indexOf('{');
      final end = raw.lastIndexOf('}');
      if (start != -1 && end != -1 && end > start) {
        final decoded = jsonDecode(raw.substring(start, end + 1));
        if (decoded is Map) {
          final text = (decoded['text'] as Object?)?.toString().trim() ?? '';
          final categoryRaw = (decoded['category'] as Object?)?.toString().toLowerCase();
          final valenceRaw = (decoded['valence'] as Object?)?.toString().toLowerCase();
          final tagsRaw = decoded['tags'];
          final tags = tagsRaw is List
              ? tagsRaw
                  .map((e) => e.toString().trim())
                  .where((e) => e.isNotEmpty)
                  .take(5)
                  .toList()
              : <String>[];
          // Accept a real JSON number or a stringified one ("2") — a small
          // quantized model quoting a number is a common enough formatting
          // slip. null/"null"/missing/anything else means "no contradiction".
          final contradictsRaw = decoded['contradicts_index'];
          final contradictsIndex = contradictsRaw is num
              ? contradictsRaw.toInt()
              : int.tryParse(contradictsRaw?.toString() ?? '');

          return (
            text: text,
            category: validCategories.contains(categoryRaw) ? categoryRaw! : 'general',
            valence: validValences.contains(valenceRaw) ? valenceRaw! : 'neutral',
            tags: tags,
            contradictsIndex: contradictsIndex,
          );
        }
      }
    } catch (_) {
      // Fall through to partial recovery below.
    }

    // Full parse failed — most likely the response got cut off mid-object
    // by the token cap before the closing brace. Try to salvage just the
    // "text" field via regex: if the model got that far before running out
    // of budget, the fact itself is usually still intact even though the
    // rest of the object (category/tags/etc) never got written.
    final match = _extractedTextFieldPattern.firstMatch(raw);
    if (match == null) return null;
    final recovered = match.group(1)?.trim() ?? '';
    if (recovered.isEmpty) return null;
    return (
      text: recovered,
      category: 'general',
      valence: 'neutral',
      tags: const [],
      contradictsIndex: null,
    );
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
