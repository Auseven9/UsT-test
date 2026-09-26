import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:llamadart/llamadart.dart';

import '../models/chat_model.dart';
import '../models/message_model.dart';
import '../models/memory_entry.dart';
import '../models/turn_telemetry.dart';
import '../services/llm_service.dart';
import '../services/chat_storage_service.dart';
import '../services/embedding_service.dart';
import '../services/helper_llm_service.dart';
import '../services/memory_service.dart';
import '../services/memory_heuristics.dart';
import '../services/reasoning_trace_service.dart';
import '../services/reminder_service.dart';
import '../services/uncertainty_heuristics.dart';
import '../services/tool_definitions.dart';
import '../services/log_service.dart';

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

/// The judgment-call portion of the memory-extraction prompt — what counts
/// as "worth remembering". Exposed (via [ChatStorageService.
/// memoryExtractionGuidance]) as a Settings-editable override, same tier as
/// the main chat system prompt, so a user who wants different judgment
/// (more/less eager, different categories emphasized) doesn't have to guess
/// blind at heuristic thresholds — they can just read and rewrite what the
/// extraction model is actually told. Public so the Settings screen can
/// show it as the placeholder/reset target.
const defaultMemoryExtractionGuidance =
    'You watch one exchange from a conversation and extract durable facts '
    'or preferences worth remembering in future, unrelated conversations — '
    'things like the user\'s name, stated preferences, ongoing projects, '
    'personal details, or something they explicitly asked to be remembered '
    '(if they said "remember that" or similar, figure out from the '
    'exchange what "that" refers to and capture the actual content, not '
    'the instruction itself). Only capture what the user themself stated '
    'about themself or their own life — never store something the '
    'assistant said about itself, and never attribute the assistant\'s own '
    'claims or opinions to the user.';

class ChatController extends GetxController {
  final LlmService _llm = Get.find<LlmService>();
  final ChatStorageService _storage = Get.find<ChatStorageService>();
  final EmbeddingService _embedding = Get.find<EmbeddingService>();
  final MemoryService _memory = Get.find<MemoryService>();
  final HelperLlmService _helper = Get.find<HelperLlmService>();
  final ReasoningTraceService _reasoningTraces = Get.find<ReasoningTraceService>();
  final ReminderService _reminders = Get.find<ReminderService>();

  // Rebuilt on every call rather than cached — `includeMemoryTools` must
  // track the user's live Persistent Memory setting, since a model still
  // holding a `search_memory` tool from before the user turned that setting
  // off would keep reading memories the toggle was meant to stop exposing.
  // Same reasoning for `includeAdvancedTools` and the Advanced Tools switch.
  List<ToolDefinition> get _tools => buildToolDefinitions(
        memory: _memory,
        embedding: _embedding,
        reasoningTraces: _reasoningTraces,
        reminders: _reminders,
        includeMemoryTools: _storage.persistentMemoryEnabled,
        includeAdvancedTools: _storage.advancedToolsEnabled,
      );

  LogService? get _log {
    try {
      return Get.find<LogService>();
    } catch (_) {
      return null;
    }
  }

  /// One short, always-current paragraph telling the model what it actually
  /// is and can do here — built from the real tool list rather than a
  /// hand-written description, so it can never drift out of sync with what
  /// tool_definitions.dart actually exposes. Without this, a model has no way
  /// to know it's running fully offline on a phone (not a hosted API), or
  /// which of its "I can't do X" instincts (inherited from training on a
  /// hosted assistant) are simply wrong here — e.g. claiming it has no way
  /// to know the date when get_current_datetime is one tool call away, or
  /// narrating a fake "saving to memory" action instead of trusting the real
  /// automatic capture described separately below.
  /// [tools] must be the exact same list actually passed to generation this
  /// turn (or null when Tools is off) — otherwise a user who has turned
  /// Tools off would still be told about tools that no schema was ever sent
  /// for, and the model either hallucinates a call that goes nowhere or
  /// claims a capability it doesn't have here.
  String _buildAppContextBlock(List<ToolDefinition>? tools) {
    final withOs = 'You are running as the on-device assistant inside '
        '"Uncensored Local AI" — a modified, uncensored local-inference '
        'app. There is no server: every reply is generated fully offline by '
        'a local model file on the user\'s own ${Platform.operatingSystem} '
        'device, and nothing leaves the device. You are not the only model '
        'in the app — a separate small embedding model handles memory '
        'search behind the scenes, and a separate small helper model may '
        'distill memories in the background; neither is something you call '
        'directly.';
    if (tools == null || tools.isEmpty) return withOs;
    final toolLines = tools.map((t) => '- ${t.name}: ${t.description}').join('\n');
    return '$withOs\nThe tools you can actually call in this conversation:\n$toolLines';
  }

  final chats = <ChatModel>[].obs;
  final activeChatId = RxnString();
  final isGenerating = false.obs;
  final streamedResponse = ''.obs;
  final streamedReasoning = ''.obs;
  final temperature = 0.7.obs;
  final systemPrompt = ''.obs;

  StreamSubscription<String>? _genSub;
  Timer? _memorySweepTimer;

  @override
  void onInit() {
    super.onInit();
    _loadChats();
    temperature.value = _storage.defaultTemperature;
    systemPrompt.value = _storage.globalSystemPrompt;
    _scheduleMemorySweep();
  }

  /// (Re)schedules the periodic memory-health sweep from the current
  /// tunable interval in Settings — call again after the user changes that
  /// setting so a new interval takes effect without an app restart.
  void _scheduleMemorySweep() {
    _memorySweepTimer?.cancel();
    final minutes = _storage.memorySweepIntervalMinutes;
    if (minutes <= 0) return;
    _memorySweepTimer = Timer.periodic(
      Duration(minutes: minutes),
      (_) => _confirmAndRunMemorySweep(),
    );
  }

  /// Public so Settings can call it right after the user changes the sweep
  /// interval, instead of the new value only taking effect on next launch.
  void rescheduleMemorySweep() => _scheduleMemorySweep();

  /// Asks the user before every sweep — the sweep now does a real probe
  /// generation call (see [_runMemorySweep]), not just a flag check, so it's
  /// no longer free enough to run silently on a timer without consent.
  /// Declining just skips this cycle; the next tick asks again.
  Future<void> _confirmAndRunMemorySweep() async {
    // Never interrupt an in-progress reply with an uninvited dialog — skip
    // this cycle silently rather than popping a barrier-blocking prompt
    // over an actively streaming response. The next tick tries again.
    if (isGenerating.value) {
      _log?.info('Memory sweep skipped — a reply is in progress.', source: 'Memory');
      return;
    }
    final accepted = await _askSweepConsent();
    if (!accepted) {
      _log?.info('Memory sweep declined for this cycle.', source: 'Memory');
      return;
    }
    // Re-check rather than trust the pre-dialog snapshot — up to 20s can
    // pass while the consent dialog sits open, plenty of time for the user
    // to have started (or resumed) a reply since the first check above.
    if (isGenerating.value) {
      _log?.info(
        'Memory sweep skipped — a reply started while the consent dialog was open.',
        source: 'Memory',
      );
      return;
    }
    await _runMemorySweep();
  }

  Future<bool> _askSweepConsent() async {
    final completer = Completer<bool>();
    var responded = false;
    void respond(bool value) {
      if (responded) return;
      responded = true;
      if (Get.isDialogOpen ?? false) Get.back();
      completer.complete(value);
    }

    try {
      unawaited(Get.dialog(
        AlertDialog(
          title: const Text('Run memory sweep?'),
          content: const Text(
            'Verifies each configured model is actually armed, probes the '
            'embedding model with a real request, and flushes any pending '
            'memory write to disk. If everything checks out, it also '
            'promotes or drops working-memory entries, rehearses the most '
            'important memories, and — if a helper model is armed — runs '
            'one more short generation to look for connections across '
            'recent memories, which can add a new note of its own.',
          ),
          actions: [
            TextButton(
              onPressed: () => respond(false),
              child: const Text('Not now'),
            ),
            ElevatedButton(
              onPressed: () => respond(true),
              child: const Text('Run sweep'),
            ),
          ],
        ),
        barrierDismissible: false,
      ));
    } catch (_) {
      // No overlay currently mounted (app backgrounded, no active route) —
      // skip this cycle rather than hang waiting for a dialog no one can see.
      return false;
    }
    // Don't leave a dialog sitting unanswered forever if the user's
    // attention is elsewhere — default to declined and dismiss it.
    Timer(const Duration(seconds: 20), () => respond(false));
    return completer.future;
  }

  /// "Is everything actually working" check, run only after the user
  /// accepts the prompt above. Verifies each model the app is configured to
  /// use is genuinely loaded (armed), does a real embedding call to confirm
  /// the embedding model actually responds (not just reports itself
  /// loaded), and flushes any pending debounced memory write to disk.
  Future<void> _runMemorySweep() async {
    _log?.info('Memory sweep starting...', source: 'Memory');
    final problems = <String>[];
    if (!_llm.isLoaded.value) problems.add('main model not armed');
    if (_storage.persistentMemoryEnabled) {
      if (!_embedding.isLoaded.value) {
        problems.add('embedding model not armed');
      } else {
        final probe = await _embedding.embed('memory sweep self-test');
        if (probe == null || probe.isEmpty) {
          problems.add('embedding model armed but not responding to requests');
        } else {
          _log?.info(
            'Embedding probe OK — ${probe.length}-dim vector returned.',
            source: 'Memory',
          );
        }
      }
      if (_storage.helperModelFilename.isNotEmpty && !_helper.isLoaded.value) {
        problems.add('helper model not armed (falling back to main model)');
      }
    }
    await _memory.flushPending();
    if (problems.isEmpty) {
      // Say only what was actually checked — with Persistent Memory off,
      // the whole embedding/helper block above never ran; and even with it
      // on, only the embedding model gets a real request/response probe —
      // the main and helper models are only confirmed *loaded*, not
      // exercised with a generation, so "armed and responding" would
      // overstate what this pass verified for those two.
      final helperNote = _storage.helperModelFilename.isEmpty
          ? 'no helper model configured'
          : 'helper model armed';
      final msg = _storage.persistentMemoryEnabled
          ? 'Memory sweep OK — main model armed, embedding model armed and '
              'responding to a real request, $helperNote, '
              '${_memory.entries.length} memories on disk.'
          : 'Memory sweep OK — main model armed (Persistent Memory is off, '
              'so memory models weren\'t checked).';
      // Log only — the user already opted into this exact run via the
      // consent dialog a moment ago, so a success toast on top of that
      // would be a second interruption for a result that isn't actionable.
      // A problem is still worth toasting since it needs attention.
      _log?.info(msg, source: 'Memory');
    } else {
      final msg = 'Memory sweep found issues: ${problems.join(', ')}.';
      _log?.warn(msg, source: 'Memory');
      _toast(msg);
    }
    // Only attempt consolidation once the sweep above confirms everything's
    // actually healthy — no point spending a generation looking for
    // connections across memory on top of a setup that's already known to
    // be broken.
    if (problems.isEmpty && _storage.persistentMemoryEnabled) {
      // Working-memory maintenance and rehearsal are both cheap (no
      // inference — maintenance is pure bookkeeping, rehearsal just
      // re-touches timestamps) so they run regardless of whether
      // consolidation itself (which does spend a generation) goes on to
      // find anything. Gated on the toggle like consolidation below —
      // with Persistent Memory off, nothing should touch the store even
      // if the sweep itself still ran with just the main model armed.
      final maintenance = await _memory.runWorkingMemoryMaintenance();
      if (maintenance.promoted > 0 || maintenance.dropped > 0) {
        _log?.info(
          'Working-memory maintenance: ${maintenance.promoted} promoted, '
          '${maintenance.dropped} dropped (never reinforced).',
          source: 'Memory',
        );
      }
      final rehearsed = await _memory.rehearseTopMemories();
      if (rehearsed.isNotEmpty) {
        _log?.info('Rehearsed ${rehearsed.length} memories.', source: 'Memory');
      }
      await _runMemoryConsolidation();
    }

    // Reasoning-trace lane maintenance runs on its own toggle, independent
    // of Persistent Memory — it's a different lane (see
    // ReasoningTraceService) and shouldn't wait on fact-memory health.
    if (_storage.reasoningTraceEnabled) {
      final traceMaintenance = await _reasoningTraces.runLaneMaintenance();
      if (traceMaintenance.compressed > 0 || traceMaintenance.dropped > 0) {
        _log?.info(
          'Reasoning-trace lane maintenance: ${traceMaintenance.compressed} '
          'compressed, ${traceMaintenance.dropped} dropped.',
          source: 'Reasoning',
        );
      }
    }
  }

  /// Looks at a handful of the most recently-touched memories together and
  /// asks the helper model whether any of them are genuinely connected —
  /// the same person/project mentioned in two separate notes, or one that
  /// adds context to another — writing a short new note capturing that
  /// link if so. This is the "the embedding model takes a little turn at
  /// the end of each sweep" consolidation pass: distinct from per-turn
  /// extraction (which only ever looks at one exchange in isolation and
  /// can't notice a connection spanning multiple past conversations).
  /// Bounded and cheap: one short generation, only ever run as part of an
  /// already-user-approved sweep, and skipped outright if there isn't
  /// enough in memory yet for a connection to even be possible.
  Future<void> _runMemoryConsolidation() async {
    if (!_storage.persistentMemoryEnabled ||
        !_embedding.isLoaded.value ||
        !_helper.isLoaded.value) {
      return;
    }
    // The helper engine has exactly one generation slot. If a per-turn
    // extraction call (see _extractAndRememberFromTurn) happens to be
    // in flight right now, calling complete() here would just get back a
    // silent null — worse, it would occupy the slot the moment the
    // in-flight call finishes, right as the NEXT turn's extraction might
    // want it. Defer entirely rather than contend for it; the next sweep
    // (only after the user approves it again) gets another chance.
    if (_helper.isBusy) {
      _log?.info('Memory consolidation skipped — helper model busy.', source: 'Memory');
      return;
    }
    final active = _memory.entries.where((e) => e.isActive).toList()
      ..sort((a, b) => b.lastAccessedAt.compareTo(a.lastAccessedAt));
    if (active.length < 3) return;

    final sample = active.take(6).toList();
    // Capped per entry, same as the extraction prompt's `capMemoryText` —
    // nothing here guarantees a stored memory stayed short (the enrichment
    // path in MemoryService.addIfNotDuplicate can grow one substantially),
    // and this listing has no other bound before it goes into a fixed
    // 2048-token helper context alongside the rest of the prompt.
    final listing = sample
        .indexed
        .map((e) => '${e.$1 + 1}. ${_capMemoryText(e.$2.text)}')
        .join('\n');

    // Unlike per-turn extraction (which paraphrases real verbatim exchange
    // text — low hallucination risk), this is a genuinely generative task:
    // asking a small model to freely describe a "connection" between
    // unrelated notes is exactly the kind of open-ended creative-writing
    // prompt small quantized models confabulate on, routinely inventing
    // names/details that appear nowhere in the source notes. Two guards
    // against that: (1) the model must cite which numbered notes it's
    // connecting, so a connection to nothing (or to notes that don't
    // exist) is a parseable, rejectable failure rather than free text; (2)
    // after embedding its own sentence, that embedding must actually sit
    // close to the notes it claims to connect — a fabricated addition
    // drifts away from its cited sources in embedding space even when the
    // wording sounds plausible.
    final request = <LlamaChatMessage>[
      LlamaChatMessage.fromText(
        role: LlamaChatRole.system,
        text: 'You review a handful of previously remembered notes and '
            'look for one genuine connection worth recording — e.g. two '
            'notes that are about the same person or project, or one that '
            'adds context to another. Only state something that is '
            'already explicitly present in the notes you cite — never '
            'invent a new name, date, number, or detail that isn\'t '
            'already written in them. Respond with ONLY a single-line '
            'JSON object, no other text: {"connection": "<one short '
            'sentence, or empty string if nothing connects>", "cites": '
            '[<numbers of the notes below it draws on, at least two, or '
            'empty if nothing connects>], "confidence": <0.0-1.0, how '
            'sure you are this is a real connection actually stated in '
            'those notes, not an inference or guess>}',
      ),
      LlamaChatMessage.fromText(role: LlamaChatRole.user, text: listing),
    ];

    String? raw;
    try {
      raw = await _helper.complete(request, maxTokens: 100).timeout(const Duration(seconds: 25));
    } on TimeoutException {
      _helper.stopGeneration();
      _log?.warn('Memory consolidation timed out.', source: 'Memory');
      return;
    }
    if (raw == null) return;
    final parsed = _parseConsolidationJson(raw.trim());
    if (parsed == null) {
      _log?.warn('Memory consolidation discarded: malformed JSON.', source: 'Memory');
      return;
    }
    final out = parsed.connection;
    if (out.isEmpty || MemoryHeuristics.isNoMemorySentinel(out)) {
      _log?.info('Memory consolidation: nothing to connect.', source: 'Memory');
      return;
    }
    final citedEntries = parsed.cites.toSet()
        .where((i) => i >= 1 && i <= sample.length)
        .map((i) => sample[i - 1])
        .toList();
    if (citedEntries.length < 2) {
      _log?.warn(
        'Memory consolidation discarded: cited fewer than two real notes.',
        source: 'Memory',
      );
      return;
    }
    const consolidationConfidenceThreshold = 0.6;
    if (parsed.confidence < consolidationConfidenceThreshold) {
      _log?.info(
        'Memory consolidation discarded: below confidence threshold '
        '(${parsed.confidence}).',
        source: 'Memory',
      );
      return;
    }

    final vector = await _embedding.embed(out);
    if (vector == null) return;

    // Grounding check: the new sentence's embedding must sit reasonably
    // close to at least one of the notes it claims to connect. A genuine
    // connecting sentence paraphrases content already in those notes, so it
    // stays near them in embedding space; a confabulated addition (a name
    // or detail invented rather than drawn from the notes) tends to drift
    // away from all of them even when the citation itself was honest about
    // which notes it meant to reference.
    const groundingThreshold = 0.35;
    final maxSimilarity = citedEntries
        .map((e) => _memory.cosineSimilarity(vector, e.embedding))
        .fold<double>(0.0, max);
    if (maxSimilarity < groundingThreshold) {
      _log?.warn(
        'Memory consolidation discarded: ungrounded (similarity '
        '${maxSimilarity.toStringAsFixed(2)} to its own cited notes) — '
        'likely a hallucinated connection: "$out"',
        source: 'Memory',
      );
      return;
    }

    final result = await _memory.addIfNotDuplicate(
      out,
      vector,
      category: 'summary',
      valence: 'neutral',
      tags: const ['consolidation'],
      // Only dedupe against other consolidation notes — a connecting
      // sentence is EXPECTED to score similar to the source memories it
      // connects (that's the whole point), so checking it against those
      // too would routinely misfire as "duplicate" and get it silently
      // discarded as a reinforcement of one of the facts it's summarizing
      // instead of stored as the new note it actually is.
      scope: (e) => e.tags.contains('consolidation'),
    );
    if (result.wasNew) {
      _log?.info('Memory consolidation wrote a new note: "$out"', source: 'Memory');
      _toast('Memory consolidated: ${_truncateForToast(out)}');
    } else if (result.wasEnriched) {
      _log?.info('Memory consolidation enriched an existing note: "$out"', source: 'Memory');
      _toast('Memory consolidated: ${_truncateForToast(out)}');
    } else {
      _log?.info('Memory consolidation: reinforced existing note.', source: 'Memory');
    }
  }

  /// Parses the consolidation prompt's `{"connection", "cites", "confidence"}`
  /// JSON — same lenient-but-safe shape as [_parseExtractionJson], but with
  /// no truncated-fragment fallback: an ungrounded free-text recovery would
  /// defeat the whole point of requiring citations, so a malformed response
  /// here is discarded outright rather than partially salvaged.
  ({String connection, List<int> cites, double confidence})?
      _parseConsolidationJson(String raw) {
    try {
      final start = raw.indexOf('{');
      final end = raw.lastIndexOf('}');
      if (start == -1 || end == -1 || end <= start) return null;
      final decoded = jsonDecode(raw.substring(start, end + 1));
      if (decoded is! Map) return null;

      final connection = (decoded['connection'] as Object?)?.toString().trim() ?? '';
      final citesRaw = decoded['cites'];
      final cites = citesRaw is List
          ? citesRaw
              .map((e) => e is num ? e.toInt() : int.tryParse(e.toString()))
              .whereType<int>()
              .toList()
          : <int>[];
      final confidenceRaw = decoded['confidence'];
      final confidence = (confidenceRaw is num
                  ? confidenceRaw.toDouble()
                  : double.tryParse(confidenceRaw?.toString() ?? ''))
              ?.clamp(0.0, 1.0) ??
          0.0;

      return (connection: connection, cites: cites, confidence: confidence);
    } catch (_) {
      return null;
    }
  }

  /// Best-effort toast — swallows the error if no overlay is currently
  /// mounted (e.g. app just backgrounded) since the matching log line above
  /// every call site already captured the same information durably.
  void _toast(String message) {
    try {
      Get.snackbar(
        'Memory',
        message,
        snackPosition: SnackPosition.BOTTOM,
        duration: const Duration(seconds: 3),
        margin: const EdgeInsets.all(12),
      );
    } catch (_) {}
  }

  /// Shared ellipsis-truncation for every "cap this before it goes
  /// somewhere length-sensitive" spot in this file — a toast (kept short:
  /// 60 chars), the exportable log (200: tool call args/results can carry
  /// arbitrary, potentially sensitive content — e.g. memory text echoed
  /// back by search_memory — straight into a buffer the user can copy/
  /// share wholesale from the Log screen), or a memory-extraction/
  /// consolidation prompt (150: these notes are themselves distilled to
  /// one short sentence, but capped defensively anyway since nothing
  /// guarantees an old one — especially one recovered via the truncated-
  /// JSON fallback elsewhere in this file, or grown by the enrichment path
  /// in MemoryService.addIfNotDuplicate — actually stayed short).
  String _cap(String s, int maxLength) =>
      s.length > maxLength ? '${s.substring(0, maxLength)}…' : s;

  String _truncateForToast(String s) => _cap(s, 60);
  String _capForLog(String s) => _cap(s, 200);
  String _capMemoryText(String s) => _cap(s, 150);

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

    // Computed once and reused both for the system prompt's tool listing
    // below and for what's actually handed to generation further down —
    // `_tools` rebuilds the whole ToolDefinition list (with closures) on
    // every access, so calling it twice per message would do that work
    // twice for no reason.
    final activeTools = _storage.toolsEnabled ? _tools : null;

    // Build message history as real chat messages — the effective system
    // prompt (per-chat override, else global) goes first so the model's own
    // Jinja chat template (read from the GGUF's tokenizer.chat_template) can
    // render it correctly for whatever family is loaded (Gemma, Qwen, Phi, …).
    var effectiveSystemPrompt =
        chat.systemPrompt.isNotEmpty ? chat.systemPrompt : systemPrompt.value;
    final appContextBlock = _buildAppContextBlock(activeTools);
    effectiveSystemPrompt = effectiveSystemPrompt.isEmpty
        ? appContextBlock
        : '$appContextBlock\n\n$effectiveSystemPrompt';

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
      // The model now has real memory-write tools (remember, update_memory,
      // supersede_memory — see tool_definitions.dart), on top of durable
      // facts still being captured automatically in the background after
      // every turn regardless. Both are real and can both fire on the same
      // fact — that's fine, MemoryService's near-duplicate/enrichment
      // logic dedupes it the same way it would any other repeat.
      effectiveSystemPrompt =
          '$effectiveSystemPrompt\n\nDurable facts and preferences are also '
          'captured automatically in the background after each message, '
          'independent of anything you do — you don\'t need to call '
          'remember for something to be saved, but you can, especially '
          'for a correction (supersede_memory) or something you want '
          'saved with a specific wording right now.';

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
    // Must exceed the worst case of everything that can now legitimately
    // occupy the main engine in the background, one after another, when no
    // helper model is armed: extraction (up to two 20s attempts), then
    // self-critique (20s), then reasoning-trace distillation (20s) — all
    // serialized through the same queue (see _backgroundCompletionChain),
    // so they add up rather than overlap. Giving up before that queue can
    // possibly drain would throw the exact "Another generation is already
    // in progress" error this wait loop exists to avoid.
    while (_llm.isGenerating.value && waitedMs < 90000) {
      await Future.delayed(const Duration(milliseconds: 100));
      waitedMs += 100;
    }

    // Start generation
    isGenerating.value = true;
    streamedResponse.value = '';
    streamedReasoning.value = '';

    final aiMsg = MessageModel(role: MessageRole.assistant, content: '');
    aiMsg.telemetry = TurnTelemetry()
      ..setContextUsage(fitResult.historyTokens, fitResult.contextSize);
    chat.messages.add(aiMsg);
    chats.refresh();

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
          aiMsg.telemetry?.setContextUsage(refit.historyTokens, refit.contextSize);
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
          _log?.info('Tool call: ${c.name}(${_capForLog(c.rawArgs)})', source: 'Tools');
          Object? result;
          try {
            result = tool == null
                ? {'error': 'Unknown tool "${c.name}"'}
                : await tool.invoke(c.args);
            _log?.info(
              'Tool result: ${c.name} -> ${_capForLog(result.toString())}',
              source: 'Tools',
            );
            // A tool can fail internally without throwing (e.g. `calculate`
            // catches a malformed expression and returns {'error': ...}
            // rather than letting the exception propagate) — telemetry
            // needs to reflect that as a failure too, not just "no tool
            // found" or "threw", or the insight panel's tool chip would
            // show green for a call the model was actually told failed.
            final succeeded = tool != null &&
                !(result is Map && result.containsKey('error'));
            aiMsg.telemetry?.addToolCall(c.name, succeeded: succeeded);
          } catch (e) {
            result = {'error': e.toString()};
            _log?.error('Tool failed: ${c.name} -> $e', source: 'Tools');
            aiMsg.telemetry?.addToolCall(c.name, succeeded: false);
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
        aiMsg.telemetry?.setContextUsage(refit.historyTokens, refit.contextSize);
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
      // Two distinct failure shapes for a model whose chat template
      // llamadart doesn't cleanly recognize (custom/community fine-tunes —
      // e.g. Alesis's malformed `<|channel>`/`<channel|>` markers, missing
      // a pipe on one side of the real `<|channel|>` form): either the
      // *content* channel ends up holding everything (native `thinking`
      // stayed empty), or — worse, and previously unhandled — the
      // *thinking* channel ends up holding everything INCLUDING the real
      // final answer, leaving `content` completely empty. The second shape
      // is why a reply could show its whole answer trapped inside the
      // collapsed Thoughts panel with the raw markers still visible: this
      // fallback only ever checked the first shape.
      final rawReasoning = aiMsg.reasoning;
      if (rawReasoning != null &&
          rawReasoning.isNotEmpty &&
          aiMsg.content.trim().isEmpty &&
          LlmService.containsLeakedControlTokens(rawReasoning)) {
        final split = LlmService.splitLeakedControlTokens(
          rawReasoning,
          wasTruncated: _llm.lastGenerationHitTokenCap.value,
          assumeWholeTextIsAnswerIfNothingTrails: false,
        );
        aiMsg.reasoning = split.reasoning;
        // An empty answer here can mean two different things, unlike the
        // content-based fallback below: generation genuinely got cut off
        // mid-thought (wasTruncated), OR generation finished normally but
        // nothing recognizable as a distinct final answer followed the
        // last marker in the reasoning text (assumeWholeTextIsAnswerIfNothingTrails:
        // false always returns empty in that case, truncated or not) — the
        // model wasn't cut off, its answer is just unrecoverable from the
        // malformed output. Claiming "cut off" for the second case would be
        // a wrong diagnosis, not just an unhelpful one.
        aiMsg.content = split.answer.isNotEmpty
            ? split.answer
            : (_llm.lastGenerationHitTokenCap.value
                ? '_(cut off before finishing — see Thoughts above)_'
                : '_(no distinct answer found — see Thoughts above)_');
      } else if ((aiMsg.reasoning == null || aiMsg.reasoning!.isEmpty) &&
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

      // Prefer the real, measured signal (per-token logit confidence)
      // when the engine can report it — falls back to the surface-
      // language heuristic only when it can't (older llamadart, no model
      // loaded, nothing generated). See UncertaintyHeuristics' own doc for
      // why the heuristic exists as a fallback at all.
      final realConfidence = await _llm.getResponseConfidence();
      final uncertainty = realConfidence != null
          ? (1.0 - realConfidence).clamp(0.0, 1.0)
          : UncertaintyHeuristics.hedgeScore(aiMsg.content);
      aiMsg.telemetry?.setUncertainty(uncertainty, isHeuristic: realConfidence == null);
    }

    // Extraction runs after every non-trivial exchange (both the user's
    // message and the model's reply are handed to it together) — only a
    // `looksTrivial` filter (single-word acknowledgements, greetings) skips
    // it, to avoid burning a background generation on "ok"/"thanks"/"hi"
    // with nothing durable in them. This used to be gated much tighter, on
    // `looksMemorable` (explicit "remember this", first-person identity
    // statements, etc.) — cheaper, but it meant most exchanges were never
    // even considered for memory, which missed real, non-trigger-worded
    // facts the model volunteered mid-conversation. Storing the raw
    // *trigger* text verbatim was also wrong on its own terms: "remember
    // that." said right after the model explained its persona would get
    // stored as literally the three words "remember that." — the thing
    // worth remembering was the preceding content, not the command pointing
    // at it. So a short bounded extraction call reads the actual exchange
    // and distills what's durable, the same judgment call a human would
    // need to make when "that" doesn't name itself.
    //
    // Running literally every turn is only safe with a dedicated helper
    // model loaded — it runs in its own engine, so it can't contend with the
    // main model's single generation slot. Without one, extraction falls
    // back to running ON the main model (see below), and firing that on
    // nearly every turn would mean the *next* message routinely waits out a
    // background generation before its own reply can even start — so
    // without a helper, stay on the tighter, trigger-worded `looksMemorable`
    // gate instead. With a helper armed, this skips nothing at all,
    // including single-word turns — the explicit ask was that extraction
    // must run every single cycle, not just non-trivial ones, so a real
    // memorable fact volunteered right after a trivial acknowledgement is
    // never the one turn that got skipped.
    final canExtractEveryTurn = _helper.isLoaded.value;
    final worthExtracting =
        canExtractEveryTurn || MemoryHeuristics.looksMemorable(trimmedText);
    if (_storage.persistentMemoryEnabled && _embedding.isLoaded.value) {
      if (worthExtracting) {
        unawaited(_extractAndRememberFromTurn(
          userMsg,
          aiMsg,
          chat.id,
          relevantMemoriesForTurn,
        ));
      } else {
        _log?.info(
          'Memory extraction skipped this turn: no helper model armed and '
          'nothing matched the memorable-turn heuristic.',
          source: 'Memory',
        );
      }
    }

    if (_storage.selfCritiqueEnabled) {
      unawaited(_runSelfCritique(userMsg, aiMsg, relevantMemoriesForTurn));
    }
    if (_storage.reasoningTraceEnabled &&
        aiMsg.reasoning != null &&
        aiMsg.reasoning!.trim().isNotEmpty) {
      unawaited(_distillReasoningTrace(aiMsg.reasoning!, chat.id));
    }

    final dueReminders = await _reminders.checkDue();
    for (final reminder in dueReminders) {
      _toast('Reminder: ${reminder.text}');
      _log?.info('Reminder due: "${reminder.text}"', source: 'Reminders');
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
        // n-gram self-speculative: drafts from tokens already in this
        // turn's own context, no second model involved — the backend
        // verifies the draft in one batch and only keeps what the real
        // model would have produced anyway, so output is identical either
        // way; this only affects speed. Off by default until measured on
        // a real device (see Settings > Hardware Configuration), same
        // "don't trust a default that hasn't been benchmarked" principle
        // as the GPU backend story.
        speculativeDecodingConfig: _storage.speculativeDecodingEnabled
            ? const SpeculativeDecodingConfig.ngramSimple()
            : null,
      ),
      enableThinking: _storage.enableModelThinking,
      tools: tools,
    );

    List<LlamaCompletionChunkToolCall>? toolCalls;
    // Real wall-clock spacing between chunks as they actually arrive — the
    // cadence spectrograph's only data source. Reset per call (per round),
    // same scope as _llm.lastGenerationTokens below.
    DateTime? lastChunkAt;
    await for (final chunk in stream) {
      final now = DateTime.now();
      if (chunk.thinking.isNotEmpty) {
        if (separatorBox[1]) {
          streamedReasoning.value += '\n\n---\n\n';
          separatorBox[1] = false;
        }
        streamedReasoning.value += chunk.thinking;
        aiMsg.reasoning = streamedReasoning.value;
        aiMsg.telemetry?.addCadenceSample(
          true,
          lastChunkAt == null ? 0 : now.difference(lastChunkAt).inMicroseconds / 1000,
        );
        lastChunkAt = now;
      }
      if (chunk.content.isNotEmpty) {
        if (separatorBox[0]) {
          streamedResponse.value += '\n\n';
          separatorBox[0] = false;
        }
        streamedResponse.value += chunk.content;
        aiMsg.content = streamedResponse.value;
        aiMsg.telemetry?.addCadenceSample(
          false,
          lastChunkAt == null ? 0 : now.difference(lastChunkAt).inMicroseconds / 1000,
        );
        lastChunkAt = now;
      }
      if (chunk.toolCalls != null && chunk.toolCalls!.isNotEmpty) {
        toolCalls = chunk.toolCalls;
      }
      chats.refresh();
    }
    // A real per-round token count, read from the same counter
    // generateChatCompletion just finished updating for exactly this call —
    // not an estimate, the actual count llama.cpp reported for this round.
    aiMsg.telemetry?.addRoundTokens(_llm.lastGenerationTokens.value);
    return toolCalls;
  }

  /// A small, fixed set of exchanges with known-correct extraction outcomes
  /// — including one negative case (nothing durable should be captured) —
  /// run through the real extraction pipeline in dry-run mode (see
  /// [_extractAndRememberFromTurn]'s `onDryRunResult`) so this tests what
  /// actually runs on a real turn, not a separate reimplementation that
  /// could silently drift from it. Nothing here writes to memory. Exists
  /// because nothing else in the app measures whether extraction is
  /// actually working — the log screen shows what happened on real turns,
  /// but not whether the current helper model / extraction guidance is
  /// any good at the job in the first place.
  Future<({int passed, int total, List<({String description, bool passed, String actual})> details})>
      runMemoryVerificationSuite() async {
    final cases = <({String user, String ai, String description, bool Function(String) check})>[
      (
        user: 'My name is Jordan and I live in Denver.',
        ai: 'Nice to meet you, Jordan! Denver is a great city.',
        description: 'Captures a stated name',
        check: (text) => text.toLowerCase().contains('jordan'),
      ),
      (
        user: 'I prefer dark roast coffee over light roast.',
        ai: 'Noted — dark roast it is.',
        description: 'Captures a stated preference',
        check: (text) =>
            text.toLowerCase().contains('dark roast') || text.toLowerCase().contains('coffee'),
      ),
      (
        user: "Remember that my dog's name is Biscuit.",
        ai: 'Got it, Biscuit!',
        description: 'Captures an explicit "remember that"',
        check: (text) => text.toLowerCase().contains('biscuit'),
      ),
      (
        user: 'I used to work at Acme Corp but now I work at Globex.',
        ai: 'Congrats on the new job at Globex!',
        description: 'Captures the current fact, not the outdated one',
        check: (text) =>
            text.toLowerCase().contains('globex') && !text.toLowerCase().contains('acme'),
      ),
      (
        user: "What's 2+2?",
        ai: '4.',
        description: 'Correctly finds nothing durable (negative case)',
        check: (text) => MemoryHeuristics.isNoMemorySentinel(text),
      ),
    ];

    // Fail fast rather than burning the full timeout on every single case
    // below — _extractAndRememberFromTurn itself bails out immediately
    // (never calling onDryRunResult) when neither model is loaded, and
    // that outcome is knowable up front here.
    if (!_helper.isLoaded.value && !_llm.isLoaded.value) {
      return (
        passed: 0,
        total: cases.length,
        details: [
          for (final c in cases)
            (description: c.description, passed: false, actual: '(no model loaded)'),
        ],
      );
    }

    final details = <({String description, bool passed, String actual})>[];
    for (final c in cases) {
      final userMsg = MessageModel(role: MessageRole.user, content: c.user);
      final aiMsg = MessageModel(role: MessageRole.assistant, content: c.ai);
      String actual = '(no result)';
      var passed = false;
      final completer = Completer<void>();
      unawaited(_extractAndRememberFromTurn(
        userMsg,
        aiMsg,
        'verification',
        const [],
        onDryRunResult: (text, category, valence, confidence) {
          actual = text.isEmpty ? '(empty)' : text;
          passed = c.check(text);
          if (!completer.isCompleted) completer.complete();
        },
      ));
      // Must exceed extraction's real worst case, not just its per-attempt
      // timeout: a malformed-JSON response triggers one retry (see the
      // attempt loop above), so the live pipeline can legitimately take up
      // to two attempts of (3s engine-slot wait + 20s generation) = 46s
      // before onDryRunResult ever fires. Must clear that, not just match
      // it — 45s was still 1s short and could false-negative a case the
      // real pipeline would have gotten right on retry.
      await completer.future.timeout(
        const Duration(seconds: 50),
        onTimeout: () => actual = '(timed out)',
      );
      details.add((description: c.description, passed: passed, actual: actual));
    }

    final passedCount = details.where((d) => d.passed).length;
    _log?.info(
      'Memory verification: $passedCount/${details.length} passed.',
      source: 'Memory',
    );
    return (passed: passedCount, total: details.length, details: details);
  }

  /// Serializes calls to [_runBackgroundCompletion] — without this, two
  /// independent background tasks (self-critique and reasoning-trace
  /// distillation can both fire off the same turn) racing to fall back to
  /// the single-slot main engine could both observe `isGenerating == false`
  /// before either sets it, and the second one's `generateChatCompletion`
  /// call throws a `StateError` instead of returning null like every other
  /// failure mode here. Chaining through one Future guarantees only one
  /// background completion — helper or main-fallback — runs at a time.
  Future<void> _backgroundCompletionChain = Future.value();

  /// Runs one short, bounded background generation — preferring the
  /// dedicated helper engine when armed, falling back to the main model's
  /// engine (waiting briefly for its generation slot, skipping rather than
  /// blocking indefinitely if it stays busy) when not. This is the same
  /// dual-engine fallback [_extractAndRememberFromTurn] uses for memory
  /// extraction, factored out so other background tasks (self-critique,
  /// reasoning-trace distillation) don't each reimplement the fallback,
  /// timeout, and main-engine-stat-preservation logic. Returns null on any
  /// failure — no model armed, busy too long, a timeout, or a race against
  /// another caller (see [_backgroundCompletionChain]).
  Future<String?> _runBackgroundCompletion(
    List<LlamaChatMessage> messages, {
    required int maxTokens,
    double temp = 0.2,
    Duration timeout = const Duration(seconds: 20),
  }) async {
    final completer = Completer<String?>();
    _backgroundCompletionChain = _backgroundCompletionChain.then((_) async {
      try {
        completer.complete(
          await _runBackgroundCompletionInner(messages, maxTokens: maxTokens, temp: temp, timeout: timeout),
        );
      } catch (e) {
        // A background task's failure must never propagate out of this
        // serialization chain — that would poison every later queued call.
        completer.complete(null);
      }
    });
    return completer.future;
  }

  Future<String?> _runBackgroundCompletionInner(
    List<LlamaChatMessage> messages, {
    required int maxTokens,
    double temp = 0.2,
    Duration timeout = const Duration(seconds: 20),
  }) async {
    final useHelper = _helper.isLoaded.value;
    if (!useHelper && !_llm.isLoaded.value) return null;

    if (useHelper) {
      try {
        return await _helper
            .complete(messages, maxTokens: maxTokens, temp: temp)
            .timeout(timeout);
      } on TimeoutException {
        _helper.stopGeneration();
        return null;
      }
    }

    var waitedMs = 0;
    while (_llm.isGenerating.value && waitedMs < 3000) {
      await Future.delayed(const Duration(milliseconds: 100));
      waitedMs += 100;
    }
    if (_llm.isGenerating.value) return null;

    // Reusing the main engine means its shared stats fields (read directly
    // by chat_bubble.dart/home_screen.dart for the "t/s" readout on the
    // user's actual last reply) would otherwise get silently overwritten
    // with this background call's numbers — snapshot and restore them so
    // the visible reply stats stay put.
    final savedSpeed = _llm.lastGenerationSpeed.value;
    final savedTokens = _llm.lastGenerationTokens.value;
    final savedHitCap = _llm.lastGenerationHitTokenCap.value;

    final buffer = StringBuffer();
    try {
      final stream = _llm
          .generateChatCompletion(
            messages: messages,
            params: GenerationParams(temp: temp, maxTokens: maxTokens),
            enableThinking: false,
          )
          .timeout(timeout);
      await for (final chunk in stream) {
        buffer.write(chunk.content);
      }
      return buffer.toString();
    } on TimeoutException {
      await _llm.stopGeneration();
      return null;
    } on StateError {
      // The main engine started a real (non-background) generation in the
      // narrow window between the isGenerating check above and this call —
      // e.g. the user sent a new message right as this background task's
      // turn came up in the serialization queue. Skip rather than crash.
      return null;
    } finally {
      _llm.lastGenerationSpeed.value = savedSpeed;
      _llm.lastGenerationTokens.value = savedTokens;
      _llm.lastGenerationHitTokenCap.value = savedHitCap;
    }
  }

  /// Second-opinion pass over an already-given answer: a bounded background
  /// generation reads the exchange (plus whatever memories were already
  /// retrieved for this turn, at no extra retrieval cost) and checks for a
  /// direct contradiction with a remembered fact, or a confident-sounding
  /// claim that isn't actually backed by anything in the exchange. This
  /// never edits or blocks the answer already shown — it's purely
  /// after-the-fact, surfaced as a soft note in the turn's telemetry (and a
  /// toast if it found something), the evaluation loop this app never had
  /// before. Best-effort and silent on failure, same as memory extraction.
  Future<void> _runSelfCritique(
    MessageModel userMsg,
    MessageModel aiMsg,
    List<MemoryEntry> relevantMemories,
  ) async {
    if (aiMsg.content.trim().isEmpty) return;
    const maxChars = 600;
    String cap(String s) => s.length > maxChars ? '${s.substring(0, maxChars)}…' : s;

    final memoryContext = relevantMemories.isEmpty
        ? ''
        : '\n\nRemembered facts relevant to this exchange:\n'
            '${relevantMemories.map((e) => '- ${cap(e.text)}').join('\n')}';

    final request = <LlamaChatMessage>[
      LlamaChatMessage.fromText(
        role: LlamaChatRole.system,
        text: 'You are reviewing an AI assistant\'s answer for two things '
            'only: (1) does it directly contradict a remembered fact listed '
            'below, and (2) does it state something as settled fact that '
            'the exchange itself gives no real basis for (a confident '
            'guess). Do not comment on style, tone, or anything else. If '
            'neither problem is present, say so plainly.$memoryContext\n\n'
            'Respond with ONLY a single-line JSON object, no other text: '
            '{"flagged": <true|false>, "issue": "<one short sentence '
            'describing the problem, or empty string if none>"}',
      ),
      LlamaChatMessage.fromText(role: LlamaChatRole.user, text: cap(userMsg.content)),
      LlamaChatMessage.fromText(role: LlamaChatRole.assistant, text: cap(aiMsg.content)),
    ];

    final result = await _runBackgroundCompletion(request, maxTokens: 100);
    if (result == null) return;

    try {
      final jsonStart = result.indexOf('{');
      final jsonEnd = result.lastIndexOf('}');
      if (jsonStart == -1 || jsonEnd == -1 || jsonEnd < jsonStart) return;
      final decoded =
          jsonDecode(result.substring(jsonStart, jsonEnd + 1)) as Map<String, dynamic>;
      final flagged = decoded['flagged'] == true;
      final issue = (decoded['issue'] as String? ?? '').trim();
      aiMsg.telemetry?.setCritique(flagged: flagged, note: issue.isEmpty ? null : issue);
      if (flagged && issue.isNotEmpty) {
        _log?.warn('Self-critique flagged this answer: $issue', source: 'Critique');
        _toast('Self-check: $issue');
      } else {
        _log?.info('Self-critique found no issue.', source: 'Critique');
      }
    } catch (_) {
      // Malformed output from a small model — silently skip rather than
      // showing a misleading or garbled note.
    }
  }

  /// Distills [reasoning] (a turn's raw chain-of-thought) down to a short
  /// gist and appends it to the reasoning-trace lane, separate from fact
  /// memory — see ReasoningTraceService's own doc for why. Falls back to a
  /// plain truncation (no model call) when neither engine is available,
  /// same "works with or without a helper" principle as every other
  /// background feature here — this lane keeps filling either way, just
  /// with a cruder gist when nothing is armed to summarize it properly.
  Future<void> _distillReasoningTrace(String reasoning, String chatId) async {
    const rawCap = 1200;
    final capped =
        reasoning.length > rawCap ? '${reasoning.substring(0, rawCap)}…' : reasoning;

    final request = <LlamaChatMessage>[
      LlamaChatMessage.fromText(
        role: LlamaChatRole.system,
        text: 'Summarize the following reasoning in one short sentence — '
            'the gist of the approach taken, not the full detail. Respond '
            'with ONLY that one sentence, nothing else.',
      ),
      LlamaChatMessage.fromText(role: LlamaChatRole.user, text: capped),
    ];

    final result = await _runBackgroundCompletion(request, maxTokens: 60);
    final summary = (result == null || result.trim().isEmpty)
        ? (capped.length > 200 ? '${capped.substring(0, 200)}…' : capped)
        : result.trim();

    await _reasoningTraces.add(chatId, summary);
  }

  /// Looks at the latest user+assistant exchange — already flagged as worth
  /// capturing by the gate in [sendMessage] (every non-trivial turn when a
  /// helper model is armed, or just [MemoryHeuristics.looksMemorable] turns
  /// otherwise) — and asks the model to distill anything durable, tagged
  /// with category/valence/tags, so something like "remember that" (which
  /// names nothing on its own) correctly captures whatever it was pointing
  /// at instead of being stored as its own three words. Also given
  /// [relevantMemories] — the same memories already retrieved for this
  /// turn's system prompt, at no extra retrieval cost — so it can flag when
  /// the new fact contradicts one of them (an updated fact), in which case
  /// the old one is marked superseded rather than left to compete equally
  /// with the current one. Bounded and cheap (short output, low temp).
  /// Best-effort throughout; any failure here is silent since it must never
  /// surface as a chat error, and a small model producing malformed output
  /// degrades to
  /// the plain-text behavior this replaced rather than losing the memory.
  Future<void> _extractAndRememberFromTurn(
    MessageModel userMsg,
    MessageModel aiMsg,
    String chatId,
    List<MemoryEntry> relevantMemories, {
    // When set, this runs the real extraction pipeline (prompt, generation,
    // parsing) exactly as normal but stops before touching memory or the
    // UI — the parsed result goes to this callback instead. Used by the
    // verification harness ([runMemoryVerificationSuite]) so it tests the
    // actual live pipeline, not a reimplementation of it that could drift
    // out of sync with what real turns actually run.
    void Function(String noteText, String category, String valence, double confidence)?
        onDryRunResult,
  }) async {
    try {
      final useHelper = _helper.isLoaded.value;
      if (!useHelper && !_llm.isLoaded.value) {
        _log?.warn(
          'Memory extraction skipped: no helper model loaded and no main '
          'model loaded either.',
          source: 'Memory',
        );
        return;
      }
      _log?.info(
        'Memory extraction triggered (using ${useHelper ? 'helper' : 'main'} '
        'model).',
        source: 'Memory',
      );

      // A bit more room than a one-sentence extraction needed, to fit the
      // JSON structure below — bumped from 130 when entity_name/entity_type/
      // location/participants/connection were added alongside the original
      // text/category/valence/tags fields, since that's meaningfully more
      // JSON for the same token budget to cover.
      const extractionMaxTokens = 220;

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

      // Computed here (rather than where it's used, below) because its
      // length feeds directly into the overhead estimate right after —
      // this is now a Settings-editable field with no length limit at the
      // storage layer, so a long custom override has to actually be
      // accounted for, not just the roughly-fixed size of the built-in
      // default this replaced.
      final customGuidance = _storage.memoryExtractionGuidance.trim();
      final guidance =
          customGuidance.isEmpty ? defaultMemoryExtractionGuidance : customGuidance;

      // JSON-format instructions + trailing overhead is roughly fixed
      // regardless of guidance (~280 tokens now that the schema also
      // describes entity_name/entity_type/location/participants/
      // connection, not just text/category/valence/tags); the guidance
      // text itself is added on top since it's no longer a fixed size now
      // that it's user-editable — a long custom override correctly eats
      // into the budget left for the actual exchange instead of silently
      // overflowing it.
      const baseOverheadTokens = 280;
      final guidanceTokens = (guidance.length / approxCharsPerToken).ceil();
      final overheadTokens = baseOverheadTokens + guidanceTokens;
      final availableTokens =
          (extractionContextEstimate - extractionMaxTokens - overheadTokens)
              .clamp(40, extractionContextEstimate);
      final maxCharsEach =
          (availableTokens * approxCharsPerToken / 2).round().clamp(100, 4000);
      String cap(String s) =>
          s.length > maxCharsEach ? '${s.substring(0, maxCharsEach)}…' : s;

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
              'a changed fact):\n${relevantMemories.indexed.map((e) => '${e.$1 + 1}. ${_capMemoryText(e.$2.text)}').join('\n')}';

      // The exchange is folded into ONE user-role message (labeled
      // "[USER]"/"[ASSISTANT]" lines), never fed as real user/assistant chat
      // turns. Feeding it as actual turns — a user turn, then an assistant
      // turn holding aiMsg's own text — puts the model "in character" as
      // the assistant that just spoke, continuing that same dialogue: small
      // quantized models (Phi, Gemma-2-2B both showed this) then blur first-
      // person ("I") between the two speakers when asked to describe the
      // exchange afterward, mislabeling who said what in the extracted note.
      // Collapsing to a single labeled block plus an explicit "you are an
      // outside observer, never a participant" instruction removes the fake
      // assistant turn entirely, so there's nothing for the model to
      // "continue being" — it only ever sees itself as a note-taker.
      final extractionRequest = <LlamaChatMessage>[
        LlamaChatMessage.fromText(
          role: LlamaChatRole.system,
          text: '$guidance'
              '$existingMemoryContext\n\n'
              'Respond with ONLY a single-line JSON object, no other text: '
              '{"text": "<one short sentence, or empty string if nothing '
              'durable>", "category": "<fact|preference|event|instruction|'
              'general>", "valence": "<positive|negative|neutral>", '
              '"tags": ["<a few short keywords>"], "entity_name": "<the '
              'one named person, place, project, or organization this is '
              'mainly about, or empty string if there isn\'t one>", '
              '"entity_type": "<person|place|project|organization|event|'
              'idea|none>", "location": "<a place named in this, or empty '
              'string>", "participants": ["<other people/entities named '
              'alongside entity_name, if any>"], "connection": "<how '
              'entity_name relates to something else already known, in a '
              'few words, or empty string if not stated>", '
              '"contradicts_index": <the number of the already-remembered '
              'item above that this replaces/updates, or null if none>, '
              '"confidence": <0.0-1.0, how sure you are this is actually '
              'true and durable — use a low number for a guess or '
              'something ambiguous, a high number only when it was stated '
              'plainly>}',
        ),
        LlamaChatMessage.fromText(
          role: LlamaChatRole.user,
          text: 'You are an outside note-taker reviewing a transcript below. '
              'You did NOT take part in it — you are neither speaker. Never '
              'write in first person ("I", "me", "my"); always say '
              '"the user" or "the assistant" explicitly when it matters who '
              'said what, and never swap the two.\n\n'
              'Transcript:\n'
              '[USER]: ${cap(userMsg.content)}\n'
              '[ASSISTANT]: ${cap(aiMsg.content)}\n\n'
              'Extract anything worth remembering from the transcript above '
              'as that JSON object, and nothing else.',
        ),
      ];

      String? extracted;
      ({
        String text,
        String category,
        String valence,
        List<String> tags,
        int? contradictsIndex,
        double confidence,
        String entityName,
        String entityType,
        String location,
        List<String> participants,
        String connection,
      })? parsed;

      // Up to two attempts: a small quantized model producing malformed
      // JSON is common enough that one retry meaningfully improves the hit
      // rate, and this whole call is already bounded (20s timeout each)
      // and only runs on turns already gated as worth extracting, so the
      // extra attempt is bounded cost, not unbounded retrying.
      for (var attempt = 0; attempt < 2; attempt++) {
        // Routed through _runBackgroundCompletion (helper engine when
        // armed, else the main engine's single slot) — the same
        // serialization chain self-critique and reasoning-trace
        // distillation use, so this can no longer race either of them for
        // the main engine and throw an uncaught StateError. Previously
        // this attempt loop hand-rolled its own copy of the exact same
        // fallback/timeout/stat-preservation logic, independent of that
        // chain — that duplication was itself the bug.
        final attemptResult = await _runBackgroundCompletion(
          extractionRequest,
          maxTokens: extractionMaxTokens,
        );

        if (attemptResult == null) {
          _log?.warn('Memory extraction returned nothing (timed out or empty).',
              source: 'Memory');
          return;
        }
        extracted = attemptResult.trim();
        parsed = _parseExtractionJson(extracted);
        // A response that starts with '{' was clearly attempting JSON (the
        // prompt asks for nothing else) — if full parsing still failed,
        // it's a broken fragment, not a natural sentence, so it must never
        // be stored verbatim (braces, quotes, field names and all) as if
        // it were the memory itself. _parseExtractionJson already tries a
        // regex-based partial recovery of just the "text" field for
        // exactly this truncated-mid-object case; if even that comes back
        // empty, retry once (small quantized models are inconsistent
        // turn-to-turn), then give up.
        final looksLikeBrokenJson = parsed == null && extracted.trimLeft().startsWith('{');
        if (!looksLikeBrokenJson) break;
        if (attempt == 0) {
          _log?.warn('Memory extraction got malformed JSON, retrying once.',
              source: 'Memory');
          continue;
        }
        _log?.warn(
          'Memory extraction discarded after retry: model produced malformed '
          'JSON: ${extracted.length > 200 ? '${extracted.substring(0, 200)}…' : extracted}',
          source: 'Memory',
        );
        return;
      }

      final noteText = parsed?.text ?? extracted ?? '';
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
      final confidence = parsed?.confidence ?? 1.0;
      final entityName = parsed?.entityName ?? '';
      final entityType = parsed?.entityType ?? 'none';
      final entityLocation = parsed?.location ?? '';
      final participants = parsed?.participants ?? const <String>[];
      final connection = parsed?.connection ?? '';

      if (onDryRunResult != null) {
        onDryRunResult(noteText, category, valence, confidence);
        return;
      }

      // Whether JSON parsed or not, the "nothing durable" check still runs
      // on the actual candidate text — a malformed-JSON model that still
      // wrote "NONE" as its `text` field (or the whole raw response, if
      // parsing failed outright) is still correctly recognized as empty.
      if (MemoryHeuristics.isNoMemorySentinel(noteText)) {
        _log?.info(
          'Memory extraction found nothing durable in this turn.',
          source: 'Memory',
        );
        return;
      }

      final vector = await _embedding.embed(noteText);
      if (vector == null) {
        _log?.warn('Memory extraction discarded: embedding failed for "$noteText".',
            source: 'Memory');
        return;
      }
      if (contradictedMemory != null) {
        // Never goes through addIfNotDuplicate's near-duplicate check — a
        // corrected fact (e.g. "favorite color is green" replacing "...is
        // blue") can easily stay above the duplicate-similarity threshold
        // against the OLD entry it's meant to replace, since the two
        // sentences are structurally almost identical, which would
        // silently reinforce the wrong (outdated) memory instead of
        // recording the correction at all. resolveContradiction itself now
        // branches on confidence: a confident correction supersedes the
        // old fact outright (the previous unconditional behavior here); a
        // low-confidence one is kept alongside it instead, both active and
        // tagged 'conflicting', until something resolves which is right.
        final resolved = await _memory.resolveContradiction(
          oldId: contradictedMemory.id,
          newText: noteText,
          newEmbedding: vector,
          confidence: confidence,
          category: category,
          valence: valence,
          tags: tags,
          sourceChatId: chatId,
          entityName: entityName,
          entityType: entityType,
          location: entityLocation,
          participants: participants,
          connection: connection,
        );
        if (!resolved) {
          // The memory being corrected was deleted out from under this
          // in-flight extraction (e.g. from the Memory screen) between
          // when it was matched and when this awaited call landed —
          // nothing to correct anymore, so say nothing succeeded.
          _log?.warn(
            'Memory correction discarded: the memory being corrected no '
            'longer exists.',
            source: 'Memory',
          );
          return;
        }
        final confident = confidence >= MemoryService.confidentCorrectionThreshold;
        _log?.info(
          confident
              ? 'Memory updated (superseded prior entry, confidence '
                  '${confidence.toStringAsFixed(2)}): "$noteText" ($category/$valence)'
              : 'Memory conflict recorded (low confidence '
                  '${confidence.toStringAsFixed(2)}, both kept active): '
                  '"$noteText" ($category/$valence)',
          source: 'Memory',
        );
        _toast(confident
            ? 'Memory updated: ${_truncateForToast(noteText)}'
            : 'Possible conflict noted: ${_truncateForToast(noteText)}');
        aiMsg.telemetry?.setExtractedValence(category, valence);
      } else {
        final result = await _memory.addIfNotDuplicate(
          noteText,
          vector,
          sourceChatId: chatId,
          category: category,
          valence: valence,
          tags: tags,
          confidence: confidence,
          // Captured automatically, not something a human deliberately
          // chose to save — starts on probation (see
          // MemoryService.runWorkingMemoryMaintenance) and earns permanence
          // by actually getting reinforced, rather than being permanent the
          // instant it's written.
          isWorkingMemory: true,
          entityName: entityName,
          entityType: entityType,
          location: entityLocation,
          participants: participants,
          connection: connection,
        );
        if (result.wasNew) {
          _log?.info('Memory saved: "$noteText" ($category/$valence)', source: 'Memory');
          _toast('Memory saved: ${_truncateForToast(noteText)}');
          aiMsg.telemetry?.setExtractedValence(category, valence);
        } else if (result.wasEnriched) {
          _log?.info('Memory enriched: "$noteText" ($category/$valence)', source: 'Memory');
          _toast('Memory updated: ${_truncateForToast(noteText)}');
          aiMsg.telemetry?.setExtractedValence(category, valence);
        } else {
          // Deliberately no telemetry marker here — this is a pure repeat
          // of something already known, nothing new was captured, and the
          // insight panel's "memory captured" chip would be misleading on
          // a turn where nothing actually changed in memory.
          _log?.info('Memory reinforced (already known): "$noteText"', source: 'Memory');
        }
      }
    } catch (e) {
      // Best-effort background task — never let this affect the chat UI,
      // but still log it so a silent failure here is diagnosable from the
      // in-app log screen instead of just presenting as "memory never
      // saves anything" with no visible cause.
      _log?.error('Memory extraction failed: $e', source: 'Memory');
    }
  }

  static final RegExp _extractedTextFieldPattern =
      RegExp(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"');

  /// Best-effort parse of the extraction model's JSON response. Returns
  /// null (never throws) if nothing usable could be recovered at all, so
  /// the caller can fall back to treating the raw text as a plain memory
  /// note — a small quantized model failing to format valid JSON should
  /// degrade gracefully, not lose the memory entirely.
  ({
    String text,
    String category,
    String valence,
    List<String> tags,
    int? contradictsIndex,
    double confidence,
    String entityName,
    String entityType,
    String location,
    List<String> participants,
    String connection,
  })?
      _parseExtractionJson(String raw) {
    const validCategories = {'fact', 'preference', 'event', 'instruction', 'general'};
    const validValences = {'positive', 'negative', 'neutral'};
    const validEntityTypes = {
      'person', 'place', 'project', 'organization', 'event', 'idea', 'none',
    };

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
          final entityName = (decoded['entity_name'] as Object?)?.toString().trim() ?? '';
          final entityTypeRaw =
              (decoded['entity_type'] as Object?)?.toString().toLowerCase();
          final entityLocation = (decoded['location'] as Object?)?.toString().trim() ?? '';
          final participantsRaw = decoded['participants'];
          final participants = participantsRaw is List
              ? participantsRaw
                  .map((e) => e.toString().trim())
                  .where((e) => e.isNotEmpty)
                  .take(10)
                  .toList()
              : <String>[];
          final connection = (decoded['connection'] as Object?)?.toString().trim() ?? '';
          // Accept a real JSON number or a stringified one ("2") — a small
          // quantized model quoting a number is a common enough formatting
          // slip. null/"null"/missing/anything else means "no contradiction".
          final contradictsRaw = decoded['contradicts_index'];
          final contradictsIndex = contradictsRaw is num
              ? contradictsRaw.toInt()
              : int.tryParse(contradictsRaw?.toString() ?? '');
          // Same leniency as above — a stringified "0.8" is as acceptable
          // as a real number. Missing/unparseable defaults to 0.5, not
          // 1.0: this is the same small-model class that produces
          // malformed JSON often enough to need the retry loop above, so a
          // dropped field reads as another symptom of imperfect output,
          // not a real signal of certainty. Defaulting high would route a
          // dropped-field contradiction straight into resolveContradiction's
          // confident-supersede branch — permanently overwriting a
          // correct memory on what might just be a formatting slip, which
          // is exactly the case the confidence field exists to catch.
          // Defaulting to the same 0.5 the truncated-fragment path below
          // uses keeps both facts active and flagged instead — the safer
          // failure mode when genuinely unsure which way it should go.
          final confidenceRaw = decoded['confidence'];
          final confidence = (confidenceRaw is num
                  ? confidenceRaw.toDouble()
                  : double.tryParse(confidenceRaw?.toString() ?? ''))
              ?.clamp(0.0, 1.0) ??
              0.5;

          return (
            text: text,
            category: validCategories.contains(categoryRaw) ? categoryRaw! : 'general',
            valence: validValences.contains(valenceRaw) ? valenceRaw! : 'neutral',
            tags: tags,
            contradictsIndex: contradictsIndex,
            confidence: confidence,
            entityName: entityName,
            entityType: validEntityTypes.contains(entityTypeRaw) ? entityTypeRaw! : 'none',
            location: entityLocation,
            participants: participants,
            connection: connection,
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
      // Recovered from a truncated fragment, not a clean parse — treat as
      // lower-confidence than a fully-formed response, since there's no
      // way to know if the model would have hedged in the part that got
      // cut off.
      confidence: 0.5,
      // Everything past "text" was cut off before it could be recovered —
      // no entity fields to salvage from a fragment this short.
      entityName: '',
      entityType: 'none',
      location: '',
      participants: const [],
      connection: '',
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
    _memorySweepTimer?.cancel();
    super.onClose();
  }
}
