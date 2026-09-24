import 'dart:async';
import 'dart:io';
import 'package:get/get.dart';
import 'package:llamadart/llamadart.dart';
import 'package:path/path.dart' as p;

import 'wakelock_service.dart';
import 'chat_storage_service.dart';
import 'log_service.dart';
import 'crash_log_service.dart';
import 'gguf_inspector.dart';

/// A single streamed piece of a generation — `content` is the visible
/// answer text, `thinking` is reasoning/chain-of-thought llamadart's chat
/// template engine separated out (empty when the model doesn't reason, or
/// when its template wasn't recognized — see
/// [LlmService.splitLeakedControlTokens] for that case). `toolCalls` arrives
/// once, fully formed, in the final chunk of a turn where the model decided
/// to call a tool instead of answering directly — llamadart parses the
/// model's tool-call output completely before emitting it, so there's no
/// incremental-fragment accumulation for callers to do.
class GenerationChunk {
  final String content;
  final String thinking;
  final List<LlamaCompletionChunkToolCall>? toolCalls;
  const GenerationChunk({this.content = '', this.thinking = '', this.toolCalls});
}

/// Wraps llamadart's LlamaEngine for model loading, generation, and lifecycle.
class LlmService extends GetxService {
  LlamaEngine? _engine;
  LlamaBackend? _backend;

  final isLoaded = false.obs;
  final isGenerating = false.obs;
  final loadedModelPath = ''.obs;
  final tokensPerSecond = 0.0.obs;
  final lastGenerationTokens = 0.obs;
  final lastGenerationSpeed = 0.0.obs;

  /// Whether the most recently completed generation produced (approximately)
  /// as many tokens as it was allowed to — i.e. it was very likely cut off
  /// by the maxTokens cap rather than reaching a natural stop. llamadart
  /// only ever reports finishReason as 'stop' or 'tool_calls', never a
  /// distinct "hit the length limit" reason, so this is the only signal
  /// available for that — used by [splitLeakedControlTokens]'s caller to
  /// decide whether a single leaked marker near the start of the text means
  /// "cut off mid-thought" versus "one stray tag before a complete answer".
  final lastGenerationHitTokenCap = false.obs;

  // ── Loading progress tracking ──────────────────────────────
  final isLoadingModel = false.obs;
  final loadingProgress = 0.0.obs; // 0.0 to 1.0
  final loadingStatusMsg = ''.obs;
  bool _loadingCancelled = false;

  /// Whether a vision projector (mmproj) is currently paired with the loaded
  /// chat model, enabling image input.
  final hasVisionProjector = false.obs;

  /// The loaded model's own trained context length, read from its GGUF
  /// metadata (e.g. `qwen2.context_length`) after load — 0 means unknown
  /// (no model loaded, or the key wasn't found). Settings uses this to cap
  /// the Context Size slider at something the model can actually use,
  /// instead of a flat guess.
  final maxTrainedContext = 0.obs;

  StreamSubscription? _generateSub;

  String get loadedModelFilename {
    final path = loadedModelPath.value;
    if (path.isEmpty) return '';
    return p.basename(path);
  }

  String get publicModelId {
    final filename = loadedModelFilename;
    if (filename.isEmpty) return 'local';
    final stem = filename.toLowerCase().endsWith('.gguf')
        ? filename.substring(0, filename.length - 5)
        : p.basenameWithoutExtension(filename);
    return stem
        .replaceAll(RegExp(r'[^A-Za-z0-9._-]+'), '-')
        .replaceAll(RegExp(r'-+'), '-')
        .replaceAll(RegExp(r'^-|-$'), '');
  }

  /// Initialize the service.
  Future<LlmService> init() async {
    // Backend is created fresh per loadModel() call — no init needed here
    return this;
  }

  /// Cancel an in-progress model load.
  void cancelLoading() {
    _loadingCancelled = true;
  }

  /// Load a GGUF model from [path] with progress tracking.
  Future<void> loadModel(String path) async {
    LogService? log;
    try { log = Get.find<LogService>(); } catch (_) {}

    // Verify file exists first
    final file = File(path);
    if (!await file.exists()) {
      log?.error('Model file not found: $path', source: 'LLM');
      throw Exception('Model file not found: $path');
    }

    final filename = p.basename(path);

    // Validate the file BEFORE handing it to the native engine — a
    // corrupted/truncated file or a non-chat GGUF (vision projector,
    // embedding model, LoRA adapter) reaching llama.cpp's loader is exactly
    // the kind of input that produces a hard native crash instead of a
    // catchable Dart exception. Reject it here with a clear message instead.
    final meta = await GgufInspector.inspect(path);
    if (meta.kind == ModelKind.unknown) {
      log?.error('Not a valid GGUF file: $filename', source: 'LLM');
      CrashLogService.instance.record(
        'model_load',
        'Rejected invalid/corrupted GGUF file before native load',
        context: 'file=$filename',
      );
      throw Exception(
        '"$filename" doesn\'t look like a valid GGUF file (its header '
        'couldn\'t be read). It may be corrupted or incomplete.',
      );
    }
    if (meta.kind != ModelKind.chat) {
      log?.error('Refusing to load non-chat GGUF as a model: $filename (${meta.kind})',
          source: 'LLM');
      throw Exception(
        '"$filename" is a ${_kindLabel(meta.kind)}, not a chat model — it '
        'can\'t be loaded this way.',
      );
    }

    log?.info('Loading model: $filename', source: 'LLM');

    _loadingCancelled = false;
    isLoadingModel.value = true;
    loadingProgress.value = 0.0;
    loadingStatusMsg.value = 'Preparing...';

    // Enable wake lock during model loading (heavy memory operation)
    WakelockService? wakelockService;
    try {
      wakelockService = Get.find<WakelockService>();
    } catch (_) {}

    // Unload previous if any — MUST fully tear down engine + backend
    if (_engine != null || isLoaded.value) {
      loadingStatusMsg.value = 'Unloading previous model...';
      loadingProgress.value = 0.05;
      await _fullTeardown();
      // Give native side time to release resources
      await Future.delayed(const Duration(milliseconds: 500));
      if (_loadingCancelled) {
        _resetLoadingState();
        return;
      }
    }

    // Fresh backend + engine for every load — prevents stale native state
    // Wrapped in try-catch to handle SELinux crashes on Android where
    // ggml_backend_load_all() attempts to scan '/' which is denied.
    try {
      _backend = LlamaBackend();
      _engine = LlamaEngine(_backend!);
    } catch (e, stack) {
      _backend = null;
      _engine = null;
      _resetLoadingState();
      log?.error('Engine init failed: $e', source: 'LLM');
      CrashLogService.instance.record(
        'model_load',
        'Native engine initialization failed: $e',
        stackTrace: stack.toString(),
        context: 'file=$filename',
      );
      throw Exception(
        'Failed to initialize AI engine. '
        'This may be a device compatibility issue. '
        'Error: $e',
      );
    }

    try {
      loadingStatusMsg.value = 'Loading into memory...';
      loadingProgress.value = 0.1;

      // Get file size for display
      final fileSize = await file.length();
      final sizeGb = (fileSize / (1024 * 1024 * 1024)).toStringAsFixed(1);
      loadingStatusMsg.value = 'Loading $sizeGb GB into memory...';

      // Start a timer to animate progress while loading
      Timer? progressTimer;
      progressTimer = Timer.periodic(const Duration(milliseconds: 300), (
        timer,
      ) {
        if (_loadingCancelled) {
          timer.cancel();
          return;
        }
        // Gradually increase progress (asymptotic approach to 0.95)
        final current = loadingProgress.value;
        if (current < 0.95) {
          loadingProgress.value = current + (0.95 - current) * 0.04;
        }
      });

      if (_loadingCancelled) {
        progressTimer.cancel();
        await _fullTeardown();
        _resetLoadingState();
        return;
      }

      // User-configurable now (Settings → Context Size) instead of a silent
      // hardcoded platform constant. Still worth knowing: a large context on
      // a RAM-constrained Android device risks the Low Memory Killer — the
      // slider's own UI carries that warning rather than silently capping it
      // here, since not applying the value the user actually chose is its
      // own kind of bug.
      final storage = Get.find<ChatStorageService>();
      final contextSize = storage.contextSize;

      // Map the string backend to GpuBackend enum
      GpuBackend parsedBackend;
      switch (storage.backendType) {
        case 'vulkan':
          parsedBackend = GpuBackend.vulkan;
          break;
        case 'opencl':
          parsedBackend = GpuBackend.opencl;
          break;
        default:
          parsedBackend = GpuBackend.cpu;
      }

      // Read gpu layers
      final userGpuLayers = storage.gpuLayers;

      // Scale worker threads with the device's actual core count instead of
      // flatlining at 4 for anything above it. Devices with 8+ cores were
      // leaving real throughput on the table under the old flat cap. Still
      // capped (at 8) rather than using every core, to leave headroom for
      // the OS and UI thread and avoid a thermal-throttle spiral. 0 below 5
      // cores means "let llama.cpp auto-detect", which is safer than a fixed
      // guess on low-core devices.
      final cores = Platform.numberOfProcessors;
      final threads = cores <= 4 ? 0 : (cores * 3 / 4).round().clamp(4, 8);

      final customTemplate = storage.customChatTemplate.trim();

      final params = ModelParams(
        contextSize: contextSize,
        gpuLayers: userGpuLayers,
        preferredBackend: parsedBackend,
        numberOfThreads: threads,
        numberOfThreadsBatch: threads,
        chatTemplate: customTemplate.isNotEmpty ? customTemplate : null,
      );

      log?.info(
          'Backend=$parsedBackend, GPU layers=$userGpuLayers, ctx=$contextSize, '
          'threads=$threads (cores=$cores)',
          source: 'LLM');

      await _engine!.loadModel(path, modelParams: params);
      progressTimer.cancel();

      if (_loadingCancelled) {
        // User cancelled while loading — full cleanup
        await _fullTeardown();
        _resetLoadingState();
        return;
      }

      loadingProgress.value = 1.0;
      loadingStatusMsg.value = 'Ready!';
      isLoaded.value = true;
      loadedModelPath.value = path;
      log?.info('Model loaded successfully: $filename', source: 'LLM');

      // Best-effort: read the model's own trained context length out of its
      // GGUF metadata so Settings can bound the slider by what the model can
      // actually use, not a flat guess. Fired off in the background rather
      // than awaited here, so a slow metadata read never adds to the time
      // loadModel() itself takes to return.
      maxTrainedContext.value = 0;
      unawaited(_loadMaxTrainedContext());

      // Enable wake lock for inference on mobile (keeps app from being killed)
      final modelName = p.basenameWithoutExtension(path);
      await wakelockService?.enableForInference(modelName: modelName);

      // Brief delay to show 100%
      await Future.delayed(const Duration(milliseconds: 300));
    } catch (e, stack) {
      isLoaded.value = false;
      loadedModelPath.value = '';
      await _fullTeardown();
      log?.error('Model load failed: $e', source: 'LLM');
      CrashLogService.instance.record(
        'model_load',
        'Model load failed: $e',
        stackTrace: stack.toString(),
        context: 'file=$filename',
      );

      // Provide a clearer, more specific error for common failure classes
      // instead of surfacing the raw native/FFI error string to the user.
      final errStr = e.toString().toLowerCase();
      if (Platform.isAndroid &&
          (errStr.contains('memory') || errStr.contains('alloc'))) {
        throw Exception(
          'Not enough RAM to load this model. '
          'Try a smaller model (e.g. Gemma 2 2B at 1.6 GB).',
        );
      }
      if (errStr.contains('no space') || errStr.contains('enospc')) {
        throw Exception(
          'Not enough free storage space to load this model.',
        );
      }
      if (errStr.contains('permission') || errStr.contains('eacces')) {
        throw Exception(
          'Permission denied reading "$filename". Try re-importing the file.',
        );
      }
      rethrow;
    } finally {
      _resetLoadingState();
    }
  }

  /// Reads the loaded model's trained context length from its GGUF metadata
  /// in the background — see [loadModel]'s call site. Silently does nothing
  /// if the engine's been torn down or reloaded by the time it finishes.
  Future<void> _loadMaxTrainedContext() async {
    final requestedFor = loadedModelPath.value;
    try {
      final metadata = await _engine?.getMetadata();
      if (metadata == null || loadedModelPath.value != requestedFor) return;
      for (final entry in metadata.entries) {
        if (entry.key.endsWith('.context_length')) {
          final parsed = int.tryParse(entry.value);
          if (parsed != null && parsed > 0) {
            maxTrainedContext.value = parsed;
          }
          break;
        }
      }
    } catch (_) {
      // Metadata read failing is fine — the slider just falls back to its
      // flat cap.
    }
  }

  void _resetLoadingState() {
    isLoadingModel.value = false;
    loadingProgress.value = 0.0;
    loadingStatusMsg.value = '';
    _loadingCancelled = false;
  }

  /// Real context window size for the loaded model, straight from the
  /// engine. Falls back to a conservative default if unavailable (no model
  /// loaded yet, or the backend can't report it).
  Future<int> getContextSize() async {
    if (_engine == null || !isLoaded.value) return 2048;
    try {
      return await _engine!.getContextSize();
    } catch (_) {
      return 2048;
    }
  }

  /// Sliding window over chat history: keeps a leading system message intact
  /// and drops the oldest turns until the remainder plus [reserveForResponse]
  /// tokens fits inside the model's real context window. Always keeps at
  /// least the most recent turn, even if it alone doesn't fit, so a single
  /// long message is still sent rather than silently dropped.
  ///
  /// Returns both the trimmed messages AND `responseBudget` — the actual
  /// number of response tokens that fit alongside that trimmed history.
  /// Callers MUST use `responseBudget` (clamped to whatever they originally
  /// wanted) as `GenerationParams.maxTokens`, not their raw requested
  /// [reserveForResponse] value: when the request only fit by falling back
  /// to the 25%-of-context floor below, the real remaining budget for a
  /// response is smaller than what was asked for, and generating with the
  /// original (too-large) maxTokens would let history+response overrun the
  /// model's real n_ctx all over again — trimming history alone doesn't
  /// help if the response cap wasn't trimmed to match.
  Future<({List<LlamaChatMessage> messages, int responseBudget})> fitToContext(
    List<LlamaChatMessage> messages, {
    int reserveForResponse = 512,
  }) async {
    if (messages.isEmpty) {
      return (messages: messages, responseBudget: reserveForResponse);
    }

    final ctx = await getContextSize();
    // If the caller's reserve request (response headroom, plus any image
    // token allowance) exceeds the model's actual context window — which
    // happened on every single turn under Android's old 1024-token default
    // context size once tool schemas and image reserves were added — the
    // old code gave up on trimming entirely and returned the FULL untrimmed
    // history. That sent a prompt far bigger than the model's real n_ctx,
    // which the native engine silently cuts off mid-content during
    // ingestion — exactly what produced garbled/chopped-looking replies.
    // Falling back to a guaranteed-positive floor (25% of context) for
    // history instead means history always gets trimmed to something that
    // actually fits, and the response budget shrinks to match (see above)
    // rather than staying at the original, now-too-large request.
    final rawBudget = ctx - reserveForResponse;
    final historyBudget =
        rawBudget > 0 ? rawBudget : (ctx * 0.25).round().clamp(1, ctx);
    final responseBudget =
        rawBudget > 0 ? reserveForResponse : (ctx - historyBudget).clamp(1, ctx);

    final hasSystem = messages.first.role == LlamaChatRole.system;
    final systemMsg = hasSystem ? messages.first : null;
    final turns = hasSystem ? messages.sublist(1) : messages;

    var total = 0;
    if (systemMsg != null) total += await countTokens(systemMsg.content);

    // Walk newest-first, keeping whatever still fits the budget.
    final keptReversed = <LlamaChatMessage>[];
    for (final msg in turns.reversed) {
      final t = await countTokens(msg.content);
      if (total + t > historyBudget && keptReversed.isNotEmpty) break;
      total += t;
      keptReversed.add(msg);
    }

    final kept = keptReversed.reversed.toList();
    final fitted = systemMsg != null ? [systemMsg, ...kept] : kept;
    return (messages: fitted, responseBudget: responseBudget);
  }

  /// Generate a chat completion using llamadart's chat-template API.
  ///
  /// Yields [GenerationChunk]s with `content` and `thinking` kept separate —
  /// llamadart's engine correctly splits these itself for templates it
  /// recognizes (Hermes/Qwen `<think>`, GPT-OSS `<|channel|>`, etc). For a
  /// model whose template *isn't* recognized (custom/community fine-tunes
  /// often ship a hand-modified or outright malformed template), nothing
  /// gets split and raw control-token-like text ends up in `content` — set
  /// [enableThinking] to steer templates that support toggling it, and see
  /// [splitLeakedControlTokens] for the defensive fallback callers should
  /// run over the final accumulated content.
  Stream<GenerationChunk> generateChatCompletion({
    required List<LlamaChatMessage> messages,
    GenerationParams params = const GenerationParams(),
    bool enableThinking = true,
    List<ToolDefinition>? tools,
  }) async* {
    if (_engine == null || !isLoaded.value) {
      throw StateError('No model loaded. Call loadModel() first.');
    }
    if (isGenerating.value) {
      throw StateError('Another generation is already in progress.');
    }

    final hasTools = tools != null && tools.isNotEmpty;

    isGenerating.value = true;
    tokensPerSecond.value = 0.0;
    final stopwatch = Stopwatch()..start();
    int tokenCount = 0;

    try {
      await for (final chunk in _engine!.create(
        messages,
        params: params,
        tools: hasTools ? tools : null,
        toolChoice: hasTools ? ToolChoice.auto : ToolChoice.none,
        enableThinking: enableThinking,
      )) {
        final choice = chunk.choices.isNotEmpty ? chunk.choices.first : null;
        final content = choice?.delta.content ?? '';
        final thinking = choice?.delta.thinking ?? '';
        final toolCalls = choice?.delta.toolCalls;
        if (content.isEmpty && thinking.isEmpty && (toolCalls == null || toolCalls.isEmpty)) {
          continue;
        }

        tokenCount++;
        if (stopwatch.elapsedMilliseconds > 0) {
          tokensPerSecond.value =
              tokenCount / (stopwatch.elapsedMilliseconds / 1000);
        }
        yield GenerationChunk(content: content, thinking: thinking, toolCalls: toolCalls);
      }
    } finally {
      stopwatch.stop();
      lastGenerationTokens.value = tokenCount;
      lastGenerationSpeed.value = tokensPerSecond.value;
      // Approximate: within 2 tokens of the requested cap counts as "hit
      // it" — the exact count can be off by a token or two depending on how
      // thinking/content/tool-call deltas get chunked.
      lastGenerationHitTokenCap.value = tokenCount >= params.maxTokens - 2;
      isGenerating.value = false;
    }
  }

  /// Matches things that look like leaked chat-template control tokens —
  /// `<|channel|>`, `<|start|>`, malformed single-pipe variants like
  /// `<|channel>` / `<channel|>`. Requires a `|` adjacent to a bracket on
  /// at least one side — that's what makes this a control-token pattern and
  /// not just "any bracketed word", which would also match ordinary
  /// placeholders and generic types in real answers like `<YOUR_API_KEY>`
  /// or `List<String>`. Neither of those contains a pipe, so neither matches.
  static final RegExp _leakedControlTokenPattern = RegExp(
    r'<\|[A-Za-z_][A-Za-z0-9_]{0,24}\|?>|<[A-Za-z_][A-Za-z0-9_]{0,24}\|>',
  );

  /// Best-effort safety net for a model whose chat template isn't recognized
  /// by llamadart's format detector (see [generateChatCompletion]'s doc) —
  /// NOT a real parser, just a heuristic: if any control-token-like marker
  /// appears in [text], treat everything up to and including the *last* one
  /// as hidden reasoning and surface only what follows as the real answer.
  /// Returns the original text unchanged (markers stripped) if that heuristic
  /// would discard everything, so a stray tag near the end can never eat a
  /// real answer.
  ///
  /// Special case: a SINGLE marker sitting right at the start of [text],
  /// combined with [wasTruncated] (generation very likely hit its token cap
  /// rather than reaching a natural stop — see [lastGenerationHitTokenCap])
  /// means generation was almost certainly cut off before the model ever
  /// transitioned out of its analysis/thinking channel into a real answer —
  /// there's no answer to extract, only a partial thought. Dumping that raw
  /// analysis narrative into the visible reply as if it were the answer is
  /// worse than being honest that generation didn't finish, so this case
  /// puts everything into `reasoning` and leaves `answer` empty instead.
  /// [wasTruncated] gates this deliberately: without it, the exact same
  /// single-marker-at-the-start shape also matches a model that leaked one
  /// stray reserved-token tag before an otherwise complete, correct short
  /// answer — treating that case as "cut off" would hide a real answer
  /// inside the collapsed Thoughts panel instead of showing it.
  static ({String reasoning, String answer}) splitLeakedControlTokens(
    String text, {
    bool wasTruncated = false,
  }) {
    final matches = _leakedControlTokenPattern.allMatches(text).toList();
    if (matches.isEmpty) return (reasoning: '', answer: text);

    if (wasTruncated && matches.length == 1 && matches.first.start <= 20) {
      final reasoning = text.replaceAll(_leakedControlTokenPattern, '').trim();
      return (reasoning: reasoning, answer: '');
    }

    final last = matches.last;
    final answer = text.substring(last.end).trim();
    if (answer.isEmpty) {
      return (
        reasoning: '',
        answer: text.replaceAll(_leakedControlTokenPattern, '').trim(),
      );
    }
    final reasoning = text
        .substring(0, last.end)
        .replaceAll(_leakedControlTokenPattern, '')
        .trim();
    return (reasoning: reasoning, answer: answer);
  }

  /// Pair a CLIP vision projector (mmproj file) with the currently loaded
  /// chat model, enabling image input for models that support it (e.g.
  /// Gemma-3/4 with its matching `mmproj-*.gguf`). Must be called after
  /// [loadModel] — the projector is loaded into the same engine/context.
  Future<void> loadVisionProjector(String mmprojPath) async {
    if (_engine == null || !isLoaded.value) {
      throw StateError('Load a chat model before pairing a vision projector.');
    }
    LogService? log;
    try { log = Get.find<LogService>(); } catch (_) {}
    try {
      await _engine!.loadMultimodalProjector(mmprojPath);
      hasVisionProjector.value = true;
      log?.info('Vision projector loaded: ${p.basename(mmprojPath)}', source: 'LLM');
    } catch (e, stack) {
      hasVisionProjector.value = false;
      log?.error('Vision projector load failed: $e', source: 'LLM');
      CrashLogService.instance.record(
        'model_load',
        'Vision projector load failed: $e',
        stackTrace: stack.toString(),
        context: 'mmproj=${p.basename(mmprojPath)}',
      );
      rethrow;
    }
  }

  /// Removes the currently paired vision projector, if any.
  Future<void> unloadVisionProjector() async {
    if (_engine == null) return;
    try {
      await _engine!.unloadMultimodalProjector();
    } catch (_) {
      // Nothing to unload, or engine already torn down — either is fine.
    } finally {
      hasVisionProjector.value = false;
    }
  }

  Future<int> countTokens(String text) async {
    if (_engine == null || !isLoaded.value) return 0;
    try {
      return await _engine!.getTokenCount(text);
    } catch (_) {
      return 0;
    }
  }

  /// Stop ongoing generation.
  Future<void> stopGeneration() async {
    _generateSub?.cancel();
    _generateSub = null;
    _engine?.cancelGeneration();
    isGenerating.value = false;
  }

  /// Full native teardown — dispose engine AND backend to prevent stale state.
  Future<void> _fullTeardown() async {
    if (_engine != null) {
      try {
        await _engine!.dispose();
      } catch (_) {
        // Engine may already be in broken state — ignore
      }
      _engine = null;
    }
    // Also destroy the backend — it can't be reused after engine disposal
    _backend = null;
    isLoaded.value = false;
    loadedModelPath.value = '';
    tokensPerSecond.value = 0.0;
    hasVisionProjector.value = false;
    maxTrainedContext.value = 0;
  }

  /// Unload the current model and free memory.
  Future<void> unloadModel() async {
    await _fullTeardown();

    // Disable wake lock when model is unloaded
    try {
      final wakelockService = Get.find<WakelockService>();
      await wakelockService.disable();
    } catch (_) {}
  }

  String _kindLabel(ModelKind kind) {
    switch (kind) {
      case ModelKind.visionProjector:
        return 'vision projector';
      case ModelKind.embedding:
        return 'embedding model';
      case ModelKind.loraAdapter:
        return 'LoRA adapter';
      case ModelKind.chat:
      case ModelKind.unknown:
        return 'file';
    }
  }

  @override
  void onClose() {
    unloadModel();
    super.onClose();
  }
}
