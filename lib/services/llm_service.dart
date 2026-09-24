import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:get/get.dart';
import 'package:llamadart/llamadart.dart';
import 'package:path/path.dart' as p;

import 'wakelock_service.dart';
import 'chat_storage_service.dart';
import 'log_service.dart';
import 'crash_log_service.dart';
import 'gguf_inspector.dart';

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

  // ── Loading progress tracking ──────────────────────────────
  final isLoadingModel = false.obs;
  final loadingProgress = 0.0.obs; // 0.0 to 1.0
  final loadingStatusMsg = ''.obs;
  bool _loadingCancelled = false;

  /// Whether a vision projector (mmproj) is currently paired with the loaded
  /// chat model, enabling image input.
  final hasVisionProjector = false.obs;

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

      // Use smaller context on Android to prevent OOM kills.
      // Desktop can handle 2048, but Android devices with limited RAM
      // need 1024 to avoid the Low Memory Killer (LMK).
      final contextSize = Platform.isAndroid ? 1024 : 2048;

      // Map the string backend to GpuBackend enum
      final storage = Get.find<ChatStorageService>();
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

      final params = ModelParams(
        contextSize: contextSize,
        gpuLayers: userGpuLayers,
        preferredBackend: parsedBackend,
        numberOfThreads: threads,
        numberOfThreadsBatch: threads,
      );

      log?.info('Backend=$parsedBackend, GPU layers=$userGpuLayers, ctx=$contextSize, threads=$threads (cores=$cores)', source: 'LLM');

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
  Future<List<LlamaChatMessage>> fitToContext(
    List<LlamaChatMessage> messages, {
    int reserveForResponse = 512,
  }) async {
    if (messages.isEmpty) return messages;

    final ctx = await getContextSize();
    final budget = ctx - reserveForResponse;
    if (budget <= 0) return messages;

    final hasSystem = messages.first.role == LlamaChatRole.system;
    final systemMsg = hasSystem ? messages.first : null;
    final turns = hasSystem ? messages.sublist(1) : messages;

    var total = 0;
    if (systemMsg != null) total += await countTokens(systemMsg.content);

    // Walk newest-first, keeping whatever still fits the budget.
    final keptReversed = <LlamaChatMessage>[];
    for (final msg in turns.reversed) {
      final t = await countTokens(msg.content);
      if (total + t > budget && keptReversed.isNotEmpty) break;
      total += t;
      keptReversed.add(msg);
    }

    final kept = keptReversed.reversed.toList();
    return systemMsg != null ? [systemMsg, ...kept] : kept;
  }

  /// Generate a chat completion using llamadart's chat-template API.
  Stream<String> generateChatCompletion({
    required List<LlamaChatMessage> messages,
    GenerationParams params = const GenerationParams(),
  }) async* {
    if (_engine == null || !isLoaded.value) {
      throw StateError('No model loaded. Call loadModel() first.');
    }
    if (isGenerating.value) {
      throw StateError('Another generation is already in progress.');
    }

    isGenerating.value = true;
    tokensPerSecond.value = 0.0;
    final stopwatch = Stopwatch()..start();
    int tokenCount = 0;

    try {
      await for (final chunk in _engine!.create(
        messages,
        params: params,
        toolChoice: ToolChoice.none,
      )) {
        final choice = chunk.choices.isNotEmpty ? chunk.choices.first : null;
        final content = choice?.delta.content;
        if (content == null || content.isEmpty) continue;

        tokenCount++;
        if (stopwatch.elapsedMilliseconds > 0) {
          tokensPerSecond.value =
              tokenCount / (stopwatch.elapsedMilliseconds / 1000);
        }
        yield content;
      }
    } finally {
      stopwatch.stop();
      lastGenerationTokens.value = tokenCount;
      lastGenerationSpeed.value = tokensPerSecond.value;
      isGenerating.value = false;
    }
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
