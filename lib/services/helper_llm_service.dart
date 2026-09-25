import 'dart:io';
import 'package:get/get.dart';
import 'package:llamadart/llamadart.dart';
import 'package:path/path.dart' as p;

import 'log_service.dart';
import 'crash_log_service.dart';

/// A small, fast chat model loaded in its own engine instance, entirely
/// separate from the main chat model in [LlmService] — dedicated to short
/// background tasks (currently: distilling a memorable exchange into a
/// persistent-memory note) that shouldn't have to compete with, wait on, or
/// slow down the main conversation model.
///
/// This matters specifically because the main model can be large and slow
/// on-device (a multi-billion-parameter uncensored fine-tune easily runs at
/// well under 1 token/sec on a phone), while the background extraction task
/// only needs a short, cheap judgment call. Running that on the main model
/// meant every "memorable" turn could tie up the only generation slot for as
/// long as the main model takes to produce ~80 tokens — with a small helper
/// model (tens of millions of parameters) running on its own engine, that
/// same job finishes almost instantly and never blocks the main model, the
/// Local API Server, or the next chat message.
class HelperLlmService extends GetxService {
  LlamaEngine? _engine;
  LlamaBackend? _backend;

  final isLoaded = false.obs;
  final isLoading = false.obs;
  final loadedModelPath = ''.obs;
  bool _isGenerating = false;

  String get loadedModelFilename =>
      loadedModelPath.value.isEmpty ? '' : p.basename(loadedModelPath.value);

  /// Whether a [complete] call is already in flight — this engine has one
  /// generation slot, so a second caller (e.g. the periodic memory-
  /// consolidation sweep landing mid-turn-extraction) should check this and
  /// defer rather than call [complete] anyway and get back a silent null.
  bool get isBusy => _isGenerating;

  Future<HelperLlmService> init() async => this;

  /// Loads [path] as the active helper model. Safe to call again with the
  /// same path (no-ops if already loaded).
  Future<void> loadModel(String path) async {
    if (loadedModelPath.value == path && isLoaded.value) return;

    LogService? log;
    try { log = Get.find<LogService>(); } catch (_) {}

    if (!await File(path).exists()) {
      throw Exception('Helper model file not found: $path');
    }

    isLoading.value = true;
    try {
      await _teardown();
      _backend = LlamaBackend();
      _engine = LlamaEngine(_backend!);
      // CPU-only and a modest context: this model only ever sees a short
      // extraction prompt (a capped exchange plus brief instructions), never
      // a full conversation history, and staying off the GPU means it never
      // contends with the main model for the same resource.
      await _engine!.loadModel(
        path,
        modelParams: const ModelParams(
          contextSize: 2048,
          gpuLayers: 0,
          preferredBackend: GpuBackend.cpu,
        ),
      );
      isLoaded.value = true;
      loadedModelPath.value = path;
      log?.info('Helper model loaded: ${p.basename(path)}', source: 'Helper');
    } catch (e, stack) {
      isLoaded.value = false;
      loadedModelPath.value = '';
      log?.error('Helper model load failed: $e', source: 'Helper');
      CrashLogService.instance.record(
        'model_load',
        'Helper model load failed: $e',
        stackTrace: stack.toString(),
        context: 'file=${p.basename(path)}',
      );
      rethrow;
    } finally {
      isLoading.value = false;
    }
  }

  Future<void> unloadModel() async {
    await _teardown();
  }

  /// Cancels an in-progress [complete] call — must be called if a caller
  /// gives up waiting (e.g. via its own `.timeout(...)`), or the native
  /// generation keeps running in the background with [isLoaded]'s
  /// generation slot held for however long it actually takes, silently
  /// blocking every other call to [complete] until it finishes on its own.
  void stopGeneration() {
    _engine?.cancelGeneration();
  }

  /// Runs a short, non-streamed chat completion and returns the accumulated
  /// answer text (thinking/reasoning content, if any, is discarded — this is
  /// for background extraction, not a user-facing reply). Returns null if no
  /// helper model is loaded, another call is already running on this engine
  /// (a single llama.cpp context can't run two generations at once — this
  /// guards it the same way LlmService.generateChatCompletion does), or
  /// generation fails for any reason.
  Future<String?> complete(
    List<LlamaChatMessage> messages, {
    required int maxTokens,
    double temp = 0.2,
  }) async {
    if (_engine == null || !isLoaded.value || _isGenerating) return null;
    _isGenerating = true;
    try {
      final buffer = StringBuffer();
      await for (final chunk in _engine!.create(
        messages,
        params: GenerationParams(temp: temp, maxTokens: maxTokens),
        toolChoice: ToolChoice.none,
        enableThinking: false,
      )) {
        final content = chunk.choices.isNotEmpty ? chunk.choices.first.delta.content : null;
        if (content != null) buffer.write(content);
      }
      return buffer.toString();
    } catch (_) {
      return null;
    } finally {
      _isGenerating = false;
    }
  }

  Future<void> _teardown() async {
    if (_engine != null) {
      try {
        await _engine!.dispose();
      } catch (_) {}
      _engine = null;
    }
    _backend = null;
    isLoaded.value = false;
    loadedModelPath.value = '';
    _isGenerating = false;
  }

  @override
  void onClose() {
    unloadModel();
    super.onClose();
  }
}
