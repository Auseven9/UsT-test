import 'dart:io';
import 'package:get/get.dart';
import 'package:llamadart/llamadart.dart';
import 'package:path/path.dart' as p;

import 'log_service.dart';
import 'crash_log_service.dart';

/// Loads a dedicated embedding model (e.g. bge-small) in its own engine
/// instance, entirely separate from the main chat model in [LlmService].
/// This lets users swap chat models freely without tearing down the memory
/// index's embedding capability, and keeps the two workloads from
/// contending over the same native context.
class EmbeddingService extends GetxService {
  LlamaEngine? _engine;
  LlamaBackend? _backend;

  final isLoaded = false.obs;
  final isLoading = false.obs;
  final loadedModelPath = ''.obs;

  String get loadedModelFilename =>
      loadedModelPath.value.isEmpty ? '' : p.basename(loadedModelPath.value);

  Future<EmbeddingService> init() async => this;

  /// Loads [path] as the active embedding model. Safe to call again with the
  /// same path (no-ops if already loaded).
  Future<void> loadModel(String path) async {
    if (loadedModelPath.value == path && isLoaded.value) return;

    LogService? log;
    try { log = Get.find<LogService>(); } catch (_) {}

    if (!await File(path).exists()) {
      throw Exception('Embedding model file not found: $path');
    }

    isLoading.value = true;
    try {
      await _teardown();
      _backend = LlamaBackend();
      _engine = LlamaEngine(_backend!);
      // Embedding models are small and short-context by design (bge-small's
      // own trained context is 512 tokens) — no GPU offload needed, and
      // keeping this CPU-only means it never competes with the chat model
      // for GPU/VRAM.
      await _engine!.loadModel(
        path,
        modelParams: const ModelParams(
          contextSize: 512,
          gpuLayers: 0,
          preferredBackend: GpuBackend.cpu,
        ),
      );
      isLoaded.value = true;
      loadedModelPath.value = path;
      log?.info('Embedding model loaded: ${p.basename(path)}', source: 'Embed');
    } catch (e, stack) {
      isLoaded.value = false;
      loadedModelPath.value = '';
      log?.error('Embedding model load failed: $e', source: 'Embed');
      CrashLogService.instance.record(
        'model_load',
        'Embedding model load failed: $e',
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

  /// Serializes calls into the native engine — without this, two callers
  /// requesting an embedding at the same moment (chat's own retrieval/
  /// extraction calls racing the periodic memory-sweep probe, for instance)
  /// would hit the same underlying llama.cpp context concurrently, which
  /// risks a native crash or a corrupted vector rather than just one call
  /// waiting briefly for the other.
  Future<void> _embedChain = Future.value();

  /// Embeds [text], or returns null if no embedding model is loaded or the
  /// backend rejects the request (e.g. a non-embedding model was loaded).
  Future<List<double>?> embed(String text) async {
    final result = _embedChain.then((_) async {
      if (_engine == null || !isLoaded.value) return null;
      try {
        // Bounds how long any ONE caller's await can hang on a stuck native
        // call — turns "this Future never resolves" into "this Future
        // resolves to null after 15s". It does NOT actually free up the
        // underlying engine: llamadart runs embedding requests on a single
        // worker isolate that processes them synchronously one at a time,
        // and Dart's Future.timeout() can't cancel in-flight native work —
        // so a genuinely hung native call still blocks that isolate (and
        // therefore every later queued embed(), including chat's own
        // retrieval/extraction calls) for as long as the hang lasts,
        // regardless of this timeout. This is still strictly better than no
        // timeout (no caller is left permanently unresolved), just not a
        // full guarantee against a stuck native call taking the whole
        // embedding pipeline down with it until the engine is reloaded.
        return await _engine!.embed(text).timeout(const Duration(seconds: 15));
      } catch (_) {
        return null;
      }
    });
    // Swallow here so one failed embed doesn't poison the chain for every
    // later caller — each caller still gets its own result/error via the
    // `result` future above.
    _embedChain = result.catchError((_) => null);
    return result;
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
  }

  @override
  void onClose() {
    unloadModel();
    super.onClose();
  }
}
