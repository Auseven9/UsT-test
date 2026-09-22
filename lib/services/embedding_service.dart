import 'dart:io';
import 'dart:math' as math;

import 'package:get/get.dart';
import 'package:llamadart/llamadart.dart';
import 'package:path/path.dart' as p;

import 'log_service.dart';

/// Wraps a second, independent llamadart engine dedicated to embeddings.
///
/// This is deliberately separate from [LlmService]'s chat engine: an
/// embedding model is architecturally different from a chat model (usually
/// much smaller, e.g. ~80-150MB) and needs to stay loaded and queryable
/// on its own schedule, independent of chat generation. Running it CPU-only
/// keeps it from contending with the chat model for GPU layers on
/// constrained hardware, since the two can be resident at the same time.
class EmbeddingService extends GetxService {
  LlamaEngine? _engine;
  LlamaBackend? _backend;

  final isLoaded = false.obs;
  final isLoading = false.obs;
  final loadedModelPath = ''.obs;

  LogService? get _log {
    try {
      return Get.find<LogService>();
    } catch (_) {
      return null;
    }
  }

  Future<EmbeddingService> init() async {
    return this;
  }

  /// Load a GGUF embedding model from [path]. Any previously loaded
  /// embedding model is fully torn down first.
  Future<void> load(String path) async {
    final file = File(path);
    if (!await file.exists()) {
      throw Exception('Embedding model file not found: $path');
    }

    if (_engine != null || isLoaded.value) {
      await unload();
    }

    isLoading.value = true;
    try {
      _backend = LlamaBackend();
      _engine = LlamaEngine(_backend!);

      // Small, CPU-only, short context — embedding queries here are single
      // chat messages, not documents.
      final contextTokens = Platform.isAndroid ? 512 : 1024;
      final params = ModelParams(
        contextSize: contextTokens,
        gpuLayers: 0,
        preferredBackend: GpuBackend.cpu,
        numberOfThreads: Platform.numberOfProcessors > 4 ? 4 : 0,
        numberOfThreadsBatch: Platform.numberOfProcessors > 4 ? 4 : 0,
      );

      await _engine!.loadModel(path, modelParams: params);

      isLoaded.value = true;
      loadedModelPath.value = path;
      _log?.info(
        'Embedding model loaded: ${p.basename(path)}',
        source: 'Embedding',
      );
    } catch (e) {
      _log?.error('Embedding model load failed: $e', source: 'Embedding');
      await _teardown();
      rethrow;
    } finally {
      isLoading.value = false;
    }
  }

  Future<void> unload() async {
    await _teardown();
  }

  Future<void> _teardown() async {
    if (_engine != null) {
      try {
        await _engine!.dispose();
      } catch (_) {
        // Engine may already be in a broken state — ignore.
      }
      _engine = null;
    }
    _backend = null;
    isLoaded.value = false;
    loadedModelPath.value = '';
  }

  /// Embed [text] into a normalized vector, or null if no embedding model
  /// is loaded or the native call fails. Never throws.
  Future<List<double>?> embed(String text) async {
    if (_engine == null || !isLoaded.value || text.isEmpty) return null;
    try {
      return await _engine!.embed(text, normalize: true);
    } catch (e) {
      _log?.warn('Embedding failed: $e', source: 'Embedding');
      return null;
    }
  }

  /// Cosine similarity between two equal-length vectors, in [-1, 1].
  /// Returns 0 for mismatched/empty vectors rather than throwing.
  static double cosineSimilarity(List<double> a, List<double> b) {
    if (a.isEmpty || b.isEmpty || a.length != b.length) return 0.0;
    var dot = 0.0, normA = 0.0, normB = 0.0;
    for (var i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    if (normA == 0.0 || normB == 0.0) return 0.0;
    return dot / (math.sqrt(normA) * math.sqrt(normB));
  }

  @override
  void onClose() {
    unload();
    super.onClose();
  }
}
