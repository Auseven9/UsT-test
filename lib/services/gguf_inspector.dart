import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

/// What role a `.gguf` file actually plays. Not every GGUF file is a
/// loadable chat model — some are companion artifacts that only make sense
/// paired with one, or serve an entirely different purpose (embeddings).
enum ModelKind {
  /// A normal causal-LM chat/instruct model — the only kind that should be
  /// offered through the regular "Load Model" chat flow.
  chat,

  /// A CLIP vision projector (`mmproj-*.gguf`) — pairs with a chat model to
  /// add image understanding. Not a standalone model.
  visionProjector,

  /// An embedding/retrieval model (BERT-family architectures with a
  /// `pooling_type`, e.g. bge, gte, e5, nomic-embed). Produces vector
  /// embeddings, not chat completions.
  embedding,

  /// A LoRA adapter — modifies a base model's weights rather than being a
  /// standalone model.
  loraAdapter,

  /// The file's kind couldn't be determined (unreadable, truncated, or not
  /// actually a GGUF file).
  unknown,
}

/// Lightweight metadata read from the front of a `.gguf` file, without
/// loading the whole model.
class GgufMetadata {
  final ModelKind kind;
  final String? architecture;
  final String? name;

  const GgufMetadata({required this.kind, this.architecture, this.name});

  static const unknown = GgufMetadata(kind: ModelKind.unknown);
}

/// Parses just the GGUF header + metadata key/value section (never the bulk
/// tensor data) to classify what a `.gguf` file actually is.
///
/// GGUF layout: `"GGUF"` magic, uint32 version, uint64 tensor_count,
/// uint64 metadata_kv_count, then that many (string key, typed value) pairs.
/// See https://github.com/ggml-org/ggml/blob/master/docs/gguf.md.
class GgufInspector {
  GgufInspector._();

  /// How many bytes of the file to read at most. GGUF metadata for every
  /// model family observed in practice (chat models, mmproj/clip files,
  /// BERT-family embedding models) lives well within the first couple of
  /// megabytes; tensor data — the actual weights — starts after it.
  static const int _maxReadBytes = 4 * 1024 * 1024;

  /// A hard cap on how many metadata entries to walk, so a corrupted or
  /// hostile file with a bogus huge `kv_count` can't spin forever.
  static const int _maxKeys = 2000;

  static const int _typeString = 8;
  static const int _typeArray = 9;
  static const Map<int, int> _fixedSizes = {
    0: 1, 1: 1, // uint8/int8
    2: 2, 3: 2, // uint16/int16
    4: 4, 5: 4, 6: 4, // uint32/int32/float32
    7: 1, // bool
    10: 8, 11: 8, 12: 8, // uint64/int64/float64
  };

  /// Inspects [path] and classifies it. Never throws — any parse failure
  /// (not a GGUF file, truncated header, unexpected layout) resolves to
  /// [GgufMetadata.unknown] rather than propagating an exception, since this
  /// runs opportunistically over arbitrary files a user dropped into the
  /// models folder.
  static Future<GgufMetadata> inspect(String path) async {
    RandomAccessFile? raf;
    try {
      final file = File(path);
      final length = await file.length();
      final readLen = length < _maxReadBytes ? length : _maxReadBytes;
      raf = await file.open();
      final bytes = await raf.read(readLen);
      return _parse(bytes);
    } catch (_) {
      return GgufMetadata.unknown;
    } finally {
      try {
        await raf?.close();
      } catch (_) {}
    }
  }

  static GgufMetadata _parse(Uint8List bytes) {
    final data = ByteData.sublistView(bytes);
    var offset = 0;

    int need(int n) {
      if (offset + n > data.lengthInBytes) {
        throw const FormatException('truncated GGUF header');
      }
      final at = offset;
      offset += n;
      return at;
    }

    // Magic: "GGUF"
    final magicAt = need(4);
    final magic = String.fromCharCodes(bytes, magicAt, magicAt + 4);
    if (magic != 'GGUF') return GgufMetadata.unknown;

    need(4); // version — not needed for classification
    need(8); // tensor_count — not needed for classification
    final kvCountAt = need(8);
    final kvCount = data.getUint64(kvCountAt, Endian.little);

    String readString() {
      final lenAt = need(8);
      final len = data.getUint64(lenAt, Endian.little);
      final strAt = need(len);
      return utf8.decode(
        bytes.sublist(strAt, strAt + len),
        allowMalformed: true,
      );
    }

    dynamic readValue(int type) {
      if (type == _typeString) return readString();
      if (type == _typeArray) {
        final elemTypeAt = need(4);
        final elemType = data.getUint32(elemTypeAt, Endian.little);
        final countAt = need(8);
        final count = data.getUint64(countAt, Endian.little);
        // We never need array contents for classification — skip them, but
        // still validate the file doesn't run past our buffer.
        for (var i = 0; i < count; i++) {
          readValue(elemType);
        }
        return null;
      }
      final size = _fixedSizes[type];
      if (size == null) {
        throw FormatException('unknown GGUF value type $type');
      }
      final at = need(size);
      if (type == 7) return bytes[at] != 0; // bool
      if (type == 4) return data.getUint32(at, Endian.little);
      if (type == 5) return data.getInt32(at, Endian.little);
      return null; // numeric types we don't care about for classification
    }

    String? architecture;
    String? generalType;
    String? name;
    var hasPoolingType = false;

    final keysToRead = kvCount < _maxKeys ? kvCount : _maxKeys;
    for (var i = 0; i < keysToRead; i++) {
      final String key;
      final int valueType;
      try {
        key = readString();
        final typeAt = need(4);
        valueType = data.getUint32(typeAt, Endian.little);
      } catch (_) {
        // Ran out of buffer before finishing metadata — work with whatever
        // we've already found rather than failing the whole classification.
        break;
      }

      try {
        if (key == 'general.architecture') {
          architecture = readValue(valueType) as String?;
        } else if (key == 'general.type') {
          generalType = readValue(valueType) as String?;
        } else if (key == 'general.name') {
          name = readValue(valueType) as String?;
        } else if (key.endsWith('.pooling_type')) {
          hasPoolingType = true;
          readValue(valueType);
        } else {
          readValue(valueType);
        }
      } catch (_) {
        break;
      }
    }

    final kind = _classify(
      architecture: architecture,
      generalType: generalType,
      hasPoolingType: hasPoolingType,
    );
    return GgufMetadata(kind: kind, architecture: architecture, name: name);
  }

  static ModelKind _classify({
    required String? architecture,
    required String? generalType,
    required bool hasPoolingType,
  }) {
    if (architecture == 'clip' || generalType == 'clip-vision') {
      return ModelKind.visionProjector;
    }
    // llama.cpp's convert_lora_to_gguf.py writes general.type = "adapter".
    if (generalType == 'adapter') {
      return ModelKind.loraAdapter;
    }
    // BERT-family embedding models (bge, gte, e5, nomic-embed, jina-bert, …)
    // declare a pooling strategy; causal chat models never do.
    if (hasPoolingType) {
      return ModelKind.embedding;
    }
    return ModelKind.chat;
  }
}
