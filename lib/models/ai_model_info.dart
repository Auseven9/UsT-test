import '../services/gguf_inspector.dart';

/// Represents a downloadable/loadable AI model from the catalog.
class AiModelInfo {
  final String id;
  final String name;
  final String filename;
  final String url;
  final double sizeGb;
  final int minRamGb;
  final String label;        // UNCENSORED / STANDARD / CUSTOM
  final String badge;        // RECOMMENDED, HERETIC, etc.
  final String systemPrompt;
  final bool recommended;

  /// What this file actually is (chat model, vision projector, embedding
  /// model, LoRA adapter). Catalog entries are always [ModelKind.chat];
  /// locally-discovered files are classified by [GgufInspector] and this
  /// field is filled in by [ModelManager] after that scan.
  final ModelKind kind;

  const AiModelInfo({
    required this.id,
    required this.name,
    required this.filename,
    required this.url,
    required this.sizeGb,
    required this.minRamGb,
    required this.label,
    required this.badge,
    required this.systemPrompt,
    this.recommended = false,
    this.kind = ModelKind.chat,
  });

  factory AiModelInfo.fromJson(Map<String, dynamic> json) {
    return AiModelInfo(
      id: json['id'] as String,
      name: json['name'] as String,
      filename: json['filename'] as String,
      url: json['url'] as String,
      sizeGb: (json['sizeGb'] as num).toDouble(),
      minRamGb: (json['minRamGb'] as num).toInt(),
      label: json['label'] as String? ?? 'STANDARD',
      badge: json['badge'] as String? ?? '',
      systemPrompt: json['systemPrompt'] as String? ?? '',
      recommended: json['recommended'] as bool? ?? false,
      kind: ModelKind.values.firstWhere(
        (k) => k.name == json['kind'],
        orElse: () => ModelKind.chat,
      ),
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'filename': filename,
        'url': url,
        'sizeGb': sizeGb,
        'minRamGb': minRamGb,
        'label': label,
        'badge': badge,
        'systemPrompt': systemPrompt,
        'recommended': recommended,
        'kind': kind.name,
      };

  AiModelInfo copyWith({ModelKind? kind}) => AiModelInfo(
        id: id,
        name: name,
        filename: filename,
        url: url,
        sizeGb: sizeGb,
        minRamGb: minRamGb,
        label: label,
        badge: badge,
        systemPrompt: systemPrompt,
        recommended: recommended,
        kind: kind ?? this.kind,
      );

  bool get isUncensored => label == 'UNCENSORED';
  bool get isStandard => label == 'STANDARD';
  bool get isCustom => label == 'CUSTOM';
  bool get isChatModel => kind == ModelKind.chat;
}
