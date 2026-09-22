// GENERATED CODE - DO NOT MODIFY BY HAND
//
// Hand-written to match build_runner's TypeAdapterGenerator output format
// (no Dart/Flutter toolchain available to run build_runner in this
// environment) — keep in sync with memory_entry.dart if fields change.

part of 'memory_entry.dart';

// **************************************************************************
// TypeAdapterGenerator
// **************************************************************************

class MemoryEntryAdapter extends TypeAdapter<MemoryEntry> {
  @override
  final int typeId = 3;

  @override
  MemoryEntry read(BinaryReader reader) {
    final numOfFields = reader.readByte();
    final fields = <int, dynamic>{
      for (int i = 0; i < numOfFields; i++) reader.readByte(): reader.read(),
    };
    return MemoryEntry(
      id: fields[0] as String,
      text: fields[1] as String,
      createdAt: fields[2] as DateTime?,
      eventDate: fields[3] as DateTime?,
      sourceChatId: fields[4] as String? ?? '',
      embedding: (fields[5] as List?)?.cast<double>(),
    );
  }

  @override
  void write(BinaryWriter writer, MemoryEntry obj) {
    writer
      ..writeByte(6)
      ..writeByte(0)
      ..write(obj.id)
      ..writeByte(1)
      ..write(obj.text)
      ..writeByte(2)
      ..write(obj.createdAt)
      ..writeByte(3)
      ..write(obj.eventDate)
      ..writeByte(4)
      ..write(obj.sourceChatId)
      ..writeByte(5)
      ..write(obj.embedding);
  }

  @override
  int get hashCode => typeId.hashCode;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is MemoryEntryAdapter &&
          runtimeType == other.runtimeType &&
          typeId == other.typeId;
}
