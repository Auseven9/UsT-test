// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'memory_fact.dart';

// **************************************************************************
// TypeAdapterGenerator
// **************************************************************************

class MemoryFactAdapter extends TypeAdapter<MemoryFact> {
  @override
  final int typeId = 3;

  @override
  MemoryFact read(BinaryReader reader) {
    final numOfFields = reader.readByte();
    final fields = <int, dynamic>{
      for (int i = 0; i < numOfFields; i++) reader.readByte(): reader.read(),
    };
    return MemoryFact(
      id: fields[0] as String,
      text: fields[1] as String,
      createdAt: fields[2] as DateTime?,
      lastRecalledAt: fields[3] as DateTime?,
      recallCount: fields[4] as int? ?? 0,
      sourceChatId: fields[5] as String?,
      embedding: (fields[6] as List?)?.cast<double>(),
    );
  }

  @override
  void write(BinaryWriter writer, MemoryFact obj) {
    writer
      ..writeByte(7)
      ..writeByte(0)
      ..write(obj.id)
      ..writeByte(1)
      ..write(obj.text)
      ..writeByte(2)
      ..write(obj.createdAt)
      ..writeByte(3)
      ..write(obj.lastRecalledAt)
      ..writeByte(4)
      ..write(obj.recallCount)
      ..writeByte(5)
      ..write(obj.sourceChatId)
      ..writeByte(6)
      ..write(obj.embedding);
  }

  @override
  int get hashCode => typeId.hashCode;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is MemoryFactAdapter &&
          runtimeType == other.runtimeType &&
          typeId == other.typeId;
}
