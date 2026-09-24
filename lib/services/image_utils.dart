import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' as ui;

/// Downscales image [bytes] so its longest side is at most [maxDim] pixels,
/// always re-encoding the result as PNG — including when no resize was
/// needed — so a caller can rely on the output always being PNG regardless
/// of the source format (JPEG, WEBP, etc). Uses `dart:ui` directly (no
/// image-processing package dependency): vision models feed images through
/// a fixed-size CLIP-style encoder anyway, so sending anything larger than a
/// modest resolution just burns RAM, CPU, and Hive storage for no quality
/// benefit, which matters on a phone.
///
/// Also corrects EXIF orientation first (see [_correctExifOrientation]) —
/// neither `dart:ui`'s JPEG decoder nor llama.cpp's native image loader
/// (stb_image) applies a photo's EXIF orientation tag automatically, so a
/// portrait phone photo decodes sideways/upside-down and gets fed to the
/// vision model rotated, which is exactly the kind of input that makes a
/// human photo look like nonsense to it.
Future<Uint8List> downscaleImageBytes(Uint8List bytes, {int maxDim = 1024}) async {
  final codec = await ui.instantiateImageCodec(bytes);
  final frame = await codec.getNextFrame();
  var image = frame.image;

  final orientation = _readJpegExifOrientation(bytes);
  if (orientation != null && orientation != 1) {
    final corrected = await _applyExifOrientation(image, orientation);
    image.dispose();
    image = corrected;
  }

  final needsResize = image.width > maxDim || image.height > maxDim;
  if (!needsResize) {
    final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    image.dispose();
    return byteData?.buffer.asUint8List() ?? bytes;
  }

  final scale = maxDim / (image.width > image.height ? image.width : image.height);
  final targetWidth = (image.width * scale).round().clamp(1, maxDim);
  final targetHeight = (image.height * scale).round().clamp(1, maxDim);

  final recorder = ui.PictureRecorder();
  final canvas = ui.Canvas(recorder);
  final src = ui.Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble());
  final dst = ui.Rect.fromLTWH(0, 0, targetWidth.toDouble(), targetHeight.toDouble());
  canvas.drawImageRect(image, src, dst, ui.Paint());
  image.dispose();
  final picture = recorder.endRecording();
  final resized = await picture.toImage(targetWidth, targetHeight);

  final byteData = await resized.toByteData(format: ui.ImageByteFormat.png);
  resized.dispose();
  return byteData?.buffer.asUint8List() ?? bytes;
}

/// Rotates/flips [image] per EXIF orientation value [orientation] (2-8; 1 is
/// already-correct and handled by the caller before this is called). Values
/// and the transform each requires per the EXIF spec:
/// 2=mirror-h, 3=rotate180, 4=mirror-v, 5=transpose, 6=rotate90cw,
/// 7=transverse, 8=rotate270cw (=90ccw).
Future<ui.Image> _applyExifOrientation(ui.Image image, int orientation) async {
  final w = image.width.toDouble();
  final h = image.height.toDouble();
  final swapDims = orientation >= 5;
  final outW = (swapDims ? image.height : image.width);
  final outH = (swapDims ? image.width : image.height);

  final recorder = ui.PictureRecorder();
  final canvas = ui.Canvas(recorder);

  switch (orientation) {
    case 2:
      canvas.translate(w, 0);
      canvas.scale(-1, 1);
      break;
    case 3:
      canvas.translate(w, h);
      canvas.rotate(pi);
      break;
    case 4:
      canvas.translate(0, h);
      canvas.scale(1, -1);
      break;
    case 5:
      canvas.rotate(pi / 2);
      canvas.scale(1, -1);
      break;
    case 6:
      canvas.translate(h, 0);
      canvas.rotate(pi / 2);
      break;
    case 7:
      canvas.translate(h, w);
      canvas.rotate(pi / 2);
      canvas.scale(-1, 1);
      break;
    case 8:
      canvas.translate(0, w);
      canvas.rotate(-pi / 2);
      break;
  }

  canvas.drawImage(image, ui.Offset.zero, ui.Paint());
  final picture = recorder.endRecording();
  return picture.toImage(outW, outH);
}

/// Reads the EXIF orientation tag (0x0112) out of a JPEG's APP1/Exif segment
/// by walking its marker segments directly — no package dependency. Returns
/// null for anything that isn't a JPEG, has no EXIF segment, or fails to
/// parse (truncated/malformed data), in which case the caller treats it the
/// same as orientation 1 (no correction applied) rather than guessing.
int? _readJpegExifOrientation(Uint8List bytes) {
  try {
    if (bytes.length < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8) return null;

    var offset = 2;
    while (offset + 4 <= bytes.length) {
      if (bytes[offset] != 0xFF) break;
      final marker = bytes[offset + 1];

      // SOS (start of scan) — actual image data follows, no more markers.
      if (marker == 0xDA || marker == 0xD9) break;
      // Markers with no payload segment.
      if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
        offset += 2;
        continue;
      }

      final segmentLength = (bytes[offset + 2] << 8) | bytes[offset + 3];
      if (marker == 0xE1 && offset + 4 + 6 <= bytes.length) {
        final exifStart = offset + 4;
        final isExif = bytes[exifStart] == 0x45 &&
            bytes[exifStart + 1] == 0x78 &&
            bytes[exifStart + 2] == 0x69 &&
            bytes[exifStart + 3] == 0x66;
        if (isExif) {
          return _parseExifOrientation(bytes, exifStart + 6);
        }
      }

      if (segmentLength < 2) break; // malformed — bail instead of looping
      offset += 2 + segmentLength;
    }
  } catch (_) {
    // Fall through to null — malformed/truncated EXIF is not worth crashing
    // an image attach over.
  }
  return null;
}

int? _parseExifOrientation(Uint8List bytes, int tiffStart) {
  if (tiffStart + 8 > bytes.length) return null;
  final bigEndian = bytes[tiffStart] == 0x4D; // 'MM' vs 'II'

  int u16(int o) => bigEndian
      ? (bytes[o] << 8) | bytes[o + 1]
      : (bytes[o + 1] << 8) | bytes[o];
  int u32(int o) => bigEndian
      ? (bytes[o] << 24) | (bytes[o + 1] << 16) | (bytes[o + 2] << 8) | bytes[o + 3]
      : (bytes[o + 3] << 24) | (bytes[o + 2] << 16) | (bytes[o + 1] << 8) | bytes[o];

  final ifdOffset = u32(tiffStart + 4);
  final ifdStart = tiffStart + ifdOffset;
  if (ifdStart + 2 > bytes.length) return null;

  final numEntries = u16(ifdStart);
  for (var i = 0; i < numEntries; i++) {
    final entryOffset = ifdStart + 2 + i * 12;
    if (entryOffset + 12 > bytes.length) break;
    final tag = u16(entryOffset);
    if (tag == 0x0112) {
      final value = u16(entryOffset + 8);
      if (value >= 1 && value <= 8) return value;
      return null;
    }
  }
  return null;
}
