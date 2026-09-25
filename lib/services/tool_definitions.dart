import 'dart:math' as math;

import 'package:flutter/services.dart' show Clipboard, ClipboardData;
import 'package:llamadart/llamadart.dart';

import '../models/memory_entry.dart';
import 'memory_service.dart';
import 'embedding_service.dart';
import 'reasoning_trace_service.dart';
import 'reminder_service.dart';

/// The full set of tools the model is given access to. Deliberately small
/// and entirely offline/local — no network calls, matching the app's
/// offline-privacy positioning (a web-search tool would need its own
/// explicit product decision, not something to add silently here).
///
/// Memory tools used to be read-only by design (`search_memory` only) so
/// the model could never narrate a false "I saved that" action or corrupt
/// its own memory unsupervised. That constraint has been explicitly
/// lifted: the model now has the same write capabilities the background
/// extraction pipeline has — `remember`, `update_memory`,
/// `supersede_memory` — callable directly, mid-conversation. Background
/// extraction still runs independently (see ChatController); the two
/// don't coordinate beyond the ordinary near-duplicate/enrichment logic
/// already in MemoryService, so the same fact stated once can legitimately
/// get written twice from two different paths — that's deduped the same
/// way any other near-duplicate is, not specially handled.
///
/// [includeMemoryTools] should mirror the user's Persistent Memory setting
/// — a model given these tools while the user has that setting off would
/// still be able to read and write memories the toggle was meant to
/// disable, which defeats the point of the toggle.
///
/// [includeAdvancedTools] gates clipboard access, in-app reminders, and
/// reasoning recall — grouped and toggled separately from memory since
/// they're a different kind of capability (device/self access, not fact
/// storage) with their own Settings switch.
List<ToolDefinition> buildToolDefinitions({
  required MemoryService memory,
  required EmbeddingService embedding,
  required ReasoningTraceService reasoningTraces,
  required ReminderService reminders,
  bool includeMemoryTools = true,
  bool includeAdvancedTools = true,
}) {
  return [
    ToolDefinition(
      name: 'get_current_datetime',
      description:
          'Returns the current date and time on the user\'s device. Use this '
          'whenever you need to know today\'s date, the current time, or do '
          'date/time math — you have no other way to know the real current '
          'time.',
      parameters: const [],
      handler: (params) async {
        final now = DateTime.now();
        return {
          'iso8601': now.toIso8601String(),
          'weekday': _weekdayName(now.weekday),
          'date': '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}',
          'time': '${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}',
        };
      },
    ),
    ToolDefinition(
      name: 'calculate',
      description:
          'Evaluates a basic arithmetic expression and returns the numeric '
          'result. Supports +, -, *, /, %, ^ (power), parentheses, and '
          'decimals. Use this for any calculation instead of doing the '
          'arithmetic yourself — it\'s exact, you sometimes aren\'t.',
      parameters: [
        ToolParam.string(
          'expression',
          description: 'The arithmetic expression to evaluate, e.g. "(12 + 8) * 3"',
          required: true,
        ),
      ],
      handler: (params) async {
        final expr = params.getRequiredString('expression');
        try {
          final result = _evaluateArithmetic(expr);
          return {'result': result};
        } catch (e) {
          return {'error': 'Could not evaluate "$expr": $e'};
        }
      },
    ),
    if (includeAdvancedTools) ...[
      ToolDefinition(
        name: 'read_clipboard',
        description:
            'Reads whatever text is currently on the user\'s device '
            'clipboard. Use this when the user refers to something they '
            'just copied ("paste this", "what I copied", "check my '
            'clipboard") instead of asking them to type it out again.',
        parameters: const [],
        handler: (params) async {
          try {
            final data = await Clipboard.getData(Clipboard.kTextPlain);
            final text = data?.text;
            if (text == null || text.isEmpty) {
              return {'text': null, 'error': 'Clipboard is empty.'};
            }
            return {'text': text};
          } catch (e) {
            return {'error': 'Could not read the clipboard: $e'};
          }
        },
      ),
      ToolDefinition(
        name: 'write_clipboard',
        description:
            'Copies text to the user\'s device clipboard, so they can paste '
            'it elsewhere. Use this when the user asks you to copy '
            'something for them.',
        parameters: [
          ToolParam.string(
            'text',
            description: 'The exact text to copy to the clipboard.',
            required: true,
          ),
        ],
        handler: (params) async {
          final text = params.getRequiredString('text');
          try {
            await Clipboard.setData(ClipboardData(text: text));
            return {'copied': true};
          } catch (e) {
            return {'error': 'Could not write to the clipboard: $e'};
          }
        },
      ),
      ToolDefinition(
        name: 'set_reminder',
        description:
            'Saves a reminder that will be surfaced the next time the app '
            'is open after it comes due. This is an in-app reminder, not a '
            'phone notification — it won\'t alert the user while the app '
            'is closed, only when they next open it after the time passes. '
            'Say that plainly if it matters for what they asked.',
        parameters: [
          ToolParam.string(
            'text',
            description: 'What to remind the user about.',
            required: true,
          ),
          ToolParam.integer(
            'minutes_from_now',
            description: 'How many minutes from now the reminder is due.',
            required: true,
          ),
        ],
        handler: (params) async {
          final text = params.getRequiredString('text').trim();
          final minutes = params.getRequiredInt('minutes_from_now');
          if (text.isEmpty) return {'error': 'text was empty — nothing saved.'};
          if (minutes < 0) {
            return {'error': 'minutes_from_now can\'t be negative.'};
          }
          final dueAt = DateTime.now().add(Duration(minutes: minutes));
          final reminder = await reminders.add(text, dueAt);
          return {'saved': true, 'id': reminder.id, 'due_at': dueAt.toIso8601String()};
        },
      ),
      ToolDefinition(
        name: 'recall_reasoning',
        description:
            'Looks back at short gists of your own past reasoning from '
            'earlier turns (not facts about the user — that\'s '
            'search_memory — but how you approached things before). Use '
            'this if it would help to check whether you\'ve reasoned '
            'through something similar recently.',
        parameters: [
          ToolParam.integer(
            'count',
            description: 'How many recent gists to return (default 5, max 15).',
            required: false,
          ),
        ],
        handler: (params) async {
          final count = (params.getInt('count') ?? 5).clamp(1, 15);
          return {'traces': reasoningTraces.recent(count: count)};
        },
      ),
    ],
    if (includeMemoryTools) ...[
      ToolDefinition(
        name: 'search_memory',
        description:
            'Searches this app\'s persistent cross-conversation memory for '
            'notes matching a query — things remembered from past '
            'conversations (the user\'s name, stated preferences, ongoing '
            'projects, etc). Use it when the user references something from '
            'an earlier conversation that isn\'t in the current context.',
        parameters: [
          ToolParam.string(
            'query',
            description: 'Keywords to search stored memories for',
            required: true,
          ),
        ],
        handler: (params) async {
          final query = params.getRequiredString('query').toLowerCase().trim();
          if (query.isEmpty) return {'results': <String>[]};
          final terms = query.split(RegExp(r'\s+')).where((t) => t.length > 2).toList();
          if (terms.isEmpty) return {'results': <String>[]};

          final matches = memory.entries.where((e) {
            if (!e.isActive) return false; // superseded — an outdated fact
            final text = e.text.toLowerCase();
            return terms.any((t) => text.contains(t));
          }).take(5).map((e) => '${e.id}: ${e.text}').toList();

          return {'results': matches};
        },
      ),
      ToolDefinition(
        name: 'remember',
        description:
            'Writes a new durable memory right now, immediately — for '
            'something worth remembering in future, unrelated '
            'conversations (the user\'s name, a stated preference, an '
            'ongoing project, something they explicitly asked to be '
            'remembered). This actually saves it; do not claim you\'ve '
            'remembered something without calling this. If something you '
            'save is a near-duplicate of an existing memory, it '
            'reinforces that memory instead of creating a second copy.',
        parameters: [
          ToolParam.string(
            'text',
            description: 'One short, self-contained sentence stating the fact.',
            required: true,
          ),
          ToolParam.string(
            'category',
            description: 'One of: fact, preference, event, instruction, general.',
            required: false,
          ),
          ToolParam.string(
            'valence',
            description: 'One of: positive, negative, neutral.',
            required: false,
          ),
        ],
        handler: (params) async {
          final text = params.getRequiredString('text').trim();
          if (text.isEmpty) return {'error': 'text was empty — nothing saved.'};
          final vector = await embedding.embed(text);
          if (vector == null) {
            return {'error': 'Embedding model unavailable — nothing saved.'};
          }
          const validCategories = {'fact', 'preference', 'event', 'instruction', 'general'};
          const validValences = {'positive', 'negative', 'neutral'};
          final category = params.getString('category')?.toLowerCase();
          final valence = params.getString('valence')?.toLowerCase();
          final result = await memory.addIfNotDuplicate(
            text,
            vector,
            category: validCategories.contains(category) ? category! : 'general',
            valence: validValences.contains(valence) ? valence! : 'neutral',
            tags: const ['model-written'],
            // Same probation period as automatic background extraction —
            // the model deciding on its own something is worth saving mid-
            // conversation isn't automatically more trustworthy than the
            // background pass making the same call.
            isWorkingMemory: true,
          );
          return {
            'saved': result.id != null,
            'was_new': result.wasNew,
            'was_reinforcement_of_existing': !result.wasNew && !result.wasEnriched,
          };
        },
      ),
      ToolDefinition(
        name: 'update_memory',
        description:
            'Rewrites the text of an existing memory in place — use this '
            'to fix or extend a memory you already have, not to record a '
            'new, unrelated fact (use remember for that). Find the '
            'memory\'s id first with search_memory; its results are '
            'formatted "id: text".',
        parameters: [
          ToolParam.string('id', description: 'The memory\'s id.', required: true),
          ToolParam.string(
            'new_text',
            description: 'The full corrected/extended text.',
            required: true,
          ),
        ],
        handler: (params) async {
          final id = params.getRequiredString('id').trim();
          final newText = params.getRequiredString('new_text').trim();
          if (newText.isEmpty) return {'error': 'new_text was empty — nothing changed.'};
          final vector = await embedding.embed(newText);
          if (vector == null) {
            return {'error': 'Embedding model unavailable — nothing changed.'};
          }
          final ok = await memory.updateEntry(id, text: newText, newEmbedding: vector);
          return {'updated': ok, if (!ok) 'error': 'No memory with that id.'};
        },
      ),
      ToolDefinition(
        name: 'supersede_memory',
        description:
            'Marks an existing memory as replaced by a new, corrected '
            'fact — use this when something you remember turned out to '
            'be wrong or outdated, not for a minor extension (use '
            'update_memory for that). The old memory is kept (never '
            'deleted) but stops being treated as current. Find the old '
            'memory\'s id first with search_memory.',
        parameters: [
          ToolParam.string(
            'old_id',
            description: 'The id of the memory being corrected.',
            required: true,
          ),
          ToolParam.string(
            'new_text',
            description: 'The corrected fact, as a full sentence.',
            required: true,
          ),
        ],
        handler: (params) async {
          final oldId = params.getRequiredString('old_id').trim();
          final newText = params.getRequiredString('new_text').trim();
          if (newText.isEmpty) return {'error': 'new_text was empty — nothing changed.'};
          final vector = await embedding.embed(newText);
          if (vector == null) {
            return {'error': 'Embedding model unavailable — nothing changed.'};
          }
          MemoryEntry? oldEntry;
          for (final e in memory.entries) {
            if (e.id == oldId) {
              oldEntry = e;
              break;
            }
          }
          if (oldEntry == null) {
            return {'error': 'No memory with that id.'};
          }
          // The model calling this tool is, by construction, confident
          // enough to act — it decided this on its own, not as a
          // hedge — so this always supersedes outright rather than
          // going through the low-confidence "keep both, flag conflict"
          // path extraction-time contradictions get. Carries the old
          // entry's category/valence/tags forward — a correction is still
          // the same kind of fact, not a reset to defaults.
          final superseded = await memory.resolveContradiction(
            oldId: oldId,
            newText: newText,
            newEmbedding: vector,
            confidence: 1.0,
            category: oldEntry.category,
            valence: oldEntry.valence,
            tags: oldEntry.tags,
          );
          return {
            'superseded': superseded,
            if (!superseded) 'error': 'That memory no longer exists.',
          };
        },
      ),
    ],
  ];
}

String _weekdayName(int weekday) {
  const names = [
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
  ];
  return names[(weekday - 1).clamp(0, 6)];
}

/// Minimal recursive-descent arithmetic evaluator — deliberately hand-rolled
/// instead of using a generic `eval`/expression package, since the only
/// grammar needed is +, -, *, /, %, ^, parentheses, and unary minus, and a
/// tiny closed grammar is easier to audit for safety than pulling in a
/// general-purpose expression evaluator dependency.
num _evaluateArithmetic(String expr) {
  final parser = _ArithmeticParser(_tokenize(expr));
  final result = parser.parseExpression();
  if (!parser.atEnd) {
    throw FormatException('unexpected trailing token "${parser.current}"');
  }
  return result;
}

/// Recursive-descent parser for +, -, *, /, %, ^, parentheses, unary minus —
/// a plain class rather than nested local functions so the parse methods can
/// freely call each other regardless of declaration order (Dart's local
/// function declarations don't support forward references the way class
/// methods do).
class _ArithmeticParser {
  final List<String> tokens;
  int pos = 0;
  _ArithmeticParser(this.tokens);

  bool get atEnd => pos >= tokens.length;
  String get current => tokens[pos];

  num parseExpression() {
    var value = parseTerm();
    while (!atEnd && (current == '+' || current == '-')) {
      final op = tokens[pos++];
      final rhs = parseTerm();
      value = op == '+' ? value + rhs : value - rhs;
    }
    return value;
  }

  num parseTerm() {
    var value = parseUnary();
    while (!atEnd && (current == '*' || current == '/' || current == '%')) {
      final op = tokens[pos++];
      final rhs = parseUnary();
      if (op == '*') {
        value = value * rhs;
      } else if (op == '/') {
        if (rhs == 0) throw const FormatException('division by zero');
        value = value / rhs;
      } else {
        value = value % rhs;
      }
    }
    return value;
  }

  // Unary minus binds *looser* than '^' (standard math convention: -2^2 ==
  // -(2^2) == -4, not (-2)^2 == 4) but tighter than '*'/'/' — hence parseTerm
  // calling parseUnary, and parseUnary falling through to parsePower rather
  // than the other way around. The exponent itself is parsed via parseUnary
  // so `2^-2` still works, and '^' is right-associative (2^3^2 == 2^(3^2)).
  num parseUnary() {
    if (!atEnd && current == '-') {
      pos++;
      return -parseUnary();
    }
    if (!atEnd && current == '+') {
      pos++;
      return parseUnary();
    }
    return parsePower();
  }

  num parsePower() {
    final base = parseAtom();
    if (!atEnd && current == '^') {
      pos++;
      final exponent = parseUnary();
      return math.pow(base, exponent);
    }
    return base;
  }

  num parseAtom() {
    if (atEnd) throw const FormatException('unexpected end of expression');
    final token = current;
    if (token == '(') {
      pos++;
      final value = parseExpression();
      if (atEnd || current != ')') {
        throw const FormatException('missing closing parenthesis');
      }
      pos++;
      return value;
    }
    final n = num.tryParse(token);
    if (n == null) throw FormatException('unexpected token "$token"');
    pos++;
    return n;
  }
}

List<String> _tokenize(String expr) {
  final tokens = <String>[];
  var i = 0;
  while (i < expr.length) {
    final c = expr[i];
    if (c.trim().isEmpty) {
      i++;
      continue;
    }
    if ('+-*/%^()'.contains(c)) {
      tokens.add(c);
      i++;
      continue;
    }
    if (RegExp(r'[0-9.]').hasMatch(c)) {
      var j = i;
      while (j < expr.length && RegExp(r'[0-9.]').hasMatch(expr[j])) {
        j++;
      }
      tokens.add(expr.substring(i, j));
      i = j;
      continue;
    }
    throw FormatException('unexpected character "$c"');
  }
  return tokens;
}
