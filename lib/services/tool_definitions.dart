import 'dart:math' as math;

import 'package:llamadart/llamadart.dart';

import 'memory_service.dart';

/// The full set of tools the model is given access to. Deliberately small
/// and entirely offline/local — no network calls, matching the app's
/// offline-privacy positioning (a web-search tool would need its own
/// explicit product decision, not something to add silently here). None of
/// these can delete or modify a stored memory — `search_memory` is
/// read-only, matching the standing constraint that the model must never be
/// able to remove its own memories.
///
/// [includeMemorySearch] should mirror the user's Persistent Memory setting
/// — a model given `search_memory` while the user has that setting off
/// would still be able to read memories collected before it was turned off,
/// which defeats the point of the toggle.
List<ToolDefinition> buildToolDefinitions({
  required MemoryService memory,
  bool includeMemorySearch = true,
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
    if (includeMemorySearch)
      ToolDefinition(
        name: 'search_memory',
        description:
            'Searches this app\'s persistent cross-conversation memory for '
            'notes matching a query — things remembered from past '
            'conversations (the user\'s name, stated preferences, ongoing '
            'projects, etc). Read-only: this cannot add, change, or delete '
            'memories. Use it when the user references something from an '
            'earlier conversation that isn\'t in the current context.',
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
          }).take(5).map((e) => e.text).toList();

          return {'results': matches};
        },
      ),
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
