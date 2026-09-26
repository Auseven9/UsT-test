import 'dart:io';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../theme/app_colors.dart';
import '../controllers/chat_controller.dart';
import '../controllers/theme_controller.dart';
import '../controllers/model_controller.dart';
import '../services/local_api_server_service.dart';
import '../services/model_manager.dart';
import '../services/gguf_inspector.dart';
import '../services/background_optimizer_service.dart';
import '../services/chat_storage_service.dart';
import '../services/embedding_service.dart';
import '../services/helper_llm_service.dart';
import '../services/memory_service.dart';
import '../services/reminder_service.dart';
import '../services/tutorial_service.dart';
import '../data/tutorial_steps.dart';
import '../widgets/tutorial_overlay.dart';
import '../services/crash_log_service.dart';
import '../services/llm_service.dart';
import '../routes/app_routes.dart';

class SettingsScreen extends StatelessWidget {
  /// When true, no Scaffold — just the body content for embedding in tabs.
  final bool embedded;

  const SettingsScreen({super.key, this.embedded = false});

  @override
  Widget build(BuildContext context) {
    if (embedded) {
      return _SettingsBody(showBackButton: false);
    }
    return Scaffold(
      backgroundColor: context.bg,
      body: _SettingsBody(showBackButton: true),
    );
  }
}

class _SettingsBody extends StatelessWidget {
  final bool showBackButton;

  const _SettingsBody({this.showBackButton = false});

  @override
  Widget build(BuildContext context) {
    final chatCtrl = Get.find<ChatController>();
    final modelManager = Get.find<ModelManager>();
    final themeCtrl = Get.find<ThemeController>();
    final apiServer = Get.find<LocalApiServerService>();
    final storage = Get.find<ChatStorageService>();
    final memory = Get.find<MemoryService>();
    final embedding = Get.find<EmbeddingService>();
    final helper = Get.find<HelperLlmService>();
    final reminders = Get.find<ReminderService>();

    return Column(
      children: [
        // ── Top bar ──────────────────────────────────
        Container(
          padding: EdgeInsets.only(
            top: showBackButton ? MediaQuery.of(context).padding.top : 0,
            left: 4,
            right: 4,
          ),
          decoration: BoxDecoration(
            color: context.bg,
            border: Border(
              bottom: BorderSide(color: context.border, width: 0.5),
            ),
          ),
          child: SizedBox(
            height: 52,
            child: Row(
              children: [
                if (showBackButton)
                  IconButton(
                    icon: Icon(Icons.arrow_back_rounded, color: context.text),
                    onPressed: () => Get.back(),
                  ),
                if (!showBackButton) const SizedBox(width: 16),
                Text(
                  'Settings',
                  style: TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w600,
                    color: context.text,
                  ),
                ),
              ],
            ),
          ),
        ),

        // ── Body ─────────────────────────────────────
        Expanded(
          child: ListView(
            padding: const EdgeInsets.all(20),
            children: [
              // ── Help ─────────────────────────────────────
              _sectionHeader(context, 'Help'),
              const SizedBox(height: 8),
              _card(
                context,
                child: ListTile(
                  leading: Container(
                    width: 36,
                    height: 36,
                    decoration: BoxDecoration(
                      color: AppColors.accent.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Icon(Icons.school_outlined,
                        size: 18, color: AppColors.accent),
                  ),
                  title: Text(
                    'Replay Tutorial',
                    style: TextStyle(color: context.text, fontSize: 14),
                  ),
                  subtitle: Text(
                    'Walks through the message box, model selector, and '
                    'every adjustable setting again, with what each one '
                    'actually does.',
                    style: TextStyle(color: context.textD, fontSize: 12),
                  ),
                  trailing: Icon(Icons.arrow_forward_ios_rounded,
                      size: 14, color: context.textD),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 4,
                  ),
                  onTap: () {
                    // Settings is already tab 2 — jump back to Chat (tab 0)
                    // first so the tour's own first few steps (message
                    // box, model selector) have something to spotlight,
                    // same as a first-run start. Reset to null first: an
                    // Rx setter that assigns the SAME value it already
                    // holds doesn't notify listeners, and requestedTab can
                    // easily already be 0 here (e.g. never touched since
                    // app launch) even though the visible tab is Settings
                    // — going null-then-0 forces the change through
                    // regardless of whatever it was before.
                    final tutorial = Get.find<TutorialService>();
                    tutorial.requestedTab.value = null;
                    tutorial.requestedTab.value = 0;
                    tutorial.replay(buildTutorialSteps());
                  },
                ),
              ),

              const SizedBox(height: 28),

              // ── Appearance ────────────────────────────────
              _sectionHeader(context, 'Appearance'),
              const SizedBox(height: 12),
              _card(
                context,
                child: Obx(
                  () => SwitchListTile(
                    title: Text(
                      'Dark Mode',
                      style: TextStyle(color: context.text, fontSize: 14),
                    ),
                    subtitle: Text(
                      themeCtrl.isDarkMode
                          ? 'Using dark theme'
                          : 'Using light theme',
                      style: TextStyle(color: context.textD, fontSize: 12),
                    ),
                    secondary: Icon(
                      themeCtrl.isDarkMode
                          ? Icons.dark_mode_rounded
                          : Icons.light_mode_rounded,
                      color: context.textM,
                    ),
                    value: themeCtrl.isDarkMode,
                    onChanged: (val) => themeCtrl.toggleTheme(),
                    activeThumbColor: AppColors.accent,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 4,
                    ),
                  ),
                ),
              ),

              if (Platform.isAndroid) ...[
                const SizedBox(height: 12),
                _card(
                  context,
                  child: ListTile(
                    title: Text(
                      'Battery Optimization',
                      style: TextStyle(color: context.text, fontSize: 14),
                    ),
                    subtitle: Text(
                      'Disable to prevent background killing',
                      style: TextStyle(color: context.textD, fontSize: 12),
                    ),
                    trailing: FutureBuilder<bool>(
                      future: BackgroundOptimizerService.isOptimizationDisabled(),
                      builder: (context, snapshot) {
                        final disabled = snapshot.data ?? false;
                        if (disabled) {
                          return const Icon(Icons.check_circle_rounded,
                              color: AppColors.green, size: 20);
                        }
                        return const Icon(Icons.arrow_forward_ios_rounded,
                            size: 14, color: AppColors.orange);
                      },
                    ),
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 4,
                    ),
                    onTap: () async {
                      await BackgroundOptimizerService.openBatterySettings();
                      // Ignore setState, it will rebuild on next visit
                    },
                  ),
                ),
              ],

              const SizedBox(height: 28),

              // ── System Prompt ─────────────────────────────
              Row(
                children: [
                  _sectionHeader(context, 'Global System Prompt'),
                  const SizedBox(width: 8),
                  const ModelInjectionBadge(),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                'Applied to all new chats. Existing chats keep their own prompt.',
                style: TextStyle(fontSize: 12, color: context.textD),
              ),
              const SizedBox(height: 12),
              TutorialTarget(
                id: 'settings.system_prompt',
                child: Obx(
                () => TextField(
                  controller:
                      TextEditingController(text: chatCtrl.systemPrompt.value)
                        ..selection = TextSelection.fromPosition(
                          TextPosition(
                            offset: chatCtrl.systemPrompt.value.length,
                          ),
                        ),
                  maxLines: 4,
                  style: TextStyle(
                    fontSize: 14,
                    color: context.text,
                    height: 1.5,
                  ),
                  decoration: InputDecoration(
                    hintText: 'e.g. You are a helpful assistant...',
                    hintStyle: TextStyle(color: context.textD),
                    filled: true,
                    fillColor: context.bgInput,
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: BorderSide(color: context.border),
                    ),
                    enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: BorderSide(color: context.border),
                    ),
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: const BorderSide(color: AppColors.accent),
                    ),
                  ),
                  onChanged: (v) => chatCtrl.setGlobalSystemPrompt(v),
                ),
              ),
              ),
              const SizedBox(height: 8),
              Align(
                alignment: Alignment.centerLeft,
                child: TextButton.icon(
                  onPressed: () {
                    chatCtrl.clearGlobalSystemPrompt();
                    Get.snackbar(
                      'Cleared',
                      'Global system prompt removed.',
                      snackPosition: SnackPosition.BOTTOM,
                    );
                  },
                  icon: const Icon(Icons.clear_rounded, size: 16),
                  style: TextButton.styleFrom(foregroundColor: AppColors.red),
                  label: const Text(
                    'Clear Prompt',
                    style: TextStyle(fontSize: 13),
                  ),
                ),
              ),

              const SizedBox(height: 28),

              // ── Temperature ───────────────────────────────
              _sectionHeader(context, 'Temperature'),
              const SizedBox(height: 12),
              TutorialTarget(
                id: 'settings.temperature',
                child: _card(
                context,
                child: Obx(
                  () => Padding(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 4,
                    ),
                    child: Row(
                      children: [
                        Icon(
                          Icons.thermostat_rounded,
                          size: 20,
                          color: context.textM,
                        ),
                        Expanded(
                          child: Slider(
                            value: chatCtrl.temperature.value,
                            min: 0.0,
                            max: 2.0,
                            divisions: 20,
                            activeColor: AppColors.accent,
                            inactiveColor: context.border,
                            label: chatCtrl.temperature.value.toStringAsFixed(
                              1,
                            ),
                            onChanged: (v) => chatCtrl.updateTemperature(v),
                          ),
                        ),
                        Container(
                          width: 44,
                          alignment: Alignment.center,
                          child: Text(
                            chatCtrl.temperature.value.toStringAsFixed(1),
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                              color: context.text,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              ),

              const SizedBox(height: 28),

              // ── Context & Sampling ──────────────────────────
              _sectionHeader(context, 'Context & Sampling'),
              const SizedBox(height: 8),
              _GenerationSettingsCard(storage: storage),

              const SizedBox(height: 28),

              // ── Hardware Configuration ──────────────────────────
              _sectionHeader(context, 'Hardware Configuration'),
              const SizedBox(height: 8),
              TutorialTarget(
                id: 'settings.hardware',
                child: _HardwareSettingsCard(storage: storage),
              ),

              const SizedBox(height: 28),

              // ── Persistent Memory ──────────────────────────
              Row(
                children: [
                  _sectionHeader(context, 'Persistent Memory'),
                  const SizedBox(width: 8),
                  const ModelInjectionBadge(),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                'Distills conversations into short notes that can surface again '
                'in any future chat. Needs an embedding model (set one from a '
                'downloaded embedding-kind file in the Models tab).',
                style: TextStyle(fontSize: 12, color: context.textD),
              ),
              const SizedBox(height: 12),
              TutorialTarget(
                id: 'settings.persistent_memory',
                child: _PersistentMemoryCard(
                  storage: storage,
                  memory: memory,
                  embedding: embedding,
                  helper: helper,
                ),
              ),

              const SizedBox(height: 28),

              // ── Reminders ──────────────────────────────────
              _sectionHeader(context, 'Reminders'),
              const SizedBox(height: 8),
              Text(
                'Set via the model\'s set_reminder tool (Advanced Tools). '
                'Surfaced in-app when due — not a phone notification, only '
                'while the app is open.',
                style: TextStyle(fontSize: 12, color: context.textD),
              ),
              const SizedBox(height: 12),
              TutorialTarget(
                id: 'settings.reminders',
                child: _RemindersCard(reminders: reminders),
              ),

              const SizedBox(height: 28),

              // ── Local API Server ──────────────────────────
              _sectionHeader(context, 'Local API Server'),
              const SizedBox(height: 8),
              Text(
                'Expose the loaded model to OpenAI-compatible local clients.',
                style: TextStyle(fontSize: 12, color: context.textD),
              ),
              const SizedBox(height: 12),
              TutorialTarget(
                id: 'settings.local_api_server',
                child: _card(
                context,
                child: Obx(() {
                  final running = apiServer.isRunning.value;
                  final starting = apiServer.isStarting.value;
                  final ready = apiServer.hasLoadedModel;
                  final busy = apiServer.isBusy;
                  final statusText = starting
                      ? 'Starting'
                      : running
                      ? ready
                            ? busy
                                  ? 'Busy'
                                  : 'Running'
                            : 'No model loaded'
                      : 'Stopped';
                  final statusColor = running && ready
                      ? AppColors.green
                      : running
                      ? AppColors.orange
                      : context.textD;

                  return Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        SwitchListTile(
                          title: Text(
                            'Local API server',
                            style: TextStyle(color: context.text, fontSize: 14),
                          ),
                          subtitle: Text(
                            'OpenAI base URL: ${apiServer.baseUrl}',
                            style: TextStyle(
                              color: context.textD,
                              fontSize: 12,
                            ),
                          ),
                          secondary: Icon(
                            Icons.api_rounded,
                            color: running ? AppColors.accent : context.textM,
                          ),
                          value: running,
                          onChanged: starting
                              ? null
                              : (enabled) async {
                                  try {
                                    if (enabled) {
                                      await apiServer.start();
                                    } else {
                                      await apiServer.stop();
                                    }
                                  } catch (e) {
                                    Get.snackbar(
                                      'Local API Error',
                                      e.toString(),
                                      snackPosition: SnackPosition.BOTTOM,
                                    );
                                  }
                                },
                          activeThumbColor: AppColors.accent,
                          contentPadding: EdgeInsets.zero,
                        ),
                        const SizedBox(height: 12),
                        SwitchListTile(
                          title: Text(
                            'Allow External Connections',
                            style: TextStyle(color: context.text, fontSize: 14),
                          ),
                          subtitle: Text(
                            'Listen on 0.0.0.0 instead of localhost',
                            style: TextStyle(
                              color: context.textD,
                              fontSize: 12,
                            ),
                          ),
                          value: apiServer.allInterfaces.value,
                          onChanged: starting
                              ? null
                              : (enabled) async {
                                  try {
                                    await apiServer.setAllInterfaces(enabled);
                                  } catch (e) {
                                    Get.snackbar(
                                      'Settings Error',
                                      e.toString(),
                                      snackPosition: SnackPosition.BOTTOM,
                                    );
                                  }
                                },
                          activeThumbColor: AppColors.orange,
                          contentPadding: EdgeInsets.zero,
                        ),
                        if (apiServer.allInterfaces.value)
                          Container(
                            margin: const EdgeInsets.only(top: 4, bottom: 8),
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: AppColors.orange.withOpacity(0.1),
                              border: Border.all(color: AppColors.orange.withOpacity(0.3)),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Row(
                              children: [
                                const Icon(Icons.warning_amber_rounded, size: 16, color: AppColors.orange),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: Text(
                                    'Anyone on your network can access your loaded model.',
                                    style: TextStyle(fontSize: 11, color: context.text),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        const SizedBox(height: 12),
                        Row(
                          children: [
                            _statusChip(context, statusText, statusColor),
                            const SizedBox(width: 10),
                            Expanded(
                              child: FutureBuilder<String?>(
                                future: apiServer.allInterfaces.value
                                    ? apiServer.getDeviceIp()
                                    : Future.value(null),
                                builder: (context, snapshot) {
                                  String url = apiServer.baseUrl;
                                  if (apiServer.allInterfaces.value && snapshot.hasData) {
                                    url = 'http://${snapshot.data}:${apiServer.port.value}/v1';
                                  }
                                  return SelectableText(
                                    url,
                                    style: TextStyle(
                                      color: context.text,
                                      fontSize: 13,
                                      fontWeight: FontWeight.w500,
                                    ),
                                  );
                                }
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        TextFormField(
                          key: ValueKey('api-port-${apiServer.port.value}'),
                          initialValue: apiServer.port.value.toString(),
                          keyboardType: TextInputType.number,
                          style: TextStyle(color: context.text, fontSize: 13),
                          decoration: InputDecoration(
                            labelText: 'Port',
                            helperText:
                                'Use API key "local" in clients that require one.',
                            labelStyle: TextStyle(color: context.textM),
                            helperStyle: TextStyle(
                              color: context.textD,
                              fontSize: 11,
                            ),
                            filled: true,
                            fillColor: context.bgInput,
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(8),
                              borderSide: BorderSide(color: context.border),
                            ),
                            enabledBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(8),
                              borderSide: BorderSide(color: context.border),
                            ),
                            focusedBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(8),
                              borderSide: const BorderSide(
                                color: AppColors.accent,
                              ),
                            ),
                          ),
                          onFieldSubmitted: (value) async {
                            final parsed = int.tryParse(value.trim());
                            if (parsed == null ||
                                parsed < 1024 ||
                                parsed > 65535) {
                              Get.snackbar(
                                'Invalid Port',
                                'Choose a port from 1024 to 65535.',
                                snackPosition: SnackPosition.BOTTOM,
                              );
                              return;
                            }
                            try {
                              await apiServer.setPort(parsed);
                              Get.snackbar(
                                'Local API Updated',
                                'Base URL is ${apiServer.baseUrl}',
                                snackPosition: SnackPosition.BOTTOM,
                              );
                            } catch (e) {
                              Get.snackbar(
                                'Local API Error',
                                e.toString(),
                                snackPosition: SnackPosition.BOTTOM,
                              );
                            }
                          },
                        ),
                        const SizedBox(height: 16),
                        SizedBox(
                          width: double.infinity,
                          child: OutlinedButton.icon(
                            onPressed: () {
                              Get.toNamed('/api-endpoints');
                            },
                            icon: const Icon(Icons.api_rounded, size: 18),
                            label: const Text('Sample Endpoints & Testing'),
                            style: OutlinedButton.styleFrom(
                              foregroundColor: context.text,
                              side: BorderSide(color: context.border),
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(8),
                              ),
                            ),
                          ),
                        ),
                        if (apiServer.errorMessage.value.isNotEmpty) ...[
                          const SizedBox(height: 10),
                          Text(
                            apiServer.errorMessage.value,
                            style: const TextStyle(
                              color: AppColors.red,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ],
                    ),
                  );
                }),
              ),
              ),

              const SizedBox(height: 28),

              // ── Storage ───────────────────────────────────
              _sectionHeader(context, 'Storage'),
              const SizedBox(height: 12),
              _card(
                context,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: [
                      Icon(
                        Icons.folder_outlined,
                        size: 20,
                        color: context.textM,
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          modelManager.modelsDir,
                          style: TextStyle(fontSize: 13, color: context.text),
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 28),

              // ── Danger Zone ───────────────────────────────
              _sectionHeader(context, 'Danger Zone'),
              const SizedBox(height: 12),
              OutlinedButton.icon(
                icon: const Icon(Icons.delete_forever_rounded, size: 18),
                label: const Text(
                  'Delete All Chats',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
                ),
                onPressed: () {
                  Get.dialog(
                    AlertDialog(
                      backgroundColor: context.bgPanel,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                      title: Text(
                        'Delete All Chats?',
                        style: TextStyle(color: context.text),
                      ),
                      content: Text(
                        'This cannot be undone.',
                        style: TextStyle(color: context.textM),
                      ),
                      actions: [
                        TextButton(
                          onPressed: () => Get.back(),
                          child: Text(
                            'Cancel',
                            style: TextStyle(color: context.textD),
                          ),
                        ),
                        ElevatedButton(
                          onPressed: () {
                            chatCtrl.chats.clear();
                            chatCtrl.activeChatId.value = null;
                            Get.back();
                            Get.snackbar(
                              'Done',
                              'All chats deleted.',
                              snackPosition: SnackPosition.BOTTOM,
                            );
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppColors.red,
                            elevation: 0,
                          ),
                          child: const Text(
                            'Delete All',
                            style: TextStyle(color: Colors.white),
                          ),
                        ),
                      ],
                    ),
                  );
                },
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppColors.red,
                  side: const BorderSide(color: AppColors.red),
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),

              const SizedBox(height: 12),
              
              OutlinedButton.icon(
                icon: const Icon(Icons.cleaning_services_rounded, size: 18),
                label: const Text(
                  'Clear Temporary Cache',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
                ),
                onPressed: () => Get.find<ModelController>().clearCache(),
                style: OutlinedButton.styleFrom(
                  foregroundColor: context.text,
                  side: BorderSide(color: context.border),
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),

              const SizedBox(height: 40),

              // ── About ─────────────────────────────────────
              Center(
                child: Column(
                  children: [
                    Text(
                      'Uncensored Local AI v2.0.0',
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w500,
                        color: context.textM,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Powered by llamadart + llama.cpp',
                      style: TextStyle(fontSize: 11, color: context.textD),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 28),

              // ── App Logs ──────────────────────────────────
              _sectionHeader(context, 'Debugging'),
              const SizedBox(height: 8),
              _card(
                context,
                child: ListTile(
                  leading: Container(
                    width: 36,
                    height: 36,
                    decoration: BoxDecoration(
                      color: AppColors.orange.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Icon(Icons.article_outlined,
                        size: 18, color: AppColors.orange),
                  ),
                  title: Text(
                    'App Logs',
                    style: TextStyle(color: context.text, fontSize: 14),
                  ),
                  subtitle: Text(
                    'View logs, errors & share with developers',
                    style: TextStyle(color: context.textD, fontSize: 12),
                  ),
                  trailing: Icon(Icons.arrow_forward_ios_rounded,
                      size: 14, color: context.textD),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 4,
                  ),
                  onTap: () => Get.toNamed('/logs'),
                ),
              ),
              const SizedBox(height: 8),
              _card(
                context,
                child: Obx(() {
                  final crashLog = Get.find<CrashLogService>();
                  final count = crashLog.entries.length;
                  return ListTile(
                    leading: Container(
                      width: 36,
                      height: 36,
                      decoration: BoxDecoration(
                        color: AppColors.red.withValues(alpha: 0.12),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Icon(Icons.bug_report_outlined,
                          size: 18, color: AppColors.red),
                    ),
                    title: Text(
                      'Crash Log',
                      style: TextStyle(color: context.text, fontSize: 14),
                    ),
                    subtitle: Text(
                      count == 0
                          ? 'No crashes recorded'
                          : '$count entr${count == 1 ? 'y' : 'ies'} — survives app restarts',
                      style: TextStyle(color: context.textD, fontSize: 12),
                    ),
                    trailing: Icon(Icons.arrow_forward_ios_rounded,
                        size: 14, color: context.textD),
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 4,
                    ),
                    onTap: () => Get.toNamed(AppRoutes.crashLog),
                  );
                }),
              ),
              const SizedBox(height: 8),
              _card(
                context,
                child: ListTile(
                  leading: Container(
                    width: 36,
                    height: 36,
                    decoration: BoxDecoration(
                      color: AppColors.accent.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Icon(Icons.speed_rounded,
                        size: 18, color: AppColors.accent),
                  ),
                  title: Text(
                    'Resource Monitor',
                    style: TextStyle(color: context.text, fontSize: 14),
                  ),
                  subtitle: Text(
                    'Live CPU, memory, and generation-speed graphs',
                    style: TextStyle(color: context.textD, fontSize: 12),
                  ),
                  trailing: Icon(Icons.arrow_forward_ios_rounded,
                      size: 14, color: context.textD),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 4,
                  ),
                  onTap: () => Get.toNamed(AppRoutes.resourceMonitor),
                ),
              ),

              const SizedBox(height: 32),
            ],
          ),
        ),
      ],
    );
  }

  Widget _sectionHeader(BuildContext context, String text) {
    return Text(
      text,
      style: TextStyle(
        fontSize: 13,
        fontWeight: FontWeight.w600,
        color: context.textM,
        letterSpacing: 0.3,
      ),
    );
  }

  Widget _card(BuildContext context, {required Widget child}) {
    return Container(
      decoration: BoxDecoration(
        color: context.bgPanel,
        border: Border.all(color: context.border),
        borderRadius: BorderRadius.circular(12),
      ),
      child: child,
    );
  }

  Widget _statusChip(BuildContext context, String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        border: Border.all(color: color.withValues(alpha: 0.35)),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        text,
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}

class _GenerationSettingsCard extends StatefulWidget {
  final ChatStorageService storage;
  const _GenerationSettingsCard({required this.storage});

  @override
  State<_GenerationSettingsCard> createState() => _GenerationSettingsCardState();
}

class _GenerationSettingsCardState extends State<_GenerationSettingsCard> {
  late int _contextSize;
  late double _topP;
  late int _topK;
  late double _minP;
  late double _repeatPenalty;
  late bool _enableThinking;
  late bool _toolsEnabled;
  late bool _advancedToolsEnabled;
  late bool _selfCritiqueEnabled;
  late bool _reasoningTraceEnabled;
  late bool _speculativeDecodingEnabled;
  late TextEditingController _customTemplateController;
  final _customTemplateFocus = FocusNode();
  bool _showAdvanced = false;

  @override
  void initState() {
    super.initState();
    _contextSize = widget.storage.contextSize;
    _topP = widget.storage.topP;
    _topK = widget.storage.topK;
    _minP = widget.storage.minP;
    _repeatPenalty = widget.storage.repeatPenalty;
    _enableThinking = widget.storage.enableModelThinking;
    _toolsEnabled = widget.storage.toolsEnabled;
    _advancedToolsEnabled = widget.storage.advancedToolsEnabled;
    _selfCritiqueEnabled = widget.storage.selfCritiqueEnabled;
    _reasoningTraceEnabled = widget.storage.reasoningTraceEnabled;
    _speculativeDecodingEnabled = widget.storage.speculativeDecodingEnabled;
    _customTemplateController =
        TextEditingController(text: widget.storage.customChatTemplate);
    // Persist on focus loss rather than every keystroke (same reasoning as
    // the sliders below persisting on release, not on every drag tick).
    _customTemplateFocus.addListener(() {
      if (!_customTemplateFocus.hasFocus) {
        widget.storage.customChatTemplate = _customTemplateController.text;
      }
    });
  }

  @override
  void dispose() {
    _customTemplateController.dispose();
    _customTemplateFocus.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Wrapped in Obx so the slider's bound updates the moment a model
    // finishes loading and reports its real trained context length —
    // without this, a Settings card kept alive in a tab's IndexedStack
    // would never see maxTrainedContext change after its first build.
    return Obx(() {
      // Bound the slider by the loaded model's own trained context length
      // when known, so it can't be dragged past what the model can actually
      // use. Falls back to the flat 8192 cap when no model is loaded yet, or
      // its metadata didn't report a context_length key.
      int sliderMax = 8192;
      try {
        final llm = Get.find<LlmService>();
        final trained = llm.maxTrainedContext.value;
        if (trained > 512) sliderMax = trained.clamp(512, 131072);
      } catch (_) {}
      if (_contextSize > sliderMax) sliderMax = _contextSize;

      return _buildCard(context, sliderMax);
    });
  }

  Widget _buildCard(BuildContext context, int sliderMax) {
    return Container(
      decoration: BoxDecoration(
        color: context.bgPanel,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: context.border),
      ),
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Context size ──
          TutorialTarget(
            id: 'settings.context_size',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('Context Size',
                  style: TextStyle(color: context.text, fontSize: 14)),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: context.bgInput,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  '$_contextSize tokens',
                  style: TextStyle(color: context.text, fontWeight: FontWeight.bold, fontSize: 12),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            'How much conversation the model can see at once. Higher uses more '
            'RAM and takes longer to process on load — reload the model after '
            'changing this.',
            style: TextStyle(fontSize: 11, color: context.textD, height: 1.4),
          ),
          Slider(
            value: _contextSize.toDouble().clamp(512, sliderMax.toDouble()),
            min: 512,
            max: sliderMax.toDouble(),
            divisions: ((sliderMax - 512) / 512).round().clamp(1, 255),
            activeColor: AppColors.accent,
            inactiveColor: context.border,
            label: '$_contextSize',
            // Persist on release, not every tick — this writes to disk
            // (Hive), and onChanged fires many times per second while
            // dragging.
            onChanged: (v) => setState(() => _contextSize = v.round()),
            onChangeEnd: (v) => widget.storage.contextSize = v.round(),
          ),
              ],
            ),
          ),

          const SizedBox(height: 8),

          // ── Model thinking toggle ──
          TutorialTarget(
            id: 'settings.model_reasoning',
            child: SwitchListTile(
            title: Text('Model Reasoning',
                style: TextStyle(color: context.text, fontSize: 14)),
            subtitle: Text(
              'Let reasoning-capable models show their thinking. Off can mean '
              'faster, shorter replies on models that support toggling it.',
              style: TextStyle(color: context.textD, fontSize: 11),
            ),
            value: _enableThinking,
            onChanged: (v) {
              setState(() => _enableThinking = v);
              widget.storage.enableModelThinking = v;
            },
            activeThumbColor: AppColors.accent,
            contentPadding: EdgeInsets.zero,
          ),
          ),

          const SizedBox(height: 8),

          // ── Tool calling toggle ──
          TutorialTarget(
            id: 'settings.tool_calling',
            child: SwitchListTile(
            title: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text('Tool Calling',
                    style: TextStyle(color: context.text, fontSize: 14)),
                const SizedBox(width: 8),
                const ModelInjectionBadge(),
              ],
            ),
            subtitle: Text(
              'Give the model access to a few offline tools: current date/'
              'time, a calculator, and memory search. With Persistent '
              'Memory also on, this includes memory write tools — the '
              'model can save, edit, or supersede memories on its own, '
              'not just read them. Adds some prompt overhead to every '
              'turn even when unused.',
              style: TextStyle(color: context.textD, fontSize: 11),
            ),
            value: _toolsEnabled,
            onChanged: (v) {
              setState(() => _toolsEnabled = v);
              widget.storage.toolsEnabled = v;
            },
            activeThumbColor: AppColors.accent,
            contentPadding: EdgeInsets.zero,
          ),
          ),

          const SizedBox(height: 8),

          // ── Advanced tools toggle ──
          TutorialTarget(
            id: 'settings.advanced_tools',
            child: SwitchListTile(
            title: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text('Advanced Tools',
                    style: TextStyle(color: context.text, fontSize: 14)),
                const SizedBox(width: 8),
                const ModelInjectionBadge(),
              ],
            ),
            subtitle: Text(
              'Clipboard read/write, in-app reminders, and recalling its own '
              'past reasoning. All on-device, no network. Requires Tool '
              'Calling above to be on.',
              style: TextStyle(color: context.textD, fontSize: 11),
            ),
            value: _advancedToolsEnabled,
            onChanged: (v) {
              setState(() => _advancedToolsEnabled = v);
              widget.storage.advancedToolsEnabled = v;
            },
            activeThumbColor: AppColors.accent,
            contentPadding: EdgeInsets.zero,
          ),
          ),

          const SizedBox(height: 8),

          Text(
            'Max tool rounds per message',
            style: TextStyle(color: context.text, fontSize: 14),
          ),
          const SizedBox(height: 4),
          Text(
            'How many tool-call round trips one message can trigger before '
            'the app forces a final answer — raise this if a task '
            'genuinely needs several steps (search memory, then save a '
            'few facts, then check something else); a model that keeps '
            'calling tools instead of answering will just spend more time '
            'and battery per round allowed.',
            style: TextStyle(color: context.textD, fontSize: 11, height: 1.4),
          ),
          const SizedBox(height: 8),
          _MaxToolRoundsRow(storage: widget.storage),

          const SizedBox(height: 8),

          // ── Self-critique toggle ──
          TutorialTarget(
            id: 'settings.self_critique',
            child: SwitchListTile(
            title: Text('Self-Critique',
                style: TextStyle(color: context.text, fontSize: 14)),
            subtitle: Text(
              'After each reply, a short background check (helper model if '
              'armed, otherwise the main model) looks for contradictions '
              'with memory or unsupported confident claims. Never edits or '
              'blocks the reply — just a soft note if it finds something. '
              'Costs one extra background generation per turn.',
              style: TextStyle(color: context.textD, fontSize: 11),
            ),
            value: _selfCritiqueEnabled,
            onChanged: (v) {
              setState(() => _selfCritiqueEnabled = v);
              widget.storage.selfCritiqueEnabled = v;
            },
            activeThumbColor: AppColors.accent,
            contentPadding: EdgeInsets.zero,
          ),
          ),

          const SizedBox(height: 8),

          // ── Reasoning trace toggle ──
          TutorialTarget(
            id: 'settings.reasoning_trace',
            child: SwitchListTile(
            title: Text('Reasoning Trace',
                style: TextStyle(color: context.text, fontSize: 14)),
            subtitle: Text(
              'Distill each turn\'s chain-of-thought into a short gist and '
              'keep it in its own lane, separate from fact memory, so the '
              'model can recall how it approached something before — not '
              'just what it concluded.',
              style: TextStyle(color: context.textD, fontSize: 11),
            ),
            value: _reasoningTraceEnabled,
            onChanged: (v) {
              setState(() => _reasoningTraceEnabled = v);
              widget.storage.reasoningTraceEnabled = v;
            },
            activeThumbColor: AppColors.accent,
            contentPadding: EdgeInsets.zero,
          ),
          ),

          const SizedBox(height: 8),

          // ── Speculative decoding toggle ──
          TutorialTarget(
            id: 'settings.speculative_decoding',
            child: SwitchListTile(
            title: Text('Speculative Decoding',
                style: TextStyle(color: context.text, fontSize: 14)),
            subtitle: Text(
              'n-gram self-speculative decoding — drafts from tokens '
              'already in the conversation and verifies them in one batch. '
              'Never changes what gets generated, only how fast, at least '
              'in theory — this app has no on-device measurement of it '
              'yet, so it starts off. Worth trying and watching your '
              'tokens/sec before and after.',
              style: TextStyle(color: context.textD, fontSize: 11),
            ),
            value: _speculativeDecodingEnabled,
            onChanged: (v) {
              setState(() => _speculativeDecodingEnabled = v);
              widget.storage.speculativeDecodingEnabled = v;
            },
            activeThumbColor: AppColors.accent,
            contentPadding: EdgeInsets.zero,
          ),
          ),

          const SizedBox(height: 8),

          // ── Advanced sampling toggle ──
          InkWell(
            onTap: () => setState(() => _showAdvanced = !_showAdvanced),
            borderRadius: BorderRadius.circular(8),
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: Row(
                children: [
                  Icon(
                    _showAdvanced ? Icons.expand_less : Icons.expand_more,
                    size: 18,
                    color: context.textM,
                  ),
                  const SizedBox(width: 4),
                  Text(
                    'Advanced Sampling',
                    style: TextStyle(
                      color: context.textM,
                      fontSize: 13,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
          ),

          if (_showAdvanced) ...[
            const SizedBox(height: 12),
            _samplingSlider(
              context,
              label: 'Top-P',
              value: _topP,
              min: 0.0,
              max: 1.0,
              divisions: 20,
              display: _topP.toStringAsFixed(2),
              onChanged: (v) => setState(() => _topP = v),
              onChangeEnd: (v) => widget.storage.topP = v,
            ),
            _samplingSlider(
              context,
              label: 'Top-K',
              value: _topK.toDouble(),
              min: 0,
              max: 100,
              divisions: 100,
              display: '$_topK',
              onChanged: (v) => setState(() => _topK = v.round()),
              onChangeEnd: (v) => widget.storage.topK = v.round(),
            ),
            _samplingSlider(
              context,
              label: 'Min-P',
              value: _minP,
              min: 0.0,
              max: 0.5,
              divisions: 25,
              display: _minP.toStringAsFixed(2),
              onChanged: (v) => setState(() => _minP = v),
              onChangeEnd: (v) => widget.storage.minP = v,
            ),
            Text(
              'These narrow which tokens the model can pick from at each step. '
              'Defaults (Top-P 0.95, Top-K 40, Min-P 0.05) are sane for most '
              'models — only change these if you know what you\'re tuning.',
              style: TextStyle(color: context.textD, fontSize: 11, height: 1.4),
            ),
            const SizedBox(height: 12),
            _samplingSlider(
              context,
              label: 'Repeat',
              value: _repeatPenalty,
              min: 1.0,
              max: 2.0,
              divisions: 20,
              display: _repeatPenalty.toStringAsFixed(2),
              onChanged: (v) => setState(() => _repeatPenalty = v),
              onChangeEnd: (v) => widget.storage.repeatPenalty = v,
            ),
            Text(
              'Repeat penalty (default 1.10) discourages the model from '
              'reusing tokens it already used.',
              style: TextStyle(color: context.textD, fontSize: 11, height: 1.4),
            ),
            const SizedBox(height: 16),
            Text('Custom Chat Template',
                style: TextStyle(color: context.text, fontSize: 14)),
            const SizedBox(height: 4),
            Text(
              'Overrides the Jinja chat template baked into the model\'s '
              'GGUF — only useful if a specific model\'s shipped template is '
              'broken or missing. Leave empty to use the model\'s own '
              'template (the normal case). Applies on next model load.',
              style: TextStyle(color: context.textD, fontSize: 11, height: 1.4),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _customTemplateController,
              focusNode: _customTemplateFocus,
              maxLines: 4,
              minLines: 2,
              style: TextStyle(
                color: context.text,
                fontSize: 12,
                fontFamily: 'monospace',
              ),
              decoration: InputDecoration(
                hintText: '{% for message in messages %}...',
                hintStyle: TextStyle(color: context.textD, fontSize: 12),
                filled: true,
                fillColor: context.bgInput,
                contentPadding: const EdgeInsets.all(10),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                  borderSide: BorderSide(color: context.border),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                  borderSide: BorderSide(color: context.border),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _samplingSlider(
    BuildContext context, {
    required String label,
    required double value,
    required double min,
    required double max,
    required int divisions,
    required String display,
    required ValueChanged<double> onChanged,
    required ValueChanged<double> onChangeEnd,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          SizedBox(
            width: 56,
            child: Text(label, style: TextStyle(color: context.text, fontSize: 13)),
          ),
          Expanded(
            child: Slider(
              value: value,
              min: min,
              max: max,
              divisions: divisions,
              activeColor: AppColors.accent,
              inactiveColor: context.border,
              label: display,
              onChanged: onChanged,
              onChangeEnd: onChangeEnd,
            ),
          ),
          SizedBox(
            width: 40,
            child: Text(
              display,
              textAlign: TextAlign.end,
              style: TextStyle(color: context.textM, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }
}

/// Real measured results from LlmService.benchmarkBackends — ranked
/// fastest first, each with a one-tap "Use This" that only appears for
/// backends that actually completed a generation (a failed/errored
/// backend has nothing to apply).
class _BenchmarkResultsList extends StatelessWidget {
  final List<BackendBenchmarkResult> results;
  final void Function(BackendBenchmarkResult) onApply;
  final String Function(String) backendLabel;

  const _BenchmarkResultsList({
    required this.results,
    required this.onApply,
    required this.backendLabel,
  });

  @override
  Widget build(BuildContext context) {
    final ranked = results.toList()
      ..sort((a, b) {
        if (a.succeeded && !b.succeeded) return -1;
        if (!a.succeeded && b.succeeded) return 1;
        if (!a.succeeded && !b.succeeded) return 0;
        return b.tokensPerSecond!.compareTo(a.tokensPerSecond!);
      });
    final fastest = ranked.isNotEmpty && ranked.first.succeeded ? ranked.first : null;

    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: context.bgHover.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: context.borderFaint),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          for (final r in ranked)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: Row(
                children: [
                  Icon(
                    r.succeeded ? Icons.check_circle_outline_rounded : Icons.error_outline_rounded,
                    size: 14,
                    color: r == fastest
                        ? AppColors.green
                        : r.succeeded
                            ? context.textM
                            : AppColors.red,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      backendLabel(r.backend),
                      style: TextStyle(
                        color: context.text,
                        fontSize: 13,
                        fontWeight: r == fastest ? FontWeight.w700 : FontWeight.normal,
                      ),
                    ),
                  ),
                  Text(
                    r.succeeded
                        ? '${r.tokensPerSecond!.toStringAsFixed(1)} t/s'
                        : 'failed',
                    style: TextStyle(
                      color: r.succeeded ? context.textM : AppColors.red,
                      fontSize: 12,
                    ),
                  ),
                  if (r.succeeded) ...[
                    const SizedBox(width: 8),
                    TextButton(
                      onPressed: () => onApply(r),
                      style: TextButton.styleFrom(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        minimumSize: Size.zero,
                        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                      ),
                      child: const Text('Use This', style: TextStyle(fontSize: 11)),
                    ),
                  ],
                ],
              ),
            ),
        ],
      ),
    );
  }
}

class _RemindersCard extends StatelessWidget {
  final ReminderService reminders;

  const _RemindersCard({required this.reminders});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: context.bgPanel,
        border: Border.all(color: context.border),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Obx(() {
        final all = reminders.reminders.toList()
          ..sort((a, b) => a.dueAt.compareTo(b.dueAt));
        if (all.isEmpty) {
          return Padding(
            padding: const EdgeInsets.all(16),
            child: Text(
              'No reminders set.',
              style: TextStyle(color: context.textD, fontSize: 12),
            ),
          );
        }
        return Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            for (final r in all)
              ListTile(
                dense: true,
                title: Text(r.text, style: TextStyle(color: context.text, fontSize: 13)),
                subtitle: Text(
                  r.fired
                      ? 'Due ${_formatDueAt(r.dueAt)} — already surfaced'
                      : 'Due ${_formatDueAt(r.dueAt)}',
                  style: TextStyle(color: context.textD, fontSize: 11),
                ),
                trailing: IconButton(
                  icon: Icon(Icons.close_rounded, size: 18, color: context.textD),
                  tooltip: 'Dismiss',
                  onPressed: () => reminders.dismiss(r.id),
                ),
              ),
          ],
        );
      }),
    );
  }

  String _formatDueAt(DateTime dueAt) {
    final now = DateTime.now();
    final diff = dueAt.difference(now);
    if (diff.inMinutes.abs() < 1) return 'now';
    if (diff.isNegative) {
      final ago = -diff;
      if (ago.inDays > 0) return '${ago.inDays}d ago';
      if (ago.inHours > 0) return '${ago.inHours}h ago';
      return '${ago.inMinutes}m ago';
    }
    if (diff.inDays > 0) return 'in ${diff.inDays}d';
    if (diff.inHours > 0) return 'in ${diff.inHours}h';
    return 'in ${diff.inMinutes}m';
  }
}

class _PersistentMemoryCard extends StatefulWidget {
  final ChatStorageService storage;
  final MemoryService memory;
  final EmbeddingService embedding;
  final HelperLlmService helper;

  const _PersistentMemoryCard({
    required this.storage,
    required this.memory,
    required this.embedding,
    required this.helper,
  });

  @override
  State<_PersistentMemoryCard> createState() => _PersistentMemoryCardState();
}

class _PersistentMemoryCardState extends State<_PersistentMemoryCard> {
  late bool _enabled;
  late String _secondOpinionFilename;

  @override
  void initState() {
    super.initState();
    _enabled = widget.storage.persistentMemoryEnabled;
    _secondOpinionFilename = widget.storage.secondOpinionModelFilename;
  }

  @override
  Widget build(BuildContext context) {
    return _cardDecoration(
      context,
      child: Obx(() {
        final hasEmbeddingModel = widget.embedding.isLoaded.value;
        return Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SwitchListTile(
                title: Text(
                  'Persistent memory',
                  style: TextStyle(color: context.text, fontSize: 14),
                ),
                subtitle: Text(
                  hasEmbeddingModel
                      ? 'Embedding model: ${widget.embedding.loadedModelFilename}'
                      : 'Set an embedding model in the Models tab first',
                  style: TextStyle(color: context.textD, fontSize: 12),
                ),
                secondary: Icon(
                  Icons.psychology_alt_rounded,
                  color: _enabled ? AppColors.accent : context.textM,
                ),
                value: _enabled,
                onChanged: !hasEmbeddingModel
                    ? null
                    : (value) {
                        setState(() => _enabled = value);
                        widget.storage.persistentMemoryEnabled = value;
                      },
                activeThumbColor: AppColors.accent,
                contentPadding: EdgeInsets.zero,
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  Icon(Icons.folder_outlined, size: 14, color: context.textD),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      widget.memory.memoryDirPath,
                      style: TextStyle(fontSize: 11, color: context.textD),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: () => Get.toNamed(AppRoutes.memory),
                  icon: const Icon(Icons.list_alt_rounded, size: 18),
                  label: Obx(
                    () => Text('View Memories (${widget.memory.entries.length})'),
                  ),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: context.text,
                    side: BorderSide(color: context.border),
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Text('Memory Extraction Helper',
                  style: TextStyle(color: context.text, fontSize: 14)),
              const SizedBox(height: 4),
              Text(
                'A small, fast model dedicated to distilling memorable turns '
                'into notes — runs in its own engine, separate from the main '
                'model, so this background task never has to wait for (or '
                'block) the main conversation. Optional: without one, this '
                'falls back to the main model, which is slower.',
                style: TextStyle(color: context.textD, fontSize: 11, height: 1.4),
              ),
              const SizedBox(height: 8),
              Obx(() {
                final loaded = widget.helper.isLoaded.value;
                final filename = widget.helper.loadedModelFilename;
                return Row(
                  children: [
                    Expanded(
                      child: Text(
                        loaded ? filename : 'Not set — using main model',
                        style: TextStyle(
                          color: loaded ? context.text : context.textD,
                          fontSize: 12,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    const SizedBox(width: 8),
                    OutlinedButton(
                      onPressed: () => _pickHelperModel(context),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: context.text,
                        side: BorderSide(color: context.border),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      child: Text(loaded ? 'Change' : 'Set'),
                    ),
                    if (loaded) ...[
                      const SizedBox(width: 8),
                      IconButton(
                        icon: const Icon(Icons.close, size: 18),
                        color: context.textD,
                        onPressed: () =>
                            Get.find<ModelController>().clearHelperModel(),
                        tooltip: 'Stop using a helper model',
                      ),
                    ],
                  ],
                );
              }),
              const SizedBox(height: 16),
              Text('Memory Extraction Instructions',
                  style: TextStyle(color: context.text, fontSize: 14)),
              const SizedBox(height: 4),
              Text(
                'What the extraction model judges as "worth remembering". '
                'Leave blank to use the built-in default.',
                style: TextStyle(color: context.textD, fontSize: 11, height: 1.4),
              ),
              const SizedBox(height: 8),
              _ExtractionGuidanceField(storage: widget.storage),
              const SizedBox(height: 16),
              TutorialTarget(
                id: 'settings.memory_verification',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
              Text('Memory Verification', style: TextStyle(color: context.text, fontSize: 14)),
              const SizedBox(height: 4),
              Text(
                'Runs 5 fixed test exchanges through the real extraction '
                'pipeline (using whatever model is currently armed) and '
                'checks the results against known-correct answers — '
                'including one negative case that should capture nothing. '
                'Nothing here is written to memory.',
                style: TextStyle(color: context.textD, fontSize: 11, height: 1.4),
              ),
              const SizedBox(height: 8),
              const _MemoryVerificationButton(),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              Text('Memory Health Sweep',
                  style: TextStyle(color: context.text, fontSize: 14)),
              const SizedBox(height: 4),
              Text(
                'On this interval, runs a real background check: verifies '
                'every configured model is actually loaded, probes the '
                'embedding model with a real request, and flushes any '
                'pending memory write to disk. If everything checks out '
                'and a helper model is armed, it also looks for '
                'connections across recent memories, which can add a new '
                'note. Runs silently by default — turn on "Ask before each '
                'sweep" below if you\'d rather approve each cycle. Set to '
                '0 to disable.',
                style: TextStyle(color: context.textD, fontSize: 11, height: 1.4),
              ),
              const SizedBox(height: 8),
              _SweepIntervalRow(storage: widget.storage),
              const SizedBox(height: 8),
              _SweepConfirmationToggle(storage: widget.storage),
              const SizedBox(height: 16),
              Text('Attention Reflection',
                  style: TextStyle(color: context.text, fontSize: 14)),
              const SizedBox(height: 4),
              Text(
                'Every this-many sweep cycles, spins up the main model to '
                'reflect in its own words on a sample of recent memories — '
                'what it remembers, and which feel more or less important '
                'to keep in mind. If a helper model is armed, that model '
                'then reads the reflection and turns it into a per-memory '
                'attention score, visible in the memory browser and graph. '
                'Spends a real main-model generation, so it runs far less '
                'often than the sweep itself.',
                style: TextStyle(color: context.textD, fontSize: 11, height: 1.4),
              ),
              const SizedBox(height: 8),
              _CadenceRow(
                getValue: () => widget.storage.attentionReflectionEveryNSweeps,
                setValue: (v) => widget.storage.attentionReflectionEveryNSweeps = v,
              ),
              const SizedBox(height: 16),
              Text('Frame Analysis',
                  style: TextStyle(color: context.text, fontSize: 14)),
              const SizedBox(height: 4),
              Text(
                'Every this-many sweep cycles, builds one "Frame" — a '
                'summary standing in for everything currently remembered — '
                'then has BOTH the main and helper model independently '
                'interpret it with the exact same instructions, looking for '
                'the deeper pattern behind how it all connects. Where they '
                'agree, that becomes a new insight. Where they don\'t, both '
                'answers are kept as a flagged, unresolved tension for you '
                'to review in the memory browser — a toast tells you which '
                'happened. Heavier than attention reflection (a full '
                'summarization pass plus two model generations), so it '
                'defaults to running less often.',
                style: TextStyle(color: context.textD, fontSize: 11, height: 1.4),
              ),
              const SizedBox(height: 8),
              _CadenceRow(
                getValue: () => widget.storage.frameAnalysisEveryNSweeps,
                setValue: (v) => widget.storage.frameAnalysisEveryNSweeps = v,
              ),
              const SizedBox(height: 16),
              Text('Second-Opinion Rotation Model',
                  style: TextStyle(color: context.text, fontSize: 14)),
              const SizedBox(height: 4),
              Text(
                'Optional — a spare downloaded model (one that isn\'t your '
                'chat model or your helper) that takes alternating turns as '
                'Frame analysis\'s second opinion, instead of always using '
                'the helper above. Unlike the helper, this model is NOT '
                'kept loaded: it\'s loaded only for its one turn, then torn '
                'down right after. Since the main model, the helper, and '
                'this one can all briefly be in memory together during '
                'that turn, only set this if your device has the RAM to '
                'spare for a moment.',
                style: TextStyle(color: context.textD, fontSize: 11, height: 1.4),
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: Text(
                      _secondOpinionFilename.isEmpty
                          ? 'Not set — always uses the helper'
                          : _secondOpinionFilename,
                      style: TextStyle(
                        color: _secondOpinionFilename.isEmpty
                            ? context.textD
                            : context.text,
                        fontSize: 12,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton(
                    onPressed: () => _pickSecondOpinionModel(context),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: context.text,
                      side: BorderSide(color: context.border),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    child: Text(_secondOpinionFilename.isEmpty ? 'Set' : 'Change'),
                  ),
                  if (_secondOpinionFilename.isNotEmpty) ...[
                    const SizedBox(width: 8),
                    IconButton(
                      icon: const Icon(Icons.close, size: 18),
                      color: context.textD,
                      onPressed: () {
                        Get.find<ModelController>().clearSecondOpinionModel();
                        setState(() => _secondOpinionFilename = '');
                      },
                      tooltip: 'Stop using a rotation model',
                    ),
                  ],
                ],
              ),
            ],
          ),
        );
      }),
    );
  }

  void _pickSecondOpinionModel(BuildContext context) {
    final manager = Get.find<ModelManager>();
    // Excludes whatever's already the main chat model or the helper — the
    // whole point of this rotation is a THIRD, otherwise-idle model taking
    // a turn; offering the main or helper model here would let a user pick
    // a multi-billion-parameter model that's already resident, so its
    // rotation turn loads a second copy of it as a third concurrent
    // engine — exactly the RAM risk this section's own description warns
    // against, not a spare model finally getting a job.
    final mainModelFilename = Get.find<LlmService>().loadedModelFilename;
    final helperFilename = widget.storage.helperModelFilename;
    final candidates = manager.downloadedModels
        .where((f) =>
            manager.kindOf(f) == ModelKind.chat &&
            f != mainModelFilename &&
            f != helperFilename)
        .toList();

    if (candidates.isEmpty) {
      Get.snackbar(
        'No Spare Chat Models Downloaded',
        'Download a chat model that isn\'t already your main chat model or '
            'your helper to use it as the second-opinion rotation model.',
        snackPosition: SnackPosition.BOTTOM,
      );
      return;
    }

    showModalBottomSheet(
      context: context,
      backgroundColor: context.bgPanel,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (sheetContext) {
        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Padding(
                padding: const EdgeInsets.all(16),
                child: Text(
                  'Choose a Second-Opinion Model',
                  style: TextStyle(
                    color: sheetContext.text,
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              Flexible(
                child: ListView.builder(
                  shrinkWrap: true,
                  itemCount: candidates.length,
                  itemBuilder: (_, i) {
                    final filename = candidates[i];
                    return ListTile(
                      leading: Icon(Icons.psychology_alt_outlined, color: sheetContext.textM),
                      title: Text(
                        filename,
                        style: TextStyle(color: sheetContext.text, fontSize: 13),
                      ),
                      onTap: () {
                        Navigator.of(sheetContext).pop();
                        Get.find<ModelController>().setSecondOpinionModel(filename);
                        setState(() => _secondOpinionFilename = filename);
                      },
                    );
                  },
                ),
              ),
              const SizedBox(height: 8),
            ],
          ),
        );
      },
    );
  }

  void _pickHelperModel(BuildContext context) {
    final manager = Get.find<ModelManager>();
    final candidates = manager.downloadedModels
        .where((f) => manager.kindOf(f) == ModelKind.chat)
        .toList();

    if (candidates.isEmpty) {
      Get.snackbar(
        'No Chat Models Downloaded',
        'Download a small chat model first to use it as the memory helper.',
        snackPosition: SnackPosition.BOTTOM,
      );
      return;
    }

    showModalBottomSheet(
      context: context,
      backgroundColor: context.bgPanel,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (sheetContext) {
        return SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Padding(
                padding: const EdgeInsets.all(16),
                child: Text(
                  'Choose a Helper Model',
                  style: TextStyle(
                    color: sheetContext.text,
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              Flexible(
                child: ListView.builder(
                  shrinkWrap: true,
                  itemCount: candidates.length,
                  itemBuilder: (_, i) {
                    final filename = candidates[i];
                    return ListTile(
                      leading: Icon(Icons.bolt_rounded, color: sheetContext.textM),
                      title: Text(
                        filename,
                        style: TextStyle(color: sheetContext.text, fontSize: 13),
                      ),
                      onTap: () {
                        Navigator.of(sheetContext).pop();
                        Get.find<ModelController>().setHelperModel(filename);
                      },
                    );
                  },
                ),
              ),
              const SizedBox(height: 8),
            ],
          ),
        );
      },
    );
  }

  Widget _cardDecoration(BuildContext context, {required Widget child}) {
    return Container(
      decoration: BoxDecoration(
        color: context.bgPanel,
        border: Border.all(color: context.border),
        borderRadius: BorderRadius.circular(12),
      ),
      child: child,
    );
  }
}

/// Editable override for the memory-extraction model's "what counts as
/// worth remembering" instructions — mirrors the Global System Prompt
/// field's pattern (plain multi-line TextField, saved on change, a Reset
/// action). Only the judgment-call portion is editable; the JSON-format
/// contract that follows it in the real prompt is fixed and never shown
/// here, since editing that away would silently break extraction parsing.
/// Runs [ChatController.runMemoryVerificationSuite] on demand and shows the
/// result — pass/fail count plus per-case detail — in a dialog. A real
/// generation per test case, so this can take a while on slow hardware;
/// the button disables itself and shows a spinner while running rather
/// than allowing overlapping runs.
class _MemoryVerificationButton extends StatefulWidget {
  const _MemoryVerificationButton();

  @override
  State<_MemoryVerificationButton> createState() => _MemoryVerificationButtonState();
}

class _MemoryVerificationButtonState extends State<_MemoryVerificationButton> {
  bool _running = false;

  Future<void> _run() async {
    setState(() => _running = true);
    try {
      final result = await Get.find<ChatController>().runMemoryVerificationSuite();
      if (!mounted) return;
      showDialog(
        context: context,
        builder: (dialogContext) => AlertDialog(
          backgroundColor: dialogContext.bgPanel,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          title: Text(
            '${result.passed}/${result.total} passed',
            style: TextStyle(color: dialogContext.text),
          ),
          content: SizedBox(
            width: double.maxFinite,
            child: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  for (final d in result.details)
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 6),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Icon(
                            d.passed ? Icons.check_circle : Icons.cancel,
                            size: 16,
                            color: d.passed ? AppColors.green : AppColors.red,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(d.description,
                                    style: TextStyle(fontSize: 13, color: dialogContext.text)),
                                Text('Got: "${d.actual}"',
                                    style: TextStyle(fontSize: 11, color: dialogContext.textD)),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                ],
              ),
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(dialogContext),
              child: Text('Close', style: TextStyle(color: dialogContext.textD)),
            ),
          ],
        ),
      );
    } finally {
      if (mounted) setState(() => _running = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton.icon(
        onPressed: _running ? null : _run,
        icon: _running
            ? const SizedBox(
                width: 14,
                height: 14,
                child: CircularProgressIndicator(strokeWidth: 2),
              )
            : const Icon(Icons.fact_check_outlined, size: 18),
        label: Text(_running ? 'Running…' : 'Run Verification'),
        style: OutlinedButton.styleFrom(
          foregroundColor: context.text,
          side: BorderSide(color: context.border),
          padding: const EdgeInsets.symmetric(vertical: 12),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        ),
      ),
    );
  }
}

class _ExtractionGuidanceField extends StatefulWidget {
  final ChatStorageService storage;
  const _ExtractionGuidanceField({required this.storage});

  @override
  State<_ExtractionGuidanceField> createState() => _ExtractionGuidanceFieldState();
}

class _ExtractionGuidanceFieldState extends State<_ExtractionGuidanceField> {
  late final TextEditingController _controller;
  final _focus = FocusNode();

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: widget.storage.memoryExtractionGuidance);
    // Persist on focus loss rather than every keystroke — same reasoning
    // as the custom chat template field in _HardwareSettingsCard: a Hive
    // box write on every character typed is needless overhead on a phone
    // for a field that's only ever read once per turn, not live-bound to
    // anything that needs to react mid-typing.
    _focus.addListener(() {
      if (!_focus.hasFocus) {
        widget.storage.memoryExtractionGuidance = _controller.text;
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _focus.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        TextField(
          controller: _controller,
          focusNode: _focus,
          maxLines: 4,
          // A long override here directly eats into the extraction
          // prompt's token budget (see the dynamic overhead calculation in
          // ChatController._extractAndRememberFromTurn) — capped so a
          // pasted wall of text can't crowd out the actual exchange being
          // extracted from.
          maxLength: 500,
          style: TextStyle(fontSize: 13, color: context.text, height: 1.5),
          decoration: InputDecoration(
            hintText: defaultMemoryExtractionGuidance,
            hintStyle: TextStyle(color: context.textD, fontSize: 12),
            filled: true,
            fillColor: context.bgInput,
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: BorderSide(color: context.border),
            ),
            enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: BorderSide(color: context.border),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: const BorderSide(color: AppColors.accent),
            ),
          ),
          // Reset button's visibility tracks emptiness; the actual storage
          // write happens on focus loss (see initState), not here.
          onChanged: (_) => setState(() {}),
        ),
        if (_controller.text.isNotEmpty)
          Align(
            alignment: Alignment.centerLeft,
            child: TextButton.icon(
              onPressed: () {
                setState(() => _controller.clear());
                widget.storage.memoryExtractionGuidance = '';
              },
              icon: const Icon(Icons.clear_rounded, size: 16),
              label: const Text('Reset to default'),
              style: TextButton.styleFrom(foregroundColor: context.textD),
            ),
          ),
      ],
    );
  }
}

/// Stepper for [ChatStorageService.maxToolRounds].
class _MaxToolRoundsRow extends StatefulWidget {
  final ChatStorageService storage;
  const _MaxToolRoundsRow({required this.storage});

  @override
  State<_MaxToolRoundsRow> createState() => _MaxToolRoundsRowState();
}

class _MaxToolRoundsRowState extends State<_MaxToolRoundsRow> {
  late int _rounds;

  @override
  void initState() {
    super.initState();
    _rounds = widget.storage.maxToolRounds;
  }

  void _set(int value) {
    final clamped = value.clamp(1, 30);
    setState(() => _rounds = clamped);
    widget.storage.maxToolRounds = clamped;
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: Text(
            '$_rounds round${_rounds == 1 ? '' : 's'}',
            style: TextStyle(color: context.text, fontSize: 13),
          ),
        ),
        IconButton(
          icon: Icon(Icons.remove_circle_outline, size: 20, color: context.textM),
          onPressed: () => _set(_rounds - 1),
        ),
        IconButton(
          icon: Icon(Icons.add_circle_outline, size: 20, color: context.textM),
          onPressed: () => _set(_rounds + 1),
        ),
      ],
    );
  }
}

/// Preset picker for the memory-health sweep interval, 5 seconds to 60
/// minutes (plus off). A plain +/- stepper doesn't work across a range
/// this wide — a step small enough to be useful at 5s would take forever
/// to reach 3600s — so this steps through a fixed list of sensible presets
/// instead. Reschedules the live timer on ChatController immediately on
/// change, rather than only taking effect after an app restart.
class _SweepIntervalRow extends StatefulWidget {
  final ChatStorageService storage;
  const _SweepIntervalRow({required this.storage});

  @override
  State<_SweepIntervalRow> createState() => _SweepIntervalRowState();
}

class _SweepIntervalRowState extends State<_SweepIntervalRow> {
  // 0 = off, then 5s up to 60min. Kept as its own ordered list (not a
  // formula) so the displayed label and the stored value can never drift
  // apart from a rounding difference.
  static const _presets = <int>[
    0, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600,
  ];

  late int _seconds;

  @override
  void initState() {
    super.initState();
    _seconds = widget.storage.memorySweepIntervalSeconds;
  }

  String _label(int seconds) {
    if (seconds <= 0) return 'Disabled';
    if (seconds < 60) return 'Every ${seconds}s';
    final minutes = seconds ~/ 60;
    return 'Every $minutes min';
  }

  void _step(int delta) {
    var index = _presets.indexOf(_seconds);
    if (index == -1) {
      // A value from an older build (minutes-based) or hand-edited store —
      // land on the closest preset rather than fail to move at all.
      index = _presets.indexWhere((p) => p >= _seconds);
      if (index == -1) index = _presets.length - 1;
    }
    final nextIndex = (index + delta).clamp(0, _presets.length - 1);
    final value = _presets[nextIndex];
    setState(() => _seconds = value);
    widget.storage.memorySweepIntervalSeconds = value;
    try {
      Get.find<ChatController>().rescheduleMemorySweep();
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: Text(
            _label(_seconds),
            style: TextStyle(color: context.text, fontSize: 13),
          ),
        ),
        IconButton(
          icon: Icon(Icons.remove_circle_outline, size: 20, color: context.textM),
          onPressed: () => _step(-1),
        ),
        IconButton(
          icon: Icon(Icons.add_circle_outline, size: 20, color: context.textM),
          onPressed: () => _step(1),
        ),
      ],
    );
  }
}

/// Toggle for [ChatStorageService.sweepRequiresConfirmation] — off by
/// default so the sweep can genuinely run in the background (essential
/// once the interval is set to a few seconds), on for anyone who'd rather
/// approve each cycle like the app originally required.
class _SweepConfirmationToggle extends StatefulWidget {
  final ChatStorageService storage;
  const _SweepConfirmationToggle({required this.storage});

  @override
  State<_SweepConfirmationToggle> createState() => _SweepConfirmationToggleState();
}

class _SweepConfirmationToggleState extends State<_SweepConfirmationToggle> {
  late bool _requireConfirmation;

  @override
  void initState() {
    super.initState();
    _requireConfirmation = widget.storage.sweepRequiresConfirmation;
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: Text(
            'Ask before each sweep',
            style: TextStyle(color: context.text, fontSize: 13),
          ),
        ),
        Switch(
          value: _requireConfirmation,
          activeTrackColor: AppColors.accent,
          onChanged: (v) {
            setState(() => _requireConfirmation = v);
            widget.storage.sweepRequiresConfirmation = v;
          },
        ),
      ],
    );
  }
}

/// Shared stepper for "every N sweep cycles" cadence settings — attention
/// reflection and Frame analysis are otherwise-identical steppers over two
/// different storage fields, so this takes plain get/set callbacks instead
/// of being written out twice.
class _CadenceRow extends StatefulWidget {
  final int Function() getValue;
  final void Function(int) setValue;
  const _CadenceRow({required this.getValue, required this.setValue});

  @override
  State<_CadenceRow> createState() => _CadenceRowState();
}

class _CadenceRowState extends State<_CadenceRow> {
  late int _everyN;

  @override
  void initState() {
    super.initState();
    _everyN = widget.getValue();
  }

  void _set(int value) {
    final clamped = value.clamp(1, 200);
    setState(() => _everyN = clamped);
    widget.setValue(clamped);
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: Text(
            'Every $_everyN sweep${_everyN == 1 ? '' : 's'}',
            style: TextStyle(color: context.text, fontSize: 13),
          ),
        ),
        IconButton(
          icon: Icon(Icons.remove_circle_outline, size: 20, color: context.textM),
          onPressed: () => _set(_everyN - 1),
        ),
        IconButton(
          icon: Icon(Icons.add_circle_outline, size: 20, color: context.textM),
          onPressed: () => _set(_everyN + 1),
        ),
      ],
    );
  }
}

class _HardwareSettingsCard extends StatefulWidget {
  final ChatStorageService storage;

  const _HardwareSettingsCard({required this.storage});

  @override
  State<_HardwareSettingsCard> createState() => _HardwareSettingsCardState();
}

class _HardwareSettingsCardState extends State<_HardwareSettingsCard> {
  late String _backend;
  late double _gpuLayers;
  bool _showManual = false;

  List<String>? _detectedBackends;
  bool _benchmarking = false;
  String? _benchmarkingCurrent;
  List<BackendBenchmarkResult>? _benchmarkResults;

  // Recommended backend for this device.
  //
  // This used to guess GPU (OpenCL, 33 layers) for any 8+ core Android
  // device on core-count alone, with no check that OpenCL actually works —
  // there's no public API to verify that before trying. That guess is
  // exactly what produced a "Recommended" config that ran at ~1 t/s on a
  // real device: the SoC had cores, but nothing confirmed the GPU backend
  // was doing real work rather than silently falling back or thrashing.
  //
  // CPU is now always the recommendation — it's the one backend that
  // reliably works everywhere. GPU backends are still available, but only
  // as an explicit, clearly-labeled opt-in under Manual Override below.
  static Map<String, dynamic> _detectBestConfig() {
    final cores = Platform.numberOfProcessors;
    return {
      'backend': 'cpu',
      'gpuLayers': 0,
      'reason': 'CPU mode — reliably works on every device ($cores cores '
          'detected). GPU backends (Vulkan/OpenCL) aren\'t verified to work '
          'on this specific device — try one manually below if you want to, '
          'but a slow or unstable result usually means it isn\'t.',
    };
  }

  @override
  void initState() {
    super.initState();
    _backend = widget.storage.backendType;
    _gpuLayers = widget.storage.gpuLayers.toDouble();
    _detectAvailableBackends();
  }

  /// Static registry check — what the native library actually has
  /// compiled in and registered on this device, independent of which one
  /// is currently configured. Cheap (no generation, no reload), so this
  /// runs automatically rather than waiting for the user to ask.
  Future<void> _detectAvailableBackends() async {
    final llm = Get.find<LlmService>();
    if (!llm.isLoaded.value) return;
    try {
      final raw = await llm.getAvailableBackends();
      if (!mounted) return;
      setState(() {
        _detectedBackends = raw
            .split(',')
            .map((s) => s.trim())
            .where((s) => s.isNotEmpty)
            .toList();
      });
    } catch (_) {
      // Leave null — the UI just won't show a detected-backends line.
    }
  }

  Future<void> _runBenchmark() async {
    final llm = Get.find<LlmService>();
    if (!llm.isLoaded.value) {
      Get.snackbar(
        'No Model Loaded',
        'Load a model first — the benchmark needs a real model to test '
            'each backend against.',
        snackPosition: SnackPosition.BOTTOM,
      );
      return;
    }

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: ctx.bgPanel,
        title: Text('Benchmark Backends?', style: TextStyle(color: ctx.text)),
        content: Text(
          'Reloads the current model once per backend and times a short '
          'real generation on each — CPU vs. GPU (Vulkan/OpenCL, whichever '
          'this device actually has). Takes roughly a minute and the model '
          'will be unavailable for chat while it runs. Your original '
          'backend setting is restored afterward either way.',
          style: TextStyle(color: ctx.textM, fontSize: 13),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text('Cancel', style: TextStyle(color: ctx.textD)),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: ElevatedButton.styleFrom(backgroundColor: AppColors.accent, elevation: 0),
            child: const Text('Run It', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;

    setState(() {
      _benchmarking = true;
      _benchmarkResults = null;
      _benchmarkingCurrent = null;
    });

    try {
      // Falls back to testing all three whenever the detected-backends
      // filter doesn't leave anything recognizable — not just when
      // detection itself failed (null), but also when it succeeded with
      // names this app doesn't recognize (e.g. a backend string other than
      // cpu/vulkan/opencl) and the filter zeroed the list out.
      final recognized = _detectedBackends
          ?.map((b) => b.toLowerCase())
          .where((b) => b.contains('cpu') || b.contains('vulkan') || b.contains('opencl'))
          .map((b) {
            if (b.contains('vulkan')) return 'vulkan';
            if (b.contains('opencl')) return 'opencl';
            return 'cpu';
          })
          .toSet()
          .toList();
      final results = await llm.benchmarkBackends(
        backends:
            (recognized == null || recognized.isEmpty) ? const ['cpu', 'vulkan', 'opencl'] : recognized,
        onProgress: (backend) {
          if (mounted) setState(() => _benchmarkingCurrent = backend);
        },
      );
      if (!mounted) return;
      setState(() {
        _benchmarking = false;
        _benchmarkResults = results;
        // Re-sync with whatever benchmarkBackends restored, in case it
        // differs from what this widget had cached.
        _backend = widget.storage.backendType;
        _gpuLayers = widget.storage.gpuLayers.toDouble();
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _benchmarking = false);
      Get.snackbar('Benchmark Failed', '$e', snackPosition: SnackPosition.BOTTOM);
    }
  }

  void _applyBenchmarkWinner(BackendBenchmarkResult winner) {
    setState(() {
      _backend = winner.backend;
      _gpuLayers = winner.backend == 'cpu' ? 0 : 33;
    });
    widget.storage.backendType = _backend;
    widget.storage.gpuLayers = _gpuLayers.toInt();
    Get.snackbar(
      'Applied',
      '${_backendLabel(winner.backend)} set as active — reload the model '
          'for it to take effect on your next chat.',
      snackPosition: SnackPosition.BOTTOM,
    );
  }

  String _backendLabel(String backend) {
    switch (backend) {
      case 'vulkan':
        return 'GPU (Vulkan)';
      case 'opencl':
        return 'GPU (OpenCL)';
      default:
        return 'CPU';
    }
  }

  void _applyAutoConfig() {
    final config = _detectBestConfig();
    setState(() {
      _backend = config['backend'] as String;
      _gpuLayers = (config['gpuLayers'] as int).toDouble();
    });
    widget.storage.backendType = _backend;
    widget.storage.gpuLayers = _gpuLayers.toInt();
    Get.snackbar(
      'Auto Config Applied',
      config['reason'] as String,
      snackPosition: SnackPosition.BOTTOM,
      duration: const Duration(seconds: 2),
    );
  }

  void _saveBackend(String val) {
    setState(() => _backend = val);
    widget.storage.backendType = val;
    // Auto-set sensible GPU layers when switching
    if (val == 'cpu') {
      setState(() => _gpuLayers = 0);
      widget.storage.gpuLayers = 0;
    } else if (_gpuLayers == 0) {
      setState(() => _gpuLayers = 33);
      widget.storage.gpuLayers = 33;
    }
  }

  void _saveGpuLayers(double val) {
    setState(() => _gpuLayers = val);
    widget.storage.gpuLayers = val.toInt();
  }

  String get _currentConfigLabel {
    switch (_backend) {
      case 'vulkan':
        return 'GPU (Vulkan) • ${_gpuLayers.toInt()} layers';
      case 'opencl':
        return 'GPU (OpenCL) • ${_gpuLayers.toInt()} layers';
      default:
        return 'CPU Only';
    }
  }

  @override
  Widget build(BuildContext context) {
    final autoConfig = _detectBestConfig();

    return Container(
      decoration: BoxDecoration(
        color: context.bgPanel,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: context.border),
      ),
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Recommended Auto Config ──
          Row(
            children: [
              Icon(Icons.auto_awesome_rounded, size: 18, color: AppColors.accent),
              const SizedBox(width: 8),
              Text(
                'Compute Device',
                style: TextStyle(color: context.text, fontSize: 15, fontWeight: FontWeight.w600),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            'Current: $_currentConfigLabel',
            style: TextStyle(color: context.textM, fontSize: 12),
          ),
          const SizedBox(height: 12),

          // Recommended button
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: _applyAutoConfig,
              icon: const Icon(Icons.tune_rounded, size: 16),
              label: const Text('Apply Recommended Settings'),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.accent,
                foregroundColor: Colors.white,
                elevation: 0,
                padding: const EdgeInsets.symmetric(vertical: 12),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
              ),
            ),
          ),
          const SizedBox(height: 6),
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: AppColors.accent.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              children: [
                Icon(Icons.info_outline_rounded, size: 14, color: AppColors.accent),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    autoConfig['reason'] as String,
                    style: TextStyle(color: context.textM, fontSize: 11),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 16),

          // ── Real backend benchmark ──
          if (_detectedBackends != null) ...[
            Text(
              'Detected on this device: ${_detectedBackends!.join(", ")}',
              style: TextStyle(color: context.textD, fontSize: 11),
            ),
            const SizedBox(height: 8),
          ],
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: _benchmarking ? null : _runBenchmark,
              icon: _benchmarking
                  ? SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2, color: context.textM),
                    )
                  : const Icon(Icons.speed_rounded, size: 16),
              label: Text(
                _benchmarking
                    ? 'Testing ${_backendLabel(_benchmarkingCurrent ?? "")}...'
                    : 'Benchmark Backends On This Device',
              ),
              style: OutlinedButton.styleFrom(
                foregroundColor: context.text,
                side: BorderSide(color: context.border),
                padding: const EdgeInsets.symmetric(vertical: 12),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
              ),
            ),
          ),
          if (_benchmarkResults != null) ...[
            const SizedBox(height: 10),
            _BenchmarkResultsList(
              results: _benchmarkResults!,
              onApply: _applyBenchmarkWinner,
              backendLabel: _backendLabel,
            ),
          ],

          const SizedBox(height: 16),

          // ── Manual Override Toggle ──
          InkWell(
            onTap: () => setState(() => _showManual = !_showManual),
            borderRadius: BorderRadius.circular(8),
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: Row(
                children: [
                  Icon(
                    _showManual ? Icons.expand_less : Icons.expand_more,
                    size: 18,
                    color: context.textM,
                  ),
                  const SizedBox(width: 4),
                  Text(
                    'Manual Override',
                    style: TextStyle(
                      color: context.textM,
                      fontSize: 13,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
          ),

          if (_showManual) ...[
            const SizedBox(height: 12),
            Row(
              children: [
                _buildBackendButton('CPU', 'cpu'),
                const SizedBox(width: 8),
                _buildBackendButton('Vulkan', 'vulkan'),
                const SizedBox(width: 8),
                _buildBackendButton('OpenCL', 'opencl'),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'GPU Layers',
                  style: TextStyle(color: context.text, fontSize: 14),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: context.bgInput,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    _gpuLayers.toInt().toString(),
                    style: TextStyle(color: context.text, fontWeight: FontWeight.bold),
                  ),
                ),
              ],
            ),
            SliderTheme(
              data: SliderTheme.of(context).copyWith(
                activeTrackColor: AppColors.accent,
                inactiveTrackColor: context.border,
                thumbColor: AppColors.accent,
                overlayColor: AppColors.accent.withValues(alpha: 0.2),
              ),
              child: Slider(
                value: _gpuLayers,
                min: 0,
                max: 99,
                divisions: 99,
                onChanged: _backend == 'cpu' ? null : _saveGpuLayers,
              ),
            ),
            Text(
              'If the app crashes when loading a model, reduce GPU layers or switch to CPU. Reload the model after changing settings.',
              style: TextStyle(color: context.textD, fontSize: 11, height: 1.4),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildBackendButton(String label, String value) {
    final selected = _backend == value;
    return Expanded(
      child: InkWell(
        onTap: () => _saveBackend(value),
        borderRadius: BorderRadius.circular(8),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: selected ? AppColors.accent : context.bgInput,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: selected ? AppColors.accent : context.border,
            ),
          ),
          child: Center(
            child: Text(
              label,
              style: TextStyle(
                color: selected ? Colors.white : context.text,
                fontSize: 12,
                fontWeight: selected ? FontWeight.bold : FontWeight.normal,
              ),
              textAlign: TextAlign.center,
            ),
          ),
        ),
      ),
    );
  }

}

