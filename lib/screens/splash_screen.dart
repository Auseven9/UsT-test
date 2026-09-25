import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:flutter_animate/flutter_animate.dart';

import '../theme/app_colors.dart';
import '../services/llm_service.dart';
import '../services/model_manager.dart';
import '../services/chat_storage_service.dart';
import '../services/local_api_server_service.dart';
import '../services/wakelock_service.dart';
import '../services/log_service.dart';
import '../services/embedding_service.dart';
import '../services/helper_llm_service.dart';
import '../services/memory_service.dart';
import '../services/reasoning_trace_service.dart';
import '../services/reminder_service.dart';
import '../services/background_optimizer_service.dart';
import '../routes/app_routes.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  String _status = 'Initializing...';

  @override
  void initState() {
    super.initState();
    _initApp();
  }

  Future<void> _initApp() async {
    try {
      // Initialize logging first
      final log = Get.find<LogService>()..init();

      setState(() => _status = 'Setting up storage...');
      log.info('Initializing storage...', source: 'Splash');
      await Get.find<ChatStorageService>().init();

      setState(() => _status = 'Loading model catalog...');
      log.info('Loading model catalog...', source: 'Splash');
      final modelManager = await Get.find<ModelManager>().init();

      setState(() => _status = 'Preparing AI engine...');
      log.info('Preparing AI engine...', source: 'Splash');
      final llm = await Get.find<LlmService>().init();

      // Must run before any model load below: loadModel() can call
      // WakelockService.enableForInference(), which starts the Android
      // foreground-service notification — that requires the notification
      // channel WakelockService.init() registers. Restoring the last chat
      // model at splash time, before this ran, meant the very first (and
      // heaviest/longest) load of the app's lifecycle silently lost its
      // OS-kill protection with no visible error anywhere.
      setState(() => _status = 'Setting up background services...');
      log.info('Setting up background services...', source: 'Splash');
      await Get.find<WakelockService>().init();

      setState(() => _status = 'Setting up memory...');
      log.info('Setting up persistent memory...', source: 'Splash');
      final memory = await Get.find<MemoryService>().init();
      final embedding = await Get.find<EmbeddingService>().init();
      final helper = await Get.find<HelperLlmService>().init();
      await Get.find<ReasoningTraceService>().init();
      await Get.find<ReminderService>().init();
      log.info('${memory.entries.length} memories loaded', source: 'Splash');

      // Restore the last-loaded chat model across restarts. Without this the
      // header silently reverts to "No model selected" after every relaunch
      // even though the app still remembers which model was selected —
      // matching the restoration already done below for the embedding and
      // helper models, so all three behave the same way across a restart.
      final storage = Get.find<ChatStorageService>();
      if (storage.lastModelId.isNotEmpty) {
        setState(() => _status = 'Reloading last model...');
        try {
          final path = modelManager.getModelPathByFilename(storage.lastModelId);
          await llm.loadModel(path);
        } catch (e) {
          log.error('Could not restore chat model: $e', source: 'Splash');
        }
      }

      // Restore the embedding model across restarts, same as the local API
      // server restores its own enabled state — otherwise persistent memory
      // would silently stop retrieving anything after every app relaunch.
      if (storage.persistentMemoryEnabled &&
          storage.memoryEmbeddingModelFilename.isNotEmpty) {
        try {
          final path = modelManager.getModelPathByFilename(
            storage.memoryEmbeddingModelFilename,
          );
          await embedding.loadModel(path);
        } catch (e) {
          log.error('Could not restore embedding model: $e', source: 'Splash');
        }
      }

      // Same restoration for the optional background-extraction helper
      // model — otherwise it would silently fall back to the (slow) main
      // model after every relaunch.
      if (storage.helperModelFilename.isNotEmpty) {
        try {
          final path = modelManager.getModelPathByFilename(
            storage.helperModelFilename,
          );
          await helper.loadModel(path);
        } catch (e) {
          log.error('Could not restore helper model: $e', source: 'Splash');
        }
      }

      setState(() => _status = 'Preparing local API...');
      log.info('Preparing local API...', source: 'Splash');
      await Get.find<LocalApiServerService>().init();

      setState(() => _status = 'Ready!');
      log.info('All services initialized successfully', source: 'Splash');
      await Future.delayed(const Duration(milliseconds: 500));

      // Prompt for battery optimization on Android (first launch only)
      if (mounted) {
        await BackgroundOptimizerService.checkAndPrompt(context);
      }

      Get.offAllNamed(AppRoutes.home);
    } catch (e) {
      setState(() => _status = 'Error: $e');
      try {
        Get.find<LogService>().error('Init failed: $e', source: 'Splash');
      } catch (_) {}
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: context.bg,
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Logo
            Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    gradient: AppColors.accentGradient,
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: [
                      BoxShadow(
                        color: AppColors.accent.withValues(alpha: 0.3),
                        blurRadius: 30,
                        spreadRadius: 5,
                      ),
                    ],
                  ),
                  child: const Icon(
                    Icons.bolt_rounded,
                    size: 42,
                    color: Colors.white,
                  ),
                )
                .animate()
                .fadeIn(duration: 600.ms)
                .scale(begin: const Offset(0.8, 0.8)),
            const SizedBox(height: 24),
            Text(
              'Uncensored Local AI',
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.w700,
                color: context.text,
                letterSpacing: -0.5,
              ),
            ).animate().fadeIn(delay: 200.ms, duration: 600.ms),
            const SizedBox(height: 8),
            Text(
              'Run uncensored LLMs natively on any device 🔓',
              style: TextStyle(fontSize: 13, color: context.textM),
            ).animate().fadeIn(delay: 400.ms, duration: 600.ms),
            const SizedBox(height: 40),
            const SizedBox(
              width: 24,
              height: 24,
              child: CircularProgressIndicator(
                strokeWidth: 2.5,
                valueColor: AlwaysStoppedAnimation(AppColors.accent),
              ),
            ).animate().fadeIn(delay: 600.ms),
            const SizedBox(height: 16),
            Text(
              _status,
              style: TextStyle(fontSize: 12, color: context.textD),
            ).animate().fadeIn(delay: 600.ms),
          ],
        ),
      ),
    );
  }
}
