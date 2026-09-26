import 'package:get/get.dart';

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
import '../services/tutorial_service.dart';
import '../services/crash_log_service.dart';
import '../services/resource_monitor_service.dart';
import '../controllers/chat_controller.dart';
import '../controllers/model_controller.dart';
import '../controllers/theme_controller.dart';

/// Initial bindings — registers all services and controllers with GetX DI.
class AppBindings extends Bindings {
  @override
  void dependencies() {
    // ── Services (async init happens in splash) ──────────────────
    Get.lazyPut(() => LlmService(), fenix: true);
    Get.lazyPut(() => ModelManager(), fenix: true);
    Get.lazyPut(() => ChatStorageService(), fenix: true);
    Get.lazyPut(() => LocalApiServerService(), fenix: true);
    Get.lazyPut(() => WakelockService(), fenix: true);
    Get.lazyPut(() => LogService(), fenix: true);
    Get.lazyPut(() => EmbeddingService(), fenix: true);
    Get.lazyPut(() => HelperLlmService(), fenix: true);
    // A second, independently-tagged instance of the exact same service —
    // not the always-loaded memory-extraction helper above, but a model
    // that stays unloaded until Frame analysis's rotation gives it a turn
    // as the "second opinion" (see ChatController._runFrameAnalysis and
    // ChatStorageService.secondOpinionModelFilename), then gets torn down
    // again right after. Same class, same load/unload discipline, just a
    // separate engine so loading this one never disturbs the helper's own
    // loaded model.
    Get.lazyPut(() => HelperLlmService(), tag: 'secondOpinion', fenix: true);
    Get.lazyPut(() => MemoryService(), fenix: true);
    Get.lazyPut(() => ReasoningTraceService(), fenix: true);
    Get.lazyPut(() => ReminderService(), fenix: true);
    // Constructed lazily, only once HomeScreen actually needs it — by then
    // ChatStorageService (which it reads/writes the completed flag
    // through) is already fully initialized by splash_screen.dart.
    Get.lazyPut(() => TutorialService(), fenix: true);
    // Already constructed and recording in main() before bindings run —
    // register that exact instance rather than a fresh one.
    Get.put(CrashLogService.instance, permanent: true);
    Get.lazyPut(() => ResourceMonitorService(), fenix: true);

    // ── Controllers ──────────────────────────────────────────────
    Get.put(
      ThemeController(),
    ); // Put instead of lazyPut since we need theme immediately
    Get.lazyPut(() => ChatController(), fenix: true);
    Get.lazyPut(() => ModelController(), fenix: true);
  }
}
