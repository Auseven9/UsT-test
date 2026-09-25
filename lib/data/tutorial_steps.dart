import '../models/tutorial_step.dart';

/// The full guided tour, in order. Each step's `targetId` must match a
/// `TutorialTarget` id somewhere in the widget tree (home_screen.dart for
/// tab 0, settings_screen.dart for tab 2) or the overlay falls back to a
/// plain centered card with no spotlight for that step.
List<TutorialStep> buildTutorialSteps() => const [
      TutorialStep(
        id: 'chat.history',
        targetId: 'chat.history',
        title: 'Chat History',
        description:
            'Opens the list of your past conversations. Each one is stored '
            'on-device only — nothing here ever leaves your phone.',
        requiredTab: 0,
      ),
      TutorialStep(
        id: 'chat.model_selector',
        targetId: 'chat.model_selector',
        title: 'Model Selector',
        description:
            'Shows which model is currently loaded — tap it to switch. The '
            'colored dot means loaded (green), loading (orange), or none '
            'loaded (red). Nothing generates until a model is loaded here.',
        requiredTab: 0,
      ),
      TutorialStep(
        id: 'chat.new_chat',
        targetId: 'chat.new_chat',
        title: 'New Chat',
        description:
            'Starts a fresh conversation. The current one stays saved in '
            'Chat History — nothing is discarded.',
        requiredTab: 0,
      ),
      TutorialStep(
        id: 'chat.input',
        targetId: 'chat.input',
        title: 'Message Box',
        description:
            'What you type here is sent to the model exactly as written, '
            'along with recent conversation history and anything relevant '
            'pulled from memory. This is the main way anything reaches the '
            'model.',
        requiredTab: 0,
        injectsIntoModel: true,
      ),
      TutorialStep(
        id: 'settings.system_prompt',
        targetId: 'settings.system_prompt',
        title: 'Global System Prompt',
        description:
            'Standing instructions sent before every message in every '
            'chat — persona, tone, rules. A single chat can override this '
            'with its own system prompt, but this is the default for '
            'anything that doesn\'t.',
        requiredTab: 2,
        injectsIntoModel: true,
      ),
      TutorialStep(
        id: 'settings.temperature',
        targetId: 'settings.temperature',
        title: 'Temperature',
        description:
            'How much randomness the model uses when picking its next '
            'word. Low = more predictable and repetitive. High = more '
            'varied, but more likely to wander or make things up.',
        requiredTab: 2,
      ),
      TutorialStep(
        id: 'settings.context_size',
        targetId: 'settings.context_size',
        title: 'Context Size',
        description:
            'How much conversation the model can see at once, in tokens. '
            'Higher means longer memory of the current chat, but more RAM '
            'and slower processing — reload the model after changing it.',
        requiredTab: 2,
      ),
      TutorialStep(
        id: 'settings.model_reasoning',
        targetId: 'settings.model_reasoning',
        title: 'Model Reasoning',
        description:
            'Lets reasoning-capable models show their step-by-step '
            'thinking in a collapsible panel before the final answer. Off '
            'can mean faster, shorter replies on models that support '
            'toggling it.',
        requiredTab: 2,
      ),
      TutorialStep(
        id: 'settings.tool_calling',
        targetId: 'settings.tool_calling',
        title: 'Tool Calling',
        description:
            'Gives the model access to a few offline tools — the current '
            'date/time, a calculator, and memory search. With Persistent '
            'Memory also on, this includes memory write tools — the model '
            'can save, edit, or supersede memories on its own. The tool '
            'descriptions themselves are sent to the model on every turn, '
            'even when unused.',
        requiredTab: 2,
        injectsIntoModel: true,
      ),
      TutorialStep(
        id: 'settings.advanced_tools',
        targetId: 'settings.advanced_tools',
        title: 'Advanced Tools',
        description:
            'Adds clipboard read/write, in-app reminders, and letting the '
            'model recall its own past reasoning. All on-device, no '
            'network. Requires Tool Calling above to be on.',
        requiredTab: 2,
        injectsIntoModel: true,
      ),
      TutorialStep(
        id: 'settings.self_critique',
        targetId: 'settings.self_critique',
        title: 'Self-Critique',
        description:
            'After each reply, a short background check looks for '
            'contradictions with memory or unsupported confident claims. '
            'Never edits or blocks the reply — just a soft note if it '
            'finds something. Costs one extra background generation per '
            'turn.',
        requiredTab: 2,
      ),
      TutorialStep(
        id: 'settings.reasoning_trace',
        targetId: 'settings.reasoning_trace',
        title: 'Reasoning Trace',
        description:
            'Distills each turn\'s chain-of-thought into a short gist kept '
            'in its own lane, separate from fact memory — lets the model '
            'recall how it approached something before, not just what it '
            'concluded.',
        requiredTab: 2,
      ),
      TutorialStep(
        id: 'settings.speculative_decoding',
        targetId: 'settings.speculative_decoding',
        title: 'Speculative Decoding',
        description:
            'A generation-speed trick — drafts from tokens already in the '
            'conversation and verifies them in one batch. Never changes '
            'what gets generated, only how fast. Off by default until '
            'you\'ve measured it on your device.',
        requiredTab: 2,
      ),
      TutorialStep(
        id: 'settings.hardware',
        targetId: 'settings.hardware',
        title: 'Hardware Configuration',
        description:
            'Chooses CPU vs. GPU (Vulkan/OpenCL) for running the model. '
            'Use "Benchmark Backends" to actually measure which is faster '
            'on your device instead of guessing — GPU is not always the '
            'win it sounds like on phones.',
        requiredTab: 2,
      ),
      TutorialStep(
        id: 'settings.persistent_memory',
        targetId: 'settings.persistent_memory',
        title: 'Persistent Memory',
        description:
            'Distills conversations into short notes that can surface '
            'again in any future chat — needs an embedding model selected '
            'below. Relevant memories are quietly added to what the model '
            'sees when they match the current conversation.',
        requiredTab: 2,
        injectsIntoModel: true,
      ),
      TutorialStep(
        id: 'settings.memory_verification',
        targetId: 'settings.memory_verification',
        title: 'Memory Verification',
        description:
            'Runs 5 fixed test exchanges through the real memory-'
            'extraction pipeline and checks the results against known-'
            'correct answers — a way to confirm memory capture is actually '
            'working before trusting it.',
        requiredTab: 2,
      ),
      TutorialStep(
        id: 'settings.reminders',
        targetId: 'settings.reminders',
        title: 'Reminders',
        description:
            'Set by the model via its set_reminder tool. Surfaced in-app '
            'when due — not a phone notification, only while the app is '
            'open.',
        requiredTab: 2,
      ),
      TutorialStep(
        id: 'settings.local_api_server',
        targetId: 'settings.local_api_server',
        title: 'Local API Server',
        description:
            'Exposes the loaded model to other apps on your device or '
            'network through an OpenAI-compatible local endpoint — for '
            'using this model outside this app.',
        requiredTab: 2,
      ),
    ];
