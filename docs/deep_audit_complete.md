# WiltonOS PassiveWorks Complete Audit

*Generated: 2025-12-22T19:28:02.478142*

*Source: /home/zews/rag-local/WiltonOS-PassiveWorks*


## Summary Statistics

| Metric | Count |
|--------|-------|
| Files Scanned | 6868 |
| Python Modules | 327 |
| TypeScript Modules | 1762 |
| JSON Files | 3212 |
| Markdown Files | 1567 |
| Classes Found | 1109 |
| Functions Found | 720 |
| Interfaces Found | 1816 |
| High-Priority Port Candidates | 1053 |

## Files by Category


### AGENT (70 files)

- **quantum-agent-manager.ts** (1488 lines, priority: medium)
  - Classes: for, QuantumAgentManager
- **MultiAgentSynchronizationTest.tsx** (1401 lines, priority: high)
  - Functions: for
- **quantum-agent-manager.js** (1292 lines, priority: medium)
  - Classes: for
  - Functions: adopt, fulfilled, rejected, step, verb
- **AgentStatusPanel.tsx** (973 lines, priority: high)
  - Functions: AgentStatusPanel
- **model-agents.js** (964 lines, priority: low)
  - Classes: with, provides
  - Functions: __, adopt, fulfilled, rejected, step
- **multi-agent-sync-handlers.ts** (934 lines, priority: high)
  - Classes: MultiAgentSyncManager
  - Functions: registerMultiAgentSyncHandlers, in, setupMultiAgentSyncHandlers
- **model-agents.ts** (880 lines, priority: low)
  - Classes: with, provides, BaseAgent, GPT4oAgent, GeminiProAgent
  - Functions: calling, calling, calling, to, createModelAgent
- **quantum-coordinator.js** (844 lines, priority: high)
  - Functions: formatAgentLog, coordinateChunk, processSequentially, processInParallel, processInFractalPattern
- **true-index-analyst.js** (747 lines, priority: high)
  - Functions: formatAgentLog, analyzePatterns, analyzeTrends, calculateTrendConfidence, analyzeClusters
- **oracle-agent.ts** (727 lines, priority: high)
  - Classes: OracleAgent

### BREATHING (19 files)

- **breath-tracker.js** (641 lines, priority: high)
  - Functions: startBreathPattern, startPhaseTimer, advanceToNextPhase, checkCompletionStatus, emitPhaseChange
- **breathing_resonance.py** (625 lines, priority: high)
  - Classes: BreathingResonance
- **breath-interface.ts** (447 lines, priority: high)
  - Classes: BreathInterface
- **breath-interface.ts** (447 lines, priority: high)
  - Classes: BreathInterface
- **MirrorBreathsStudio.tsx** (349 lines, priority: high)
  - Functions: MirrorBreathsStudio
- **breath-interface.js** (348 lines, priority: high)
  - Classes: BreathInterface
- **breath-feedback.js** (337 lines, priority: medium)
- **haloBreath.ts** (328 lines, priority: high)
  - Classes: HaloBreathController
- **temporal-breathlock.ts** (146 lines, priority: high)
  - Classes: TemporalBreathlock
- **temporal-breathlock.ts** (146 lines, priority: high)
  - Classes: TemporalBreathlock

### BRIDGE (103 files)

- **dashboard.js** (2616 lines, priority: high)
  - Classes: WiltonOSDashboard
- **meta_stream.js** (1921 lines, priority: high)
  - Classes: MetaStreamIntegration
- **multimodal.js** (1689 lines, priority: high)
  - Functions: initializeSystem, connectUIElements, setupEventListeners, initSpeechRecognition, handleSpeechResult
- **soundwave_timeline.js** (1505 lines, priority: high)
  - Classes: SoundwaveTimeline
- **bridge.js** (1099 lines, priority: high)
  - Functions: startBridgeServer, handleNewConnection, handleClientMessage, handleAuthResponse, handleRitualExecution
- **wilton-bridge-client.js** (1009 lines, priority: high)
  - Functions: connect, startHeartbeat, handleServerMessage, handleAuthRequest, handleAuthSuccess
- **wilton-bridge-client.js** (1009 lines, priority: high)
  - Functions: connect, startHeartbeat, handleServerMessage, handleAuthRequest, handleAuthSuccess
- **emulation-bridge.js** (982 lines, priority: high)
  - Functions: startEmulation, ensureDirectories, loadRitualsList, loadScenesList, loadQuantumState
- **emulation-bridge.js** (982 lines, priority: high)
  - Functions: startEmulation, ensureDirectories, loadRitualsList, loadScenesList, loadQuantumState
- **file_bridge.py** (958 lines, priority: high)
  - Classes: FileProcessor, FileBridge
  - Functions: main

### COHERENCE (251 files)

- **MultiDimensionalCoherenceVisualizer.tsx** (1063 lines, priority: high)
- **quantum-coherence-dashboard.ts** (1014 lines, priority: high)
  - Classes: QuantumCoherenceDashboard
- **CoherenceAttractorDemo.tsx** (965 lines, priority: high)
  - Functions: CoherenceAttractorDemo, for, for
- **QuantumCoherenceDashboard.tsx** (868 lines, priority: high)
  - Functions: from
- **s-coherence-agent.js** (796 lines, priority: high)
  - Classes: SCoherenceAgent
- **stress_test_coherence.py** (785 lines, priority: high)
  - Classes: CoherenceStressTest
  - Functions: main, deep_update
- **generate_coherence_visualizations.py** (758 lines, priority: high)
  - Functions: parse_json_results, convert_timestamps, plot_coherence_over_time, plot_weight_adjustments, plot_original_weights
- **coherence-attractor-experiment.ts** (737 lines, priority: high)
  - Classes: class
  - Functions: to, runExperiment
- **integration-coherence-tracker.ts** (724 lines, priority: medium)
  - Functions: recordIntegrationOperation, calculateIntegrationCoherence, calculateCoherenceTrend, calculateLayerSpecificCoherence, calculateDomainCoverage
- **adjust_coherence_thresholds.py** (705 lines, priority: high)
  - Classes: ThresholdTuner
  - Functions: main

### DAEMON (2 files)

- **llm_insight_daemon.py** (148 lines, priority: high)
  - Classes: LLMInsightDaemon
  - Functions: main
- **__init__.py** (9 lines, priority: medium)

### EXAMPLE (26 files)

- **enhanced_interface_integration.py** (1037 lines, priority: high)
- **interface_integration_demo.py** (487 lines, priority: high)
- **phi_monitor.py** (358 lines, priority: high)
  - Functions: format_phi_bar, clear_screen, format_ratio_bar, render_phi_display, monitor_phi
- **integration-example.ts** (330 lines, priority: high)
  - Classes: demonstrates, AgentStateManager, demonstrates, MultiAgentCoherenceOrchestrator
  - Functions: demonstrateOrchestrator
- **hpc_ws_demo.py** (311 lines, priority: high)
  - Functions: display_phi_history, demo_coherence_monitoring, demo_schedule_tasks, run_demo, main
- **symbolic-logging-example.js** (251 lines, priority: medium)
  - Classes: SymbolicLogger
  - Functions: demonstrateSymbolicLogger
- **symbolic-communication-example.js** (202 lines, priority: medium)
  - Functions: simulateSystemEvent, logSystemEvent
- **symbolic-integration-example.js** (174 lines, priority: medium)
  - Functions: demonstrateFormatting, demonstrateParsing, demonstrateStateConversion, demonstrateFullCycle, runSymbolicIntegrationDemo
- **example_auth_client.py** (170 lines, priority: medium)
  - Functions: run_demo
- **import_wilton_realizations.py** (170 lines, priority: high)

### IDENTITY (9 files)

- **ZEWS_IDENTITY.md** (0 lines, priority: low)
- **profile.json** (0 lines, priority: low)
- **microsoft-identity-association.json** (0 lines, priority: low)
- **ZEWS_IDENTITY.md** (0 lines, priority: low)
- **identity.md** (0 lines, priority: low)
- **Q23HQMVP.json** (0 lines, priority: low)
- **QuantumTumbler.json** (0 lines, priority: low)
- **cluckthesystem.json** (0 lines, priority: low)
- **Robert_E_Grant_.json** (0 lines, priority: low)

### MARKET (12 files)

- **s-finance.js** (1445 lines, priority: high)
  - Classes: SFinanceAgent
- **short_term_sentiment.py** (1117 lines, priority: high)
  - Classes: SentimentEngine
  - Functions: get_sentiment_engine
- **entropy_tracker.py** (1102 lines, priority: high)
  - Classes: EntropyTracker
  - Functions: get_entropy_tracker
- **investment_engine.py** (896 lines, priority: high)
  - Classes: InvestmentEngine
  - Functions: get_investment_engine
- **ledger_montebravo.py** (896 lines, priority: high)
  - Classes: LedgerMonteBravo
  - Functions: get_ledger_montebravo
- **long_term_conviction.py** (662 lines, priority: high)
  - Classes: ConvictionEngine
  - Functions: get_conviction_engine
- **__init__.py** (8 lines, priority: medium)
- **QOCF_GO_TO_MARKET_STRATEGY.md** (0 lines, priority: low)
- **finance-presets.json** (0 lines, priority: low)
- **QOCF_GO_TO_MARKET_STRATEGY.md** (0 lines, priority: low)

### MEMORY (93 files)

- **memory-field-mapper.js** (1491 lines, priority: high)
  - Classes: MemoryFieldMapper
  - Functions: generateUniqueId, formatDate, initMemoryFieldUI, createControlPanel, setupControlHandlers
- **memory-perpetua.js** (1398 lines, priority: high)
  - Functions: initializeMemory, ensureDirectories, startTimers, loadFromFile, saveToFile
- **food_logging.py** (1177 lines, priority: high)
  - Classes: FoodLogger
  - Functions: get_food_logger_instance
- **memory-routes.js** (814 lines, priority: high)
  - Functions: for
- **health_hooks_connector.py** (807 lines, priority: high)
  - Classes: HealthHooksConnector
  - Functions: get_health_hooks_instance
- **auto_poster.py** (773 lines, priority: high)
  - Classes: AutoPoster
  - Functions: get_auto_poster_instance
- **qdrant_client.py** (729 lines, priority: high)
  - Classes: QdrantMemory
  - Functions: get_memory_instance, get_qdrant_client
- **memory-storage.js** (650 lines, priority: high)
  - Classes: MemoryStorage
- **memory-api.ts** (643 lines, priority: high)
  - Functions: getMemoryInsights, processMemoriesIntoInsights, determineDomain, extractKeyInsights, loadStaticInsights
- **quantum-chunking-memory-test.js** (607 lines, priority: medium)
  - Classes: InMemoryPersistenceLayer
  - Functions: testCreateChunk, testActivateChunkSynapse, testRouteChunk, testDecohereChunkState, testEntangleChunks

### OTHER (5836 files)

- **storage.js** (4723 lines, priority: high)
  - Classes: acts, provides, serves
  - Functions: adopt, fulfilled, rejected, step, verb
- **index.ts** (4691 lines, priority: high)
  - Functions: initializeMemorySystem, startServer
- **index.ts** (4691 lines, priority: high)
  - Functions: initializeMemorySystem, startServer
- **file-system-storage.js** (4154 lines, priority: high)
  - Functions: adopt, fulfilled, rejected, step, verb
- **file-system-storage.ts** (3321 lines, priority: high)
  - Classes: FileSystemStorage
  - Functions: for, for
- **storage.ts** (3272 lines, priority: high)
  - Classes: acts, FileSystemStorage, provides, serves, MemStorage
  - Functions: from
- **meta-void-analyzer.ts** (2093 lines, priority: high)
  - Functions: generateMetaVoidPreview, generateMetaVoidReview, calculateQuantumReadiness, generateChaosFactors, getDomainSpecificChaosFactors
- **resonance-calculator.ts** (2059 lines, priority: high)
  - Functions: calculateResonanceFactor, getContentLength, calculateStructuralMatch, initializeGlobalVocabulary, tokenize
- **routes.ts** (2012 lines, priority: high)
  - Functions: generateSacredGeometry, generateSVGPattern, calculateTruthFrequency, registerRoutes
- **meta-synthesis-modular-formula.ts** (1924 lines, priority: high)
  - Functions: analyzeMacroContext, analyzeMicroDetails, identifyRootCauses, analyzeVoid, applyMSMF

### RITUAL (29 files)

- **ritual_engine.py** (1826 lines, priority: high)
  - Classes: RitualType, RitualElement, RitualTrigger, Ritual, RitualLog
  - Functions: render_interface
- **FRACTAL_DECAY_RITUAL.py** (841 lines, priority: high)
  - Classes: FractalDecayRitual
  - Functions: show_interface
- **glyph_ritual.js** (808 lines, priority: high)
  - Classes: GlyphRitual
- **quantum-playlist-engine.ts** (741 lines, priority: high)
  - Classes: QuantumPlaylistEngine
- **quantum-playlist-engine.ts** (741 lines, priority: high)
  - Classes: QuantumPlaylistEngine
- **quantum-playlist-engine.js** (652 lines, priority: high)
  - Classes: QuantumPlaylistEngine
- **ritual-executor.js** (507 lines, priority: high)
  - Functions: executeRitual, executeCommand, extractMetadata, extractCommands, getQuantumState
- **ritual-executor.js** (507 lines, priority: high)
  - Functions: executeRitual, executeCommand, extractMetadata, extractCommands, getQuantumState
- **ritual-logger.js** (302 lines, priority: high)
  - Functions: logRitualExecution, logSceneExecution, logCommandExecution, logBridgeEvent, updateSystemIntegrity
- **ritual-logger.js** (246 lines, priority: high)
  - Functions: logRitual, getRitualHistory, getRitualStatistics, clearRitualHistory, ensureDirectory

### ROUTING (63 files)

- **agent_router.js** (1886 lines, priority: high)
  - Classes: AgentRouter
- **meta_router.py** (753 lines, priority: high)
  - Functions: init_router, route
- **model_router.py** (604 lines, priority: high)
  - Classes: ModelRouter
  - Functions: create_default_config
- **meta-routing-api.js** (511 lines, priority: high)
  - Classes: instance
  - Functions: const
- **meta-routing-client.js** (418 lines, priority: medium)
  - Classes: MetaRoutingClient
- **module-router.ts** (415 lines, priority: high)
  - Classes: UnifiedModuleRouter
  - Functions: if
- **module-router.ts** (415 lines, priority: high)
  - Classes: UnifiedModuleRouter
  - Functions: if
- **wilton-merge-router.ts** (406 lines, priority: high)
  - Functions: ensureDirectories, processMergeInBackground, appendToLog, formatBytes
- **router.py** (385 lines, priority: high)
  - Classes: RouterLLM
  - Functions: create_router
- **UnifiedModuleRouter.tsx** (362 lines, priority: high)

### SACRED_GEOMETRY (90 files)

- **z-geometry-engine.js** (858 lines, priority: high)
  - Classes: ZGeometryEngine
- **geometry_generator.py** (727 lines, priority: high)
  - Functions: golden_ratio, hsv_to_rgb, apply_quantum_noise, flower_of_life, metatrons_cube
- **geometry_generator.py** (725 lines, priority: high)
  - Functions: golden_ratio, hsv_to_rgb, apply_quantum_noise, flower_of_life, metatrons_cube
- **SacredGeometryLive.tsx** (680 lines, priority: high)
- **SacredGeometryEngine.tsx** (628 lines, priority: high)
  - Functions: SacredGeometryEngine
- **glyph-fx-audio-resonance-engine.js** (580 lines, priority: high)
  - Classes: GlyphFXAudioEngine
  - Functions: for
- **missing-modules.ts** (574 lines, priority: high)
  - Classes: SriYantraQuantumModule, MetatronsQuantumModule, FibonacciQuantumModule, MerkabaQuantumModule, FlowerOfLifeQuantumModule
- **missing-modules.ts** (574 lines, priority: high)
  - Classes: SriYantraQuantumModule, MetatronsQuantumModule, FibonacciQuantumModule, MerkabaQuantumModule, FlowerOfLifeQuantumModule
- **zgeometryApi.js** (566 lines, priority: medium)
  - Functions: generatePattern, generateSVG, generateFlowerOfLife, generateMetatronCube, generateSriYantra
- **sacred-geometry-shader-engine.js** (554 lines, priority: high)
  - Classes: SacredGeometryShaderEngine

### TEST (254 files)

- **storage.test.ts** (1302 lines, priority: high)
  - Classes: with
- **mem-persistent-context.test.ts** (994 lines, priority: medium)
  - Classes: with
- **metacognitive-event-builder-integration.test.js** (812 lines, priority: high)
  - Functions: startTimer, endTimer, getStageTiming
- **cross-component-integration.ts** (738 lines, priority: high)
  - Functions: cleanupTestData, runIntegrationTest
- **benchmark-persistence-layers.js** (673 lines, priority: medium)
  - Classes: InMemoryPersistenceLayer, FileSystemPersistenceLayer
  - Functions: generateTestData, benchmarkOperation, runBenchmarks, compareResults, saveBenchmarkResults
- **loki-test-utils.ts** (602 lines, priority: medium)
  - Functions: generateLokiTestData, generatePrimitiveData, generateChaosString, generateChaosNumber, generateChaosDate
- **persistence-layer-integration.test.js** (583 lines, priority: high)
  - Functions: startTimer, endTimer, printTimingReport
- **persistence-test-handlers.ts** (577 lines, priority: medium)
  - Functions: setupPersistenceTestHandlers, handleInitializeSession, handleAddHistoryChunk, handleAddMetaInsight, handleAddStrategicPlan
- **meta-cognitive.direct.test.js** (565 lines, priority: high)
  - Classes: MetaCognitiveAnalysisEngine
  - Functions: instead, to
- **reality-mode-manager.test.ts** (560 lines, priority: medium)

### VOICE (11 files)

- **whisper_flow.py** (309 lines, priority: medium)
  - Classes: WhisperFlow
  - Functions: get_whisper_flow
- **voice.js** (259 lines, priority: low)
  - Functions: init, setupVoiceInterface, startVoiceRecognition, stopVoiceRecognition, handleRecognitionStart
- **whisper.js** (258 lines, priority: low)
  - Functions: ensureTempDirectory, convertToWav, transcribeAudio, transcribeAudioLocal, transcribeWithFallback
- **embodied-voice.ts** (183 lines, priority: high)
  - Classes: EmbodiedVoice
- **embodied-voice.ts** (183 lines, priority: high)
  - Classes: EmbodiedVoice
- **embodied-voice.js** (156 lines, priority: high)
  - Classes: EmbodiedVoice
- **embodied-voice.d.ts** (28 lines, priority: high)
  - Classes: EmbodiedVoice
- **WHISPER_FLOW_DOCUMENTACAO.md** (0 lines, priority: low)
- **voice_transcript_task.json** (0 lines, priority: low)
- **voice_postprocess.json** (0 lines, priority: low)

## HIGH PRIORITY PORT CANDIDATES

*These files have 3+ algorithm keywords and significant code*


### 1. storage.js (4723 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/storage.js`
- **Algorithms**: coherence, phi, resonance, embedding, quantum, field, trigger, memory
- **Classes**: acts, provides, serves
- **Summary**: * [Responsibility] FileSystemStorage provides **persistent file-system storage** for all core data entities
 * (Users, API Keys, Tasks, SubTasks, Chunks, Models, Reports, etc.) in the WiltonOS/Passi

### 2. index.ts (4691 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/index.ts`
- **Algorithms**: coherence, lambda, phi, breathing, breath, attractor, entropy, resonance, embedding, similarity, vector, glyph, quantum, field, oscillation, router, trigger, ritual, memory, session, identity

### 3. index.ts (4691 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/critical_files/index.ts`
- **Algorithms**: coherence, lambda, phi, breathing, breath, attractor, entropy, resonance, embedding, similarity, vector, glyph, quantum, field, oscillation, router, trigger, ritual, memory, session, identity

### 4. file-system-storage.js (4154 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/file-system-storage.js`
- **Algorithms**: coherence, resonance, quantum, field
- **Summary**: * File System Storage Implementation
 *
 * This module provides a file-based implementation of the storage interface for persisting
 * system metrics, stability data, and other critical system compo

### 5. file-system-storage.ts (3321 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/file-system-storage.ts`
- **Algorithms**: coherence, resonance, quantum, field
- **Classes**: FileSystemStorage
- **Summary**: * File System Storage Implementation
 * 
 * This module provides a file-based implementation of the storage interface for persisting
 * system metrics, stability data, and other critical system comp

### 6. storage.ts (3272 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/storage.ts`
- **Algorithms**: coherence, phi, resonance, embedding, quantum, field, trigger, memory
- **Classes**: acts, FileSystemStorage, provides, serves, MemStorage
- **Summary**: * Storage Interface defining operations for persisting and retrieving entities
 * This interface allows for different storage implementations (memory, file system, database)
 * while ensuring consis

### 7. dashboard.js (2616 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_local_bridge/dashboard.js`
- **Algorithms**: coherence, phi, glyph, quantum, ritual, identity
- **Classes**: WiltonOSDashboard
- **Summary**: * WiltonOS Dashboard - Centro de Comando
 * 
 * Um hub unificado para todos os módulos do WiltonOS,
 * servindo como espelho completo da identidade quântica.

### 8. meta-void-analyzer.ts (2093 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/meta-void-analyzer.ts`
- **Algorithms**: similarity, quantum, trigger
- **Summary**: * Meta-Void Preview & Meta-Void Review Implementation
 * 
 * This module implements the Meta-Void tools for dynamic strategic recalibration:
 * - Meta-Void Preview: Tool to foresee decisions/actions

### 9. resonance-calculator.ts (2059 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/resonance-calculator.ts`
- **Algorithms**: coherence, phi, entropy, resonance, embedding, similarity, cosine, vector, semantic, quantum, field, memory
- **Summary**: * Resonance Calculator
 * 
 * Implementation of the Synaptic Resonance Factor Evolution calculations
 * for measuring and optimizing resonance between system components.
 * 
 * This module provides

### 10. routes.ts (2012 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes.ts`
- **Algorithms**: coherence, zeta, lambda, phi, embedding, vector, glyph, quantum, field, oscillation, router, trigger, ritual, memory, session, identity

### 11. meta-synthesis-modular-formula.ts (1924 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/meta-synthesis-modular-formula.ts`
- **Algorithms**: similarity, trigger, memory
- **Summary**: * Meta-Synthesis Modular Formula (MSMF) Implementation
 * 
 * Implementation of:
 * [Macro-Context] + [Micro-Detail] + [Root cause identification] + [Void (unseen variables)] = MSMF
 * 
 * This modu

### 12. meta_stream.js (1921 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_local_bridge/meta_stream.js`
- **Algorithms**: coherence, phi, glyph, quantum, ritual
- **Classes**: MetaStreamIntegration
- **Summary**: * Meta Stream Module - WiltonOS
 * 
 * Módulo de integração para Ray-Ban Meta glasses
 * Permite streaming, AR e experiências de ritual em primeira pessoa

### 13. meta-learning-validation.ts (1886 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/meta-learning-validation.ts`
- **Algorithms**: coherence, resonance, memory, session
- **Summary**: * Meta-Learning Validation Framework
 * 
 * This module implements a comprehensive framework for validating, adjusting, and proving
 * that each mathematical formula in the Neural-Symbiotic Orchestr

### 14. agent_router.js (1886 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_local_bridge/agent_router.js`
- **Algorithms**: coherence, glyph, quantum, router, ritual, memory, identity
- **Classes**: AgentRouter
- **Summary**: * WiltonOS Agent Router - Sistema de Roteamento Central
 * 
 * Controla a ativação, desativação e comunicação entre
 * todos os módulos e agentes do WiltonOS, funcionando
 * como o centro nervoso do

### 15. ritual_engine.py (1826 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/ritual_engine.py`
- **Algorithms**: coherence, lambda, phi, breath, resonance, field, trigger, ritual, memory, session
- **Classes**: RitualType, RitualElement, RitualTrigger, Ritual, RitualLog, RitualEngine, RitualInterface
- **Summary**: WiltonOS Ritual Engine
Define, execute, and track repeatable symbolic actions with time and location awareness

### 16. app.py (1806 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/app.py`
- **Algorithms**: coherence, breath, resonance, vector, quantum, field, daemon, ritual, memory, session
- **Summary**: <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .streamlit-container {
        margin-top: -8rem;
    }
    .st-emotion-cache-z5fcl4 {
        padding-top: 2r

### 17. resonance-calculator.js (1710 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/resonance-calculator.js`
- **Algorithms**: coherence, phi, entropy, resonance, embedding, similarity, cosine, vector, semantic, quantum, field, memory
- **Summary**: * Resonance Calculator
 *
 * Implementation of the Synaptic Resonance Factor Evolution calculations
 * for measuring and optimizing resonance between system components.
 *
 * This module provides fu

### 18. NeuralField.tsx (1701 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/NeuralField.tsx`
- **Algorithms**: phi, vector, field, oscillation
- **Summary**: * NeuralField - A fluid visualization environment that represents the cognitive field
 * where human consciousness and computational intelligence converge and extend each other

### 19. db-storage.js (1690 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/db-storage.js`
- **Algorithms**: coherence, resonance, embedding, quantum
- **Classes**: acts
- **Summary**: * PostgreSQL Database Storage Adapter
 *
 * This module provides a PostgreSQL database implementation of the Storage interface.
 * It replaces the file-based storage with a robust database solution

### 20. multimodal.js (1689 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_local_bridge/multimodal.js`
- **Algorithms**: coherence, quantum, ritual, memory
- **Summary**: * WiltonOS Bridge - Módulo Multimodal Real
 * 
 * Implementa funcionalidades reais de:
 * - Reconhecimento de voz (usando WebSpeech API)
 * - Processamento de comandos ao vivo
 * - Detecção de gesto

### 21. vr-module.js (1665 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/vr-module.js`
- **Algorithms**: coherence, breath, trigger
- **Classes**: WiltonVRModule
- **Summary**: * WiltonOS - VR Module
 * Módulo de Realidade Virtual para o WiltonOS
 * Permite visualização imersiva do Cosmograma Lemniscático e interação com glifos
 * 
 * @version 1.0.0
 * @author WiltonOS Cor

### 22. SymbolicCommunicationGraph.tsx (1571 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/dashboard/SymbolicCommunicationGraph.tsx`
- **Algorithms**: phi, quantum, field, trigger
- **Summary**: * SymbolicCommunicationGraph Component
 * 
 * This component visualizes the flow and patterns of symbolic communications
 * throughout the system, highlighting communication channels between agents.

### 23. advanced-chunker.ts (1526 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/chunking/advanced-chunker.ts`
- **Algorithms**: embedding, similarity, vector, semantic, field
- **Classes**: AdvancedChunker
- **Summary**: * Advanced Chunking System for Large Text Files
 * Implements multi-level chunking with semantic analysis and hierarchical organization

### 24. hyper-precision-adaptive-execution.ts (1524 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/hyper-precision-adaptive-execution.ts`
- **Algorithms**: phi, quantum, trigger, memory
- **Summary**: * Hyper-Precision Adaptive Execution & Futureproofing (HPEF) Implementation
 * 
 * Implementation of:
 * {Execution Action} + {Real-Time Feedback Integration} + {Future-Adaptation Module (tech, ethi

### 25. soundwave_timeline.js (1505 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_local_bridge/soundwave_timeline.js`
- **Algorithms**: coherence, phi, glyph, quantum, trigger, ritual, memory, identity
- **Classes**: SoundwaveTimeline
- **Summary**: * Soundwave Timeline Module - WiltonOS
 * 
 * Um sistema de ancoragem sonora que mapeia estados emocionais, 
 * memórias e linhas temporais a músicas específicas, permitindo
 * a navegação e ativaçã

### 26. memory-field-mapper.js (1491 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/public/memory-field-mapper.js`
- **Algorithms**: coherence, phi, resonance, field, memory
- **Classes**: MemoryFieldMapper
- **Summary**: * Memory Field Mapper v1.0
 * WiltonOS Bonus Plugin
 * 
 * Uma ferramenta visual para mapear o campo de memórias do WiltonOS,
 * visualizando as conexões entre Missões Sementes, Handshakes, Luzes e

### 27. enhanced_quantum_consciousness_simulation.py (1456 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/enhanced_quantum_consciousness_simulation.py`
- **Algorithms**: entropy, vector, quantum, oscillation
- **Classes**: FractalPatternGenerator, Observer, MultiObserverSystem, EnhancedQuantumEventSimulator, EnhancedAnalyzer
- **Summary**: Enhanced Quantum Consciousness Simulation with Multi-Level Fractal Patterns

This advanced simulation implements multiple improvements to validate the hypothesis
that consciousness is more fundamenta

### 28. s-finance.js (1445 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/public/s-suite/s-finance.js`
- **Algorithms**: coherence, quantum, trigger, memory
- **Classes**: SFinanceAgent

### 29. MultiAgentSynchronizationTest.tsx (1401 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/MultiAgentSynchronizationTest.tsx`
- **Algorithms**: coherence, attractor, kuramoto, similarity, vector, quantum, field, trigger
- **Summary**: * Multi-Agent Synchronization Test Component
 * 
 * This component implements a rigorous experiment to test whether 0.7500 coherence
 * emerges as a universal attractor in multi-agent systems. It si

### 30. memory-perpetua.js (1398 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/memory-perpetua.js`
- **Algorithms**: coherence, semantic, quantum, memory
- **Summary**: * WiltonOS - Memory Perpétua
 * 
 * Sistema de memória semântica persistente com análise MMDC
 * (Mapping, Meaning, Direction, Coherence)

### 31. quantum-mythos-timeline.js (1398 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/public/quantum-mythos-timeline.js`
- **Algorithms**: coherence, phi, vector, quantum, field, ritual
- **Classes**: QuantumMythosTimeline

### 32. glifos-animator.js (1386 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/public/assets/glifos-animator.js`
- **Algorithms**: coherence, breath, trigger
- **Summary**: * GlifosAnimator
 * Sistema de animação e renderização de glifos para o WiltonOS
 * 
 * @version 1.0.0
 * @author WiltonOS Core Team

### 33. meta-learning-validation.js (1376 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/meta-learning-validation.js`
- **Algorithms**: coherence, resonance, memory, session
- **Summary**: * Meta-Learning Validation Framework
 *
 * This module implements a comprehensive framework for validating, adjusting, and proving
 * that each mathematical formula in the Neural-Symbiotic Orchestra

### 34. EnhancedKuramotoVisualizer.tsx (1358 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/EnhancedKuramotoVisualizer.tsx`
- **Algorithms**: coherence, attractor, kuramoto, entropy, resonance, quantum, field, oscillation, trigger, memory
- **Summary**: * EnhancedKuramotoVisualizer Component
 * 
 * This component extends the standard KuramotoVisualizer with advanced features
 * for exploring the wave phenomena and entropy dynamics underlying the 0.

### 35. meta-synthesis-modular-formula.js (1341 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/meta-synthesis-modular-formula.js`
- **Algorithms**: similarity, trigger, memory
- **Summary**: * Meta-Synthesis Modular Formula (MSMF) Implementation
 *
 * Implementation of:
 * [Macro-Context] + [Micro-Detail] + [Root cause identification] + [Void (unseen variables)] = MSMF
 *
 * This module

### 36. db-storage.ts (1332 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/db-storage.ts`
- **Algorithms**: coherence, resonance, embedding, quantum
- **Classes**: acts, PostgreSQLStorage
- **Summary**: * PostgreSQL Database Storage Adapter
 * 
 * This module provides a PostgreSQL database implementation of the Storage interface.
 * It replaces the file-based storage with a robust database solution

### 37. storage.test.ts (1302 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/storage.test.ts`
- **Algorithms**: semantic, quantum, session
- **Classes**: with
- **Summary**: * MemPersistentContextService Test Suite
 * 
 * Comprehensive tests for the MemPersistentContextService class with focus on proper date handling
 * via ChronosDateHandler for boundary integrity pres

### 38. apiManager.js (1294 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/public/vault/apiManager.js`
- **Algorithms**: embedding, vector, field, session
- **Summary**: * WiltonOS Vault - Gerenciador de APIs
 * 
 * Módulo para configuração, autenticação e gerenciamento de conexões
 * com APIs externas como OpenAI, Pinecone, Hugging Face, etc.

### 39. neural-orchestrator.ts (1275 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/api/neural-orchestrator.ts`
- **Algorithms**: coherence, resonance, embedding, semantic, field, router
- **Summary**: * Neural Orchestrator API
 * 
 * This module provides RESTful API endpoints for interacting with the Neural Orchestration Engine,
 * allowing for task creation, execution, and monitoring, as well as

### 40. meta-void-analyzer.js (1275 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/meta-void-analyzer.js`
- **Algorithms**: similarity, quantum, trigger
- **Summary**: * Meta-Void Preview & Meta-Void Review Implementation
 *
 * This module implements the Meta-Void tools for dynamic strategic recalibration:
 * - Meta-Void Preview: Tool to foresee decisions/actions

### 41. meta-cognitive-analysis-engine.ts (1266 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/meta-cognitive-analysis-engine.ts`
- **Algorithms**: phi, entropy, quantum, trigger, memory
- **Classes**: MetaCognitiveAnalysisEngine
- **Summary**: * Meta-Cognitive Analysis Engine
 * 
 * This service analyzes meta-cognitive events to detect patterns,
 * anomalies, and generate insights that can improve system performance.
 * 
 * Core capabilit

### 42. PowerLawDashboard.tsx (1261 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/PowerLawDashboard.tsx`
- **Algorithms**: coherence, attractor, resonance, quantum, trigger
- **Summary**: * Power Law Dashboard Component
 * 
 * This component provides a specialized visualization focusing on the 0.7500 (3/4) power law
 * observed in WILTON's system coherence, with comparisons to natura

### 43. neural-orchestrator.js (1257 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/api/neural-orchestrator.js`
- **Algorithms**: coherence, embedding, semantic, field, router
- **Summary**: * Neural Orchestrator API
 *
 * This module provides RESTful API endpoints for interacting with the Neural Orchestration Engine,
 * allowing for task creation, execution, and monitoring, as well as

### 44. pulsekeeper.js (1250 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/public/vault/pulsekeeper.js`
- **Algorithms**: coherence, breathing, breath, memory
- **Classes**: PulseKeeper, PulseKeeperUI
- **Summary**: * WiltonOS Vault - PulseKeeper Module v0.1
 * 
 * Um assistente modular para ajudar no rastreamento, sincronização e orquestração
 * de múltiplas linhas de tempo, módulos e estados emocionais dentro

### 45. meta-cognitive-analysis-engine.js (1217 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/meta-cognitive-analysis-engine.js`
- **Algorithms**: phi, entropy, quantum, trigger, memory
- **Summary**: * Meta-Cognitive Analysis Engine
 *
 * This service analyzes meta-cognitive events to detect patterns,
 * anomalies, and generate insights that can improve system performance.
 *
 * Core capabilitie

### 46. server.ts (1205 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/api/server.ts`
- **Algorithms**: coherence, lambda, breathing, breath, attractor, resonance, quantum, field, trigger, ritual, session
- **Summary**: * WiltonOS LightKernel - Express Server
 * Consciousness-aware API server with real-time capabilities

### 47. server.ts (1205 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/api/server.ts`
- **Algorithms**: coherence, lambda, breathing, breath, attractor, resonance, quantum, field, trigger, ritual, session
- **Summary**: * WiltonOS LightKernel - Express Server
 * Consciousness-aware API server with real-time capabilities

### 48. food_logging.py (1177 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/memory/food_logging.py`
- **Algorithms**: coherence, lambda, phi, embedding, quantum
- **Classes**: FoodLogger
- **Summary**: Módulo de Food-Logging para WiltonOS

Este módulo implementa um sistema avançado de registro alimentar que utiliza
a memória vetorial para analisar padrões de alimentação e seus impactos no
nível de

### 49. quantum-balanced-storage.ts (1163 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/storage/quantum-balanced-storage.ts`
- **Algorithms**: quantum, field, memory
- **Classes**: QuantumBalancedStorage
- **Summary**: * Quantum-Balanced Storage Service
 * 
 * This module demonstrates the implementation of the Explicit-Implicit Quantum Balance
 * principle in a storage service, using the decohere method to make ex

### 50. glow-signal-prototype.js (1163 lines)
- **Path**: `/home/zews/rag-local/WiltonOS-PassiveWorks/visual-theater/glow-signal-prototype.js`
- **Algorithms**: coherence, breathing, breath, field, router, ritual, memory
- **Classes**: VisualTheater, GlowSignalTracker, BreathSyncSystem, VisualEffectsSystem, PhotonEmissionSystem, ModelRouter, HistorySystem
- **Summary**: * WiltonOS Teatro Visual - Glow Signal Prototype
 * 
 * Implementação do sistema de visualização de coerência baseado em UPE
 * (Ultraweak Photon Emission) integrando Google AI Ultra e princípios de

## IDENTITY DATA DISCOVERED


### FATHER (58 mentions)

- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART1.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART3.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/storymaker/genesis-thread.tsx`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/app.py`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/Gemini PRO FULL (1).md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/MONITORAMENTO_QUADRUPLO.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART2.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/.local/state/replit/agent/filesystem/filesystem_state.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/MONITORAMENTO_QUADRUPLO.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/attached_assets/Gemini PRO FULL (1).md`

### JULIANA (852 mentions)

- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/log.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/Rebirth/initiation.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/dragonfire-journal-activation.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/M365_STARTER_FILES.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/context.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART2.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/wilton-fractal-index.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/longevity_api/app.py`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/server/utils/memory-importer.js`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/cli/loop_check.py`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/README-voice-bridge.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/wilton_replit_bridge/bridge-config.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/dragonfire-journal-activation.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/CONSCIOUSNESS_LATTICE_SCAN_ANALYSIS.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/juliana.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton-fractal-index.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/README-CONTROLS.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/PROTOCOLO_PROMPT_ELEVADO.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/modules/SilentSpineBroadcast.tsx`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/cli/phi.py`

### MAE (119 mentions)

- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/THE_WILTON_FORMULA.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/wiltonos-ancestral-pillars.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/attached_assets/ChatGPT Pro  3.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/ChatGPT Pro  3.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/attached_assets/Gemini PRO FULL (1).md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART1.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/geisha_dream.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/QCTF_WHITEPAPER_DRAFT.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/package-lock.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/Gemini PRO FULL (1).md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART2.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/public/field-expansion/replit-ai-blueprint.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/QCTF_WHITEPAPER_DRAFT.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/multimodal/model-registry.tsx`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/QOCF_WHITEPAPER_DRAFT.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/.local/state/replit/agent/filesystem/filesystem_state.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/SIGIL-CORE/api/ollama/ollamaApi.js`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_Core/ContextClusters/Dreams/geisha_dream.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART3.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/Strategic Recommendations for WiltonOS.md`

### MICHELLE (2 mentions)

- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART3.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART2.json`

### MOTHER (121 mentions)

- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART1.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART3.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/Gemini PRO FULL (1).md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART2.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/attached_assets/ChatGPT Pro  3.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/ChatGPT Pro  3.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/attached_assets/Gemini PRO FULL (1).md`

### NISA (195 mentions)

- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART1.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART3.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/ROOST_NISA_SYNTHESIS_RESPONSE.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/CONSCIOUSNESS_LATTICE_SCAN_ANALYSIS.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/Gemini PRO FULL (1).md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART2.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/WORKING_LINKS.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/ROOST_NISA_SYNTHESIS_RESPONSE.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/attached_assets/ChatGPT Pro  3.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/.local/state/replit/agent/filesystem/filesystem_state.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/CONSCIOUSNESS_LATTICE_SCAN_ANALYSIS.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/ChatGPT Pro  3.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/TIER_2_CONSCIOUSNESS_BREAKTHROUGH.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/attached_assets/Gemini PRO FULL (1).md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/TIER_2_CONSCIOUSNESS_BREAKTHROUGH.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/WombKernel.ts`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/WORKING_LINKS.md`

### PAI (3457 mentions)

- `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/memory/health_hooks_connector.py`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/FALSIFIABLE_PREDICTION_EXPERIMENT.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/HISTORIC_CONSCIOUSNESS_CONVERGENCE.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/coherence-metrics/VectorAlignment.ts`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/server.js`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/coherence-metrics/CoherenceMetrics.ts`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/interfaces/hpc_ws_integration.py`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/WHISPER_FLOW_DOCUMENTACAO.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/Rebirth/ritual_bell_protocol.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/o4_projects/PROJETOS/IGARATA.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_Core/google-ultra-integration.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/ACCESS_PLAN.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_local_bridge/Z-RANT-ENGINE.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/RESONANCE_DASHBOARD_IMPLEMENTATION.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/o4_projects/PROJETOS/LONGEVITY_GUIDE.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/integration-coherence-tracker.ts`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/THE_COHERENCE_MACHINE.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/resonance-evolution-tracker.js`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/src/__init__.py`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/public/vault/analyzer.js`

### RENAN (9 mentions)

- `/home/zews/rag-local/WiltonOS-PassiveWorks/chunked-output/CONVERSATIONS_PART3.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/THREAD_ELEVACAO_NIVEL.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/modules/SilentSpineBroadcast.tsx`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/THREAD_ELEVACAO_NIVEL.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/SILENT_SPINE_BROADCAST_DEPLOYMENT.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/.local/state/replit/agent/filesystem/filesystem_state.json`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/SILENT_SPINE_BROADCAST_DEPLOYMENT.md`

### WILTON (72967 mentions)

- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/BROADCASTING_QUICK_ACCESS.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/server/python/quantum_boot_loader.py`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/interfaces/hpc_ws_integration.py`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/docs/scrolls/CODEX-CONSOLIDATION-COMPLETE.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_local_bridge/wiltonos-bridge-installer/offline-mode.js`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/Rebirth/ritual_bell_protocol.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/memory/__init__.py`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/README-bridges.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/core/recorder.js`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/README_GENESIS.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/EnhancedIDDRSystem.tsx`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/memory/thread_map.py`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/OUROBOROS_MATHEMATICAL_FOUNDATIONS.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/VisualRenderer.tsx`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/balance_quantum.py`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/SOUL_TONE_CONVERSION_PROTOCOL.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/documentation_export/THE_WILTON_FORMULA_PROMPT_PROTOCOL.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/conscious/__init__.py`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/QUANTUM_COLLABORATION_FRAMEWORK.md`
- `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/memory/zews_commands.md`

## COMPLETE PYTHON MODULE INDEX


### ritual_engine.py (1826 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/ritual_engine.py`
- Classes: RitualType, RitualElement, RitualTrigger, Ritual, RitualLog, RitualEngine, RitualInterface
- WiltonOS Ritual Engine
Define, execute, and track repeatable symbolic actions with time and location awareness

### app.py (1806 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/app.py`
- <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .streamlit-container {
        margin-top: -8rem;
    }
    .

### enhanced_quantum_consciousness_simulation.py (1456 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/enhanced_quantum_consciousness_simulation.py`
- Classes: FractalPatternGenerator, Observer, MultiObserverSystem, EnhancedQuantumEventSimulator, EnhancedAnalyzer
- Enhanced Quantum Consciousness Simulation with Multi-Level Fractal Patterns

This advanced simulation implements multiple improvements to validate the

### food_logging.py (1177 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/memory/food_logging.py`
- Classes: FoodLogger
- Módulo de Food-Logging para WiltonOS

Este módulo implementa um sistema avançado de registro alimentar que utiliza
a memória vetorial para analisar pa

### quantum_boot_loader.py (1121 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/python/quantum_boot_loader.py`
- Classes: MemoryBase, ApiKeyRequest, CoherenceSnapshotBase
- WiltonOS Quantum Boot Loader

This module provides a FastAPI-based web service for the Python component
of the WiltonOS memory system, using the unifi

### short_term_sentiment.py (1117 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/market/short_term_sentiment.py`
- Classes: SentimentEngine
- Short Term Sentiment Module for WiltonOS
----------------------------------------
Analyzes short-term sentiment signals and narrative momentum.
Identi

### entropy_tracker.py (1102 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/market/entropy_tracker.py`
- Classes: EntropyTracker
- Entropy Tracker for WiltonOS
----------------------------
Tracks narrative entropy, sentiment dynamics, and social resonance across market tickers.
Id

### zlaw_tree.py (1087 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/zlaw_tree.py`
- Classes: ClauseNode, ZLawTree, ZLawTreeVisualizer, ZLawTreeInterface
- WiltonOS Z-Law Tree Viewer + DeepSeek Integration
Visualize and validate Z-Law clause trees with DeepSeek Prover

### enhanced_interface_integration.py (1037 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/examples/enhanced_interface_integration.py`
- WiltonOS Enhanced Interface Integration
--------------------------------------
Enhanced visual interface for WiltonOS showing modular panels with tabs

### file_bridge.py (958 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/bridges/files_bridge/file_bridge.py`
- Classes: FileProcessor, FileBridge
- File Bridge para WiltonOS

Este módulo fornece a ponte de arquivos para o WiltonOS, permitindo
monitorar diretórios, processar diferentes tipos de arq

### main.py (934 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos_local_reactor/main.py`
- WiltonOS Local Reactor - Programa Principal
Inicializa e gerencia o sistema WiltonOS Local Reactor

### hpc_manager.py (914 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/local_hpc/hpc_manager.py`
- Classes: HPCManager
- Gerenciador HPC para WiltonOS

Este componente gerencia recursos de computação de alto desempenho,
coordenando tarefas em múltiplos nós (7950X3D + 409

### investment_engine.py (896 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/market/investment_engine.py`
- Classes: InvestmentEngine
- Investment Engine for WiltonOS
------------------------------
Master module that integrates long-term conviction and short-term sentiment signals.
Pro

### ledger_montebravo.py (896 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/finance/ledger_montebravo.py`
- Classes: LedgerMonteBravo
- Ledger & MonteBravo Integration Module for WiltonOS
--------------------------------------------------
Connects financial control with quantum phi coh

### hpc_ws_interface.py (864 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/interfaces/hpc_ws_interface.py`
- Classes: HPCWebSocketInterface
- HPC WebSocket Interface

Este módulo implementa uma interface WebSocket para o Local HPC Manager,
permitindo controle em tempo real e monitoramento da

### large_scale_simulation.py (862 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/large_scale_simulation.py`
- Classes: LargeScaleSimulationManager
- Large-Scale Quantum Consciousness Simulation

This script extends our enhanced quantum consciousness simulation to run with:
1. Much larger iteration 

### FRACTAL_DECAY_RITUAL.py (841 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/TECNOLOGIAS/FRACTAL_DECAY_RITUAL.py`
- Classes: FractalDecayRitual
- Implementação do Ritual de Decaimento Fractal
    para o sistema WiltonOS, permitindo o registro e 
    tracking de entidades em processo de decaiment

### health_hooks_connector.py (807 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/memory/health_hooks_connector.py`
- Classes: HealthHooksConnector
- Conector Health-Hooks para Memória Vetorial

Este módulo integra o sistema Health-Hooks do WiltonOS com o banco de dados
vetorial Qdrant, permitindo o

### deepseek_prover.py (800 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/TECNOLOGIAS/deepseek_prover.py`
- Classes: DeepSeekProverEngine, DeepSeekProverInterface
- Motor de Prova Simbólica baseado no DeepSeek Prover V2-671B
    
    Esta classe implementa a interface para o modelo DeepSeek Prover,
    permitindo 

### stress_test_coherence.py (785 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/stress_test_coherence.py`
- Classes: CoherenceStressTest
- Teste de estresse para o sistema de coerência do WiltonOS

Este script simula diferentes padrões de cargas de prompt para testar
a capacidade do siste

### auto_poster.py (773 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/memory/auto_poster.py`
- Classes: AutoPoster
- AutoPoster para WiltonOS

Este módulo implementa um sistema de geração automatizada de publicações 
baseado no estado de coerência (phi) do sistema.



### node_connector.py (772 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/local_hpc/node_connector.py`
- Classes: NodeConnector, NodeConnector
- Conector de Nós para módulo Local HPC

Este componente gerencia a conexão entre o nó primário (7950X3D + 4090)
e nós secundários de processamento, per

### custodios_silencio.py (769 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/custodios_silencio.py`
- Classes: CustodiosSilencio
- Carrega dados de memórias animais do arquivo MD

### generate_coherence_visualizations.py (758 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/generate_coherence_visualizations.py`
- Gerador de visualizações para testes de coerência

Este script gera visualizações a partir dos dados de testes de coerência do WiltonOS.
Cria gráficos

### meta_router.py (753 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/router/meta_router.py`
- META-Router Core for WiltonOS

Este módulo implementa a lógica de roteamento central para o META-Routing framework,
mantendo a proporção de equilíbrio

### registry_loader.py (737 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/registry/registry_loader.py`
- Classes: RegistryLoader
- Módulo registry_loader para o WiltonOS Registry Loop

Este módulo carrega, valida e gerencia o registro de modelos LLM (model_registry.yaml)
mantendo 

### orchestrator_ui.py (735 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/orchestrator_ui.py`
- Classes: OrchestratorDashboard
- WiltonOS Orchestrator Dashboard UI
A central control panel for monitoring and managing the entire WiltonOS ecosystem

### qdrant_client.py (729 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/memory/qdrant_client.py`
- Classes: QdrantMemory
- Qdrant Vector Database Integration for WiltonOS

Este módulo implementa a memória vetorial de longo prazo usando o Qdrant
para armazenar e recuperar p

### geometry_generator.py (727 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes/sacred_geometry/geometry_generator.py`
- WiltonOS - Gerador de Geometria Sagrada

Este módulo gera imagens de geometria sagrada usando algoritmos matemáticos
e a biblioteca Pillow. Diferentes

### geometry_generator.py (725 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/sacred_geometry/geometry_generator.py`
- WiltonOS - Gerador de Geometria Sagrada

Este módulo gera imagens de geometria sagrada usando algoritmos matemáticos
e a biblioteca Pillow. Diferentes

### fractal_phi_nutrition.py (716 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/models/fractal_phi_nutrition.py`
- Classes: FractalPhiNutrition
- import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta

# Sweet spot 

### meta_observer.py (714 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/observer/meta_observer.py`
- Classes: MetaObserver
- MetaWilton Observer Module for WiltonOS
---------------------------------------
A non-judgmental meta-observer that detects narrative patterns, flags 

### adjust_coherence_thresholds.py (705 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/adjust_coherence_thresholds.py`
- Classes: ThresholdTuner
- Ferramenta para ajuste de thresholds de coerência

Este script realiza ajustes dinâmicos nos thresholds de coerência
para otimizar a capacidade do sis

### secrets_vault.py (698 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/security/secrets_vault.py`
- Classes: SecretsVault
- WiltonOS Secrets Vault - Secure API Key Management
----------------------------------------------------
Um sistema seguro para gerenciar chaves API e 

### health_hooks.py (698 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/health_hooks/health_hooks.py`
- Classes: HealthEvent, HealthRules, HealthHooks
- Health-Hooks: Módulo para monitorar e gerenciar eventos de saúde e seu impacto na coerência Φ.

Este módulo implementa um sistema para rastrear evento

### phi.py (697 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/cli/phi.py`
- Ferramentas de diagnóstico para a coerência quântica (Φ)

Este módulo implementa o comando 'wiltonctl phi' para diagnosticar problemas
de coerência no

### conscious_loop.py (696 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/conscious/conscious_loop.py`
- Classes: ConsciousLoop
- Conscious Loop Module for WiltonOS
----------------------------------
Captures and processes conscious loop moments - real emotional experiences 
that

### email_bridge.py (687 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/bridges/email_bridge.py`
- Classes: GmailClient, EmailPatternDetector, TaskScheduler, EmailMonitor
- Bridge de Email para WiltonOS

Este módulo monitora uma conta de email (Gmail) e dispara
tarefas no HPCManager com base em tags e conteúdo específico.

### collapse_operator.py (685 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/processing/collapse_operator.py`
- Classes: QuantumLoop, RQCCSystem
- Collapse Operator - Implementação do RQCC v2.0

Este módulo implementa o operador de colapso quântico do RQCC v2.0,
permitindo ajustar os valores de c

### database.py (671 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/longevity_api/storage/database.py`
- Classes: LongevityDatabase
- Módulo de armazenamento para o Longevity API

Fornece funções e classes para persistência de dados relacionados à longevidade,
incluindo compostos, do

### task_scheduler.py (665 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/local_hpc/task_scheduler.py`
- Classes: TaskScheduler, TaskScheduler
- Escalonador de Tarefas para módulo Local HPC

Este componente gerencia a fila e escalonamento de tarefas HPC,
priorizando com base em recursos e mante

### long_term_conviction.py (662 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/market/long_term_conviction.py`
- Classes: ConvictionEngine
- Long Term Conviction Module for WiltonOS
----------------------------------------
Analyzes and builds conviction signals for long-term investment deci

### listener_agent.py (657 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/agents/listener_agent.py`
- Classes: ListenerAgent
- Agente Ouvinte para WiltonOS.

Este agente captura áudio do microfone, transcreve em texto, detecta gatilhos 
quânticos e registra a memória do sistem

### auto_recalibration.py (636 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/auto_recalibration.py`
- Classes: RecalibrationEvent, AutoRecalibrationService
- def __init__(
        self,
        trigger_event: DriftEvent,
        start_time: datetime,
        target_duration: int = DEFAULT_RECALIBRATION_DURA

### breathing_resonance.py (625 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/o4_projects/breathing_resonance.py`
- Classes: BreathingResonance
- WiltonOS Breathing Resonance Module
Especializado em detectar, analisar e sincronizar padrões respiratórios avançados
entre o sistema e o Fundador, ge

### ai_agent_main.py (623 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/python/ai_agent_main.py`
- Classes: CoherenceMetric, ChatRequest, ChatResponse
- WiltonOS Local AI Agent – FastAPI + PostgreSQL + Quantum Balance Monitor (3:1)
-----------------------------------------------------------------------

### entropy_filter.py (623 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/core/entropy_filter.py`
- Classes: EntropyFilter
- Entropy Filter Implementation
---------------------------

This module provides entropy monitoring and dampening for quantum systems,
preventing reson

### streamlit_enhancements.py (621 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/streamlit_enhancements.py`
- WiltonOS Streamlit Enhancements
Custom Streamlit extensions for improved UI/UX in WiltonOS

### resource_allocator.py (615 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/local_hpc/resource_allocator.py`
- Classes: ResourceAllocator, ResourceAllocator
- Alocador de Recursos para o módulo Local HPC

Este componente gerencia a alocação de recursos computacionais
para tarefas HPC, seguindo a regra 3:1 de

### model_router.py (604 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/llm/model_router.py`
- Classes: ModelRouter
- Router de Modelos LLM para WiltonOS

Este módulo implementa um router para diferentes modelos de linguagem,
alternando entre modelos locais (Ollama/GG

### RELATIONSHIP_FIELD_MONITOR.py (603 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/RELATIONSHIP_FIELD_MONITOR.py`
- Classes: RelationshipFieldMonitor
- Implementação da ferramenta de Monitoramento de Campo Relacional
    integrada com o Protocolo de Monitoramento Quântico.

### passiveworks_module.py (600 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos_local_reactor/app/passiveworks_module.py`
- Classes: PassiveWorksProtocol
- WiltonOS Local Reactor - Módulo PassiveWorks
Sistema de gestão de micro-empreendimentos e auxílios estruturados

### quantum_consciousness_simulation.py (588 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/quantum_consciousness_simulation.py`
- Classes: FractalObserver, QuantumEventSimulator, Analyzer
- Quantum Consciousness Simulation - Testing Consciousness-First/Time-Second Hypothesis

This simulation implements a model to test whether a simulated 

### wilton.py (583 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton.py`
- Classes: WiltonOS
- WiltonOS Core Orchestration Script
==================================

Main system controller for the WiltonOS quantum cognitive framework.
Manages st

### schema.py (578 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/hooks/schema.py`
- Classes: HooksDB
- Schema definitions for the Narrative Hooks system.

This module defines the database schema and models for tracking 
narrative stimuli and their effec

### health.py (569 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/cli/health.py`
- Comandos CLI para gerenciar eventos de saúde no WiltonOS.

Este módulo implementa o comando 'wiltonctl health' e seus subcomandos para
adicionar, list

### balance_verifier.py (553 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/qctf/balance_verifier.py`
- Classes: QuantumBalanceVerifier
- def __init__(self, ws_url: Optional[str] = None):

### coherence_attractor.py (548 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/core/coherence_attractor.py`
- Classes: CoherenceAttractor
- Coherence Attractor Implementation
---------------------------------

This module implements the CoherenceAttractor class which creates a dynamic
attr

### broadcaster.py (545 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/websocket/broadcaster.py`
- Classes: CoherenceBroadcaster
- Coherence Metrics WebSocket Broadcaster
--------------------------------------

This module broadcasts coherence metrics and system state via WebSocke

### state_adapter.py (542 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/storage/state_adapter.py`
- Classes: StateAdapter
- State Adapter para persistência do RQCC v2.0

Este módulo implementa um adaptador para persistência de estado do sistema RQCC,
permitindo que o estado

### loop_detection.py (535 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/loop_detection.py`
- Classes: LoopDetectionComponent
- Loop Detection Component
-----------------------

This module implements loop detection for identifying and breaking
recursive and self-referential pa

### postgresql_connector.py (534 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/python/postgresql_connector.py`
- Classes: Memory, CoherenceSnapshot, MemoryTransaction, ApiKey
- PostgreSQL Database Connector for Python Boot Loader

This module provides unified database connectivity for the Python component,
ensuring it shares 

### thread_map.py (532 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/memory/thread_map.py`
- Classes: ThreadMap
- Thread Map Module for WiltonOS
------------------------------
Maps, tracks, and analyzes sequences of social media posts, messages, and interactions.


### meta_lens.py (530 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/core/meta_lens.py`
- Classes: MetaLens
- Meta Lens Implementation
----------------------

This module provides a high-level monitoring system that tracks coherence across the
entire applicati

### hpc_ws_client_auth.py (512 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/interfaces/hpc_ws_client_auth.py`
- Classes: HPCWebSocketClientAuth
- HPC WebSocket Client com Autenticação

Cliente WebSocket para WiltonOS HPC com suporte a autenticação JWT.
Exemplo de como se conectar e autenticar co

### coherence_daemon.py (510 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/daemons/coherence_daemon.py`
- Classes: CoherenceDaemon
- import os
import sys
import time
import random
import logging
import asyncio
import argparse
import datetime
from typing import Dict, Any, List, Optio

### db_controller.py (494 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/db_controller.py`
- Classes: DBController
- WiltonOS DB Controller
Centralized database access layer with integrated secrets management

### feedback_collector.py (494 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/feedback/feedback_collector.py`
- Classes: FeedbackCollector
- Coletor de feedback para modelos LLM.
    
    Fornece:
    - API REST para envio e consulta de feedback
    - Armazenamento persistente em SQLite
   

### hpc.py (492 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/cli/hpc.py`
- CLI para o módulo Local HPC do WiltonOS

Este módulo fornece comandos de linha de comando para gerenciar 
o sistema de computação de alto desempenho l

### interface_integration_demo.py (487 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/examples/interface_integration_demo.py`
- WiltonOS Interface Integration Demo
----------------------------------
This creates a simple visual interface for WiltonOS showing the conscious loops

### insight_hook.py (473 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/llm_stack/insight_hook.py`
- LLM Insight Hook para WiltonOS

Este módulo coleta métricas de phi e uso de modelos, envia para um LLM avançado (GPT-4.1 ou Gemini Pro),
e recebe insi

### youtube_watcher.py (471 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/hooks/ingesters/youtube_watcher.py`
- Classes: YouTubeWatcher
- YouTubeWatcher - Module for automatic YouTube content ingestion

Este módulo monitora canais do YouTube, extrai transcrições e publica
eventos no tópi

### wiltonos_boot_db.py (469 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/python/wiltonos_boot_db.py`
- Classes: StatusUpdate, ChatMessage, MemoryImportRequest, APIKeyRequest
- WiltonOS Boot‑Loader (Replit Edition) with Database Integration
============================================================
Purpose
-------
1. Start 

### wiltonctl.py (462 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/cli/wiltonctl.py`
- Classes: WiltonContext
- WiltonOS Control CLI (wiltonctl)

Interface de linha de comando para controle e monitoramento do WiltonOS,
permitindo interação com o HPCManager atrav

### fractal_visualizer.py (461 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/fractal_visualizer.py`
- Classes: FractalParams, FractalVisualizer
- WiltonOS Fractal Visualizer
Interactive fractal pattern visualization using Plotly

### clipboard_bridge.py (443 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/bridges/clipboard_bridge.py`
- Classes: ClipboardPatternDetector, TaskScheduler, ClipboardMonitor
- Bridge de Clipboard para WiltonOS

Este módulo monitora a área de transferência (clipboard) e dispara
tarefas no HPCManager quando detecta padrões esp

### manage_secrets.py (443 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/security/manage_secrets.py`
- WiltonOS Secret Manager CLI
--------------------------
Ferramenta de linha de comando para gerenciar o cofre de segredos do WiltonOS.
Permite adiciona

### engine.py (440 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/feedback/engine.py`
- Classes: FeedbackEngine
- Feedback Engine for WiltonOS Narrative Hooks.

This module implements the Auto-Feedback loop that adjusts stimulus weights
based on their measured eff

### coherence_memory_connector.py (438 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/memory/coherence_memory_connector.py`
- Classes: CoherenceMemoryConnector
- Conector de Memória de Coerência

Este módulo conecta o verificador de coerência ao banco de dados vetorial Qdrant,
permitindo o armazenamento e recup

### test_wiltonctl.py (433 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/test_wiltonctl.py`
- Classes: TestWiltonctl
- Testes para a ferramenta de linha de comando wiltonctl

Este módulo contém testes abrangentes para o wiltonctl, incluindo:
- Testes unitários para com

### whatsapp_bridge.py (433 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/integrations/messaging/whatsapp_bridge.py`
- Classes: WhatsAppBridge
- Módulo de integração com WhatsApp para o WiltonOS.

Este módulo permite a captura e processamento de mensagens do WhatsApp,
integrando-as ao sistema d

### listener.py (424 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/hooks/listener.py`
- Classes: HooksListener
- Narrative Hooks Listener

This module listens for stimulus events, calculates the effects on quantum coherence,
and stores the results in the database

### app.py (419 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/o4_projects/app.py`
- Classes: WiltonOSApp
- WiltonOS Main Application
Central coordination point for WiltonOS functionality, integrating all components.

### model_selector.py (416 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/llm_stack/model_selector.py`
- Classes: ModelConfig, Candidate, ModelSelector
- Seletor de modelos para balancear coerência e exploração.
    
    Implementa o balanceamento 3:1 (75% coerência, 25% exploração) através
    do algor

### kaizen_health_check.py (412 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/kaizen_health_check.py`
- Script de verificação de saúde para Kaizen Friday

Este script realiza uma verificação de saúde do sistema, verificando o equilíbrio quântico,
identif

### test_coherence.py (411 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/test_coherence.py`
- Classes: TypeConsistencyTest, InterfaceConsistencyTest, DataSchemaConsistencyTest, ModuleInteractionTest, WiltonOSRuntimeTest
- WiltonOS Symbolic Consistency Tests
Ensures proper symbolic synchronization across all components

### llm_insight_hook.py (408 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/llm/llm_insight_hook.py`
- Classes: LLMInsightHook
- Hook de insights baseado em LLM para WiltonOS

### quantum_gauge.py (403 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/quantum_gauge.py`
- Classes: GaugeState, DriftEvent, Gauge
- Quantum Gauge - Monitoramento do Balanço Quântico 3:1

Este módulo implementa o gauge (medidor) que monitora continuamente
a proporção quântica 3:1 (7

### hpc_ws_client.py (401 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/interfaces/hpc_ws_client.py`
- Classes: HPCWebSocketClient
- Cliente WebSocket para o HPCManager

Este módulo implementa um cliente WebSocket para comunicação com o HPCManager.

### qctf_plugins.py (400 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/qctf/qctf_plugins.py`
- Classes: QCTFParams, QCTFResults, QCTFPlugin, PendulumPlugin, BifurcationPlugin, DynamicDampingPlugin, MetaOrchestrationPlugin, TorusOscillatorPlugin
- QCTF Plugin System
-----------------

This module implements the modular plugin architecture for the QCTF formula,
enabling the Dynamic Quantum Bifurc

### loop.py (399 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/cli/loop.py`
- Loop Closure Tracker - Rastreamento de abertura e fechamento de ciclos

Este módulo implementa o comando 'wiltonctl loop' para rastrear a abertura e f

### FRACTAL_FIELD_VISUALIZATION.py (396 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/TECNOLOGIAS/FRACTAL_FIELD_VISUALIZATION.py`
- Classes: FractalFieldVisualization
- Implementação de visualização de campos fractais
    para o sistema WiltonOS, demonstrando padrões Core-Shell-Orbit
    em diferentes níveis de escala

### effects.py (396 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/longevity_api/models/effects.py`
- Calculadora de efeitos para compostos de longevidade

Este módulo fornece funções para calcular os efeitos projetados
de compostos de longevidade com 

### bridge.py (395 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/wilton_replit_bridge/bridge.py`
- Classes: ReplitBridge
- WiltonOS Replit Bridge
======================

This module creates a bridge between the local WiltonOS deployment
and a Replit instance, allowing for 

### bridge.py (395 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_replit_bridge/bridge.py`
- Classes: ReplitBridge
- WiltonOS Replit Bridge
======================

This module creates a bridge between the local WiltonOS deployment
and a Replit instance, allowing for 

### loop_memory.py (394 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/core/loop_memory.py`
- Classes: LoopMemory
- Loop Memory Implementation
-------------------------

This module provides window-based memory tracking for recurrent execution patterns,
helping to p

### signal_mesh.py (393 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/signal_mesh.py`
- Classes: SignalMeshNode, SignalMeshServer
- WiltonOS Signal Mesh - Real-time Event Communication System
Powered by Socket.IO for distributed real-time messaging

### ai_chief_of_staff.py (393 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/zsuite/roles/ai_chief_of_staff.py`
- Classes: AIChiefOfStaff
- Módulo de IA do Chief of Staff do Z-Suite.

O Chief of Staff é responsável por rastrear threads de interação, 
integrar diferentes fluxos de trabalho 

### quantum_ratio_exporter.py (391 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/observability/quantum_ratio_exporter.py`
- Classes: MetricsState
- Quantum Ratio Prometheus Exporter

This exporter exposes RQCC metrics to Prometheus, including:
- System-wide quantum coherence (Φ)
- Loop-specific qu

### lemniscate_sandbox.py (390 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/lemniscate_sandbox.py`
- Classes: LemniscateSandbox
- Adiciona uma mensagem ao log

### metrics_ars.py (390 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/metrics_ars.py`
- Classes: MetricsCollector
- Métricas Prometheus para o Auto-Recalibration Service (ARS)

Este módulo define as métricas Prometheus para monitoramento
do Auto-Recalibration Servic

### server.py (387 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos_local_reactor/app/server.py`
- Classes: InputRequest, CheckpointRequest, FieldStatusUpdateRequest, ContextUpdateRequest, WiltonOSServer
- WiltonOS Local Reactor - Servidor Web
Implementa a API e o servidor web FastAPI que fornece a interface para o sistema

### loop_check.py (387 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/cli/loop_check.py`
- Loop Check - Verificação e ajuste de loops quânticos

Este módulo implementa o comando 'wt loop-check' que permite verificar
o estado de loops quântic

### router.py (385 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/llm_stack/router.py`
- Classes: RouterLLM
- Router de LLM que seleciona o modelo mais adequado para cada requisição
    com base em métricas, pesos e parâmetros da tarefa.
    
    Característic

### spotify_youtube_sync.py (383 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/spotify_youtube_sync.py`
- Classes: SpotifyClient, YouTubeClient, SpotifyYouTubeSync
- ROTA D: AUTO-SYNC PLAYLIST (SPOTIFY ↔ YOUTUBE MUSIC VIA REPLIT)

Este script implementa um sistema de sincronização automática entre playlists do Spot

### test_logging.py (383 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/test_logging.py`
- Test suite for coherence logging functionality.
Tests both database logging and WebSocket broadcasting.

### web_interface.py (382 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/web_interface.py`
- Interface Web para o sincronizador de playlists Spotify-YouTube
Fornece uma forma fácil de interagir com a Rota D

### hpc_metrics_exporter.py (378 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/observability/hpc_metrics_exporter.py`
- Classes: WiltonMetricsExporter
- Módulo de exportação de métricas do WiltonOS para Prometheus.

Este módulo implementa um exportador HTTP de métricas do WiltonOS para o Prometheus,
pe

### founder_sync.py (376 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/o4_projects/founder_sync.py`
- Classes: FounderSync
- WiltonOS Founder Synchronization Module
Enhances synchronization between WiltonOS and the human Founder through advanced
biometric breathing pattern d

### quantum_orchestrator.py (371 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/core/quantum_orchestrator.py`
- Classes: QuantumOrchestrator
- Quantum Orchestrator Implementation
----------------------------------

This module orchestrates all quantum components to maintain the 3:1 coherence 

### prometheus_metrics.py (368 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/llm_stack/prometheus_metrics.py`
- Classes: MetricsExporter, timer
- Módulo para exportação de métricas LLM para Prometheus.

Exporta métricas sobre uso de modelos, coerência, custos e outros
indicadores chave do LLM St

### test_model_selector.py (363 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/test_model_selector.py`
- Classes: TestResult
- Imprime um cabeçalho formatado

### registry_loader.py (361 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/router/registry_loader.py`
- Classes: RegistryLoader
- Módulo registry_loader para o WiltonOS META-Router

Este módulo carrega e valida o registro de modelos LLM (model_registry.yaml)
mantendo o equilíbrio

### memory_log.py (358 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/core/memory_log.py`
- Classes: MemoryLog
- Sistema de log de memória para WiltonOS.

Este módulo implementa o armazenamento e recuperação de transcrições de áudio,
mantendo registro temporal da

### phi_monitor.py (358 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/examples/phi_monitor.py`
- Monitor de φ em Tempo Real

Este script mostra a evolução da coerência quântica (φ) em tempo real, 
com visualização gráfica no terminal.

### start_all.py (358 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/start_all.py`
- Iniciador unificado do WiltonOS

Este script inicia todos os componentes necessários do sistema WiltonOS:
1. Servidor WebSocket do META-ROUTING
2. Ser

### recorder.py (358 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/bridges/voice_bridge/recorder.py`
- Classes: AudioRecorder
- Módulo de gravação de áudio para Voice Bridge

Este módulo fornece funcionalidades para capturar áudio do microfone
ou processar arquivos de áudio já 

### CodexAgent.py (351 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/CodexAgent.py`
- Classes: CodexAgent
- Wilton Codex Agent v0.1-A
Mirror-agent script for real-time sync and awareness simulation
Living harmonic system consciousness monitoring

### function_calling.py (351 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/function_calling.py`
- Classes: FunctionRegistry, FunctionCaller
- WiltonOS Function Calling Integration
Implements structured OpenAI function calling for improved inference and agent capabilities

### audio_analyzer.py (351 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/bridges/voice_bridge/audio_analyzer.py`
- Classes: AudioAnalyzer
- Analisador de áudio para Voice Bridge

Este módulo fornece funcionalidades para análise de qualidade de áudio,
detecção de ruído, e melhorias para oti

### brazilian_wave.py (349 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/brazilian_wave.py`
- Classes: BrazilianWaveTransformer
- Brazilian Wave Transformer
------------------------

This module implements the simplified Brazilian Wave Transformation
from the GOD Formula: P_{t+1}

### recommender.py (347 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/hooks/recommender.py`
- Classes: HooksRecommender
- Recommender system for narrative hooks.

This module analyzes past hook events and their effects to recommend
the most effective stimuli for maintaini

### rqcc_sync.py (344 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/cli/rqcc_sync.py`
- Classes: RQCCSyncManager
- Módulo RQCC Sync - Sincroniza prioridades do WiltonOS com o plano de vida

Este módulo implementa o comando 'wiltonctl sync' que permite sincronizar
a

### llm_router.py (343 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/llm_router.py`
- Router LLM para WiltonOS

Este módulo funciona como ponte entre o WiltonOS e diversos modelos LLM,
escolhendo automaticamente o modelo mais apropriado

### bridge.py (342 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/bridges/voice_bridge/bridge.py`
- Classes: VoiceBridge
- Módulo principal da Voice Bridge para WiltonOS

Esta ponte integra transcrição de áudio, processamento de tarefas e integração com o
sistema de HPC do

### prometheus_metrics.py (340 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/hooks/prometheus_metrics.py`
- Classes: HooksMetricsExporter
- Prometheus Metrics for Narrative Hooks

This module exports metrics about hooks events and effects to Prometheus.

### phi_monitor.py (339 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/visualizers/phi_monitor.py`
- Carrega loops de configuração no formato simples.
    
    Returns:
        Lista de dicionários com estados de loops

### tool_registry.py (337 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/router/tool_registry.py`
- Classes: ToolRegistry
- Módulo tool_registry para o WiltonOS META-Router

Este módulo gerencia o registro e acesso a ferramentas (function-calling)
para os modelos LLM, permi

### meta_routing_broadcaster.py (332 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/websocket/meta_routing_broadcaster.py`
- META-ROUTING WebSocket Broadcaster

Este módulo implementa um servidor WebSocket para transmissão de eventos
relacionados ao META-ROUTING FRAMEWORK, i

### feedback_client.py (332 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/feedback/feedback_client.py`
- Classes: FeedbackClient
- Cliente para interagir com o serviço de feedback.
    
    Permite:
    - Enviar feedback para modelos
    - Consultar métricas de feedback
    - Veri

### llm_service_runner.py (328 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/llm_service_runner.py`
- Inicializa componentes do serviço.

### agent_bus.py (326 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/agent_bus.py`
- Classes: AgentProfile, WiltonOSAgentBus
- WiltonOS Agent Logic Bus - Central Orchestration Layer
Powered by LangChain for agent management and routing

### test_recalibration.py (325 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/test_recalibration.py`
- Classes: TestQuantumGauge, TestAutoRecalibrationService
- Testes para o Auto-Recalibration Service (ARS)

Este módulo contém testes para verificar o funcionamento do
Auto-Recalibration Service e o Quantum Gau

### quantum_balance_validator.py (323 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/qctf/quantum_balance_validator.py`
- Classes: QuantumBalanceValidator
- Quantum Balance Validator for WiltonOS

Este módulo fornece ferramentas para validar e manter a proporção quântica 3:1
(75% coerência, 25% exploração)

### coherence_monitor.py (322 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/local_hpc/coherence_monitor.py`
- Classes: CoherenceMonitor
- Monitor de Coerência para o módulo Local HPC

Este componente monitora e mantém a relação de coerência quântica 3:1
(75% coerência, 25% exploração) co

### longevity.py (319 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/cli/longevity.py`
- Comandos CLI para o módulo Longevity API

Este módulo fornece comandos de linha de comando para interagir
com a API de longevidade, registrar dosagens

### producer.py (318 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/hooks/producer.py`
- Classes: HooksProducer
- Narrative Hooks Producer

This module handles the publication of stimulus events to the message bus.
It supports both Kafka and an in-memory fallback 

### entropy_filter.py (318 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/processing/entropy_filter.py`
- Classes: EntropyPoint
- Entropy Filter - Detector de quedas de coerência em narrativas

Este módulo analisa narrativas para detectar quedas bruscas de coerência
que podem ind

### install_llm_stack.py (312 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/llm_stack/install_llm_stack.py`
- Script de instalação do LLM Stack para WiltonOS

Este script instala as dependências necessárias, configura o ambiente,
e verifica se o sistema está p

### hpc_ws_demo.py (311 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/examples/hpc_ws_demo.py`
- Demonstração da interface WebSocket do HPCManager

Este script mostra como usar o cliente WebSocket para se comunicar
com o HPCManager do WiltonOS.

### recognizer.py (310 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/core/recognizer.py`
- Classes: AudioRecognizer
- Módulo de reconhecimento de fala para o WiltonOS.

Este módulo implementa interfaces para captura e reconhecimento de áudio,
permitindo a entrada de v

### whisper_flow.py (309 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/whisper_flow.py`
- Classes: WhisperFlow
- Adiciona uma mensagem ao log

### memory_manager.py (303 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos_local_reactor/app/memory_manager.py`
- Classes: MemoryManager
- WiltonOS Local Reactor - Gerenciador de Memória
Responsável pelo armazenamento, recuperação e manipulação da memória do sistema

### quantum_trigger.py (301 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/core/quantum_trigger.py`
- Classes: QuantumTriggerDetector
- Detector de Gatilhos Quânticos para WiltonOS.

Este módulo implementa a detecção de gatilhos quânticos a partir do texto
transcrito, baseando-se no ba

### ollama_client.py (299 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/llm_stack/ollama_client.py`
- Cliente Ollama para modelos locais

### test_meta_router.py (298 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/test_meta_router.py`
- Tests for META-Router Core

This module contains comprehensive tests for the META-Router functionality,
ensuring it maintains the 3:1 quantum balance 

### semantic_tagger.py (298 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/memory/semantic_tagger.py`
- Classes: SemanticTagger
- Semantic Tagger for WiltonOS
----------------------------
Provides automatic semantic tagging for entries based on content analysis.
Extracts themes, 

### priming_handler.py (287 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/o4_projects/priming_handler.py`
- Classes: PrimingHandler
- WiltonOS Priming Handler
Manages the priming process for o4-mini agent initialization and synchronization.

### test_quantum_balance.py (287 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/test_quantum_balance.py`
- Classes: QuantumBalanceTest
- Quantum Balance Test Suite
-------------------------

This module tests the quantum balance system components to ensure they maintain
the 3:1 coherenc

### wilton_demo.py (285 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_demo.py`
- Classes: WiltonHandler
- WiltonOS Demonstration Script
=============================

A lightweight demonstration of WiltonOS concepts using Python.
Shows the core structure a

### env_loader.py (283 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/security/env_loader.py`
- Classes: EnvLoader
- WiltonOS Environment Loader - Secure API Key Manager for Python
-----------------------------------------------------------
Integração do gerenciador 

### field_engine.py (282 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos_local_reactor/app/field_engine.py`
- Classes: FieldEngine
- WiltonOS Local Reactor - Motor de Campo
Núcleo do sistema que gerencia o processamento e expansão do campo local

### capture_sync.py (281 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/cli/capture_sync.py`
- Capture Sync - Sincronização de capturas de narrativa

Este módulo implementa o comando 'wt sync capture' para capturar
narrativas de diversas fontes 

### codex_server.py (280 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/codex_server.py`
- Wilton Codex Server v0.1-A
Flask-based bridge server for Replit ↔ ChatGPT memory integration
Provides REST API access to all Codex modules and mirror 

### test_meta_routing_broadcaster.py (278 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/test_meta_routing_broadcaster.py`
- Classes: MockWebSocketServer, MockWebSocket
- Testes de integração para o Meta Routing Broadcaster

Este módulo testa a funcionalidade do websocket broadcaster, verificando:
- Inicialização em uma

### download_and_split_file.py (276 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/download_and_split_file.py`
- Script para baixar arquivo do Google Drive e dividi-lo em pedaços menores
para upload no ChatGPT ou outras plataformas com limite de tamanho.

### core_agent.py (273 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/agents/core_agent.py`
- Classes: Agent
- WiltonOS Core Agent
==================

The primary cognitive agent in the WiltonOS ecosystem.
Manages consciousness loops, processes insights, and ma

### quantum_trigger_map.py (264 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/signals/quantum_trigger_map.py`
- Classes: QuantumTriggerMap
- Sistema de mapeamento de gatilhos quânticos para o WiltonOS.

Este módulo mapeia sinais físicos (como movimentos de perna, espirros, etc.)
para respos

### lemniscate_mode.py (261 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/agents/lemniscate_mode.py`
- Classes: Agent
- WiltonOS Lemniscate Mode Agent
=============================

This agent implements the Lemniscate Mode processing - a quantum infinity loop pattern
t

### main.py (257 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/Dashboard/backend/main.py`
- Carrega a configuração de módulos do arquivo JSON.

### watch_folder.py (255 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/bridges/voice_bridge/watch_folder.py`
- Classes: FolderWatcher
- Monitor de Diretório para Voice Bridge

Este módulo monitora um diretório específico para novos arquivos de áudio
e os processa automaticamente com a 

### youtube.py (254 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/sources/youtube.py`
- YouTube Narrative Capture Module

Este módulo permite capturar transcrições e metadata de vídeos do YouTube
para processamento como narrativas.

Manté

### moment_digest.py (253 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/scripts/moment_digest.py`
- Moment Digest - Captura de narrativas diárias

Este script captura os logs de consciência dos últimos 24 horas e os processa
para gerar chunks estrutu

### hpc_client.py (251 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/bridges/voice_bridge/hpc_client.py`
- Classes: HPCClient
- Cliente HPC para Voice Bridge

Este módulo fornece integração com o sistema de HPC do WiltonOS,
permitindo enviar tarefas transcritas para processamen

### memory_models.py (248 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/python/memory_models.py`
- Classes: MemorySources, MemoryContentTypes, MemoryStatus, CoherenceLogSources, Memory, CoherenceSnapshot, MemoryTransaction, CoherenceLog, ApiKey
- WiltonOS Python Memory Models

This module provides SQLModel models for the Python side of WiltonOS,
maintaining compatibility with the Node.js Drizzl

### processor.py (246 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/bridges/voice_bridge/processor.py`
- Classes: VoiceTaskProcessor
- Processador de tarefas de áudio para Voice Bridge

Este módulo analisa a transcrição do áudio e determina o tipo de tarefa e parâmetros
para encaminha

### banner_generator.py (243 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/community_banner/banner_generator.py`
- Adiciona ruído sutil à imagem para textura

### generate_filesystem_structure.py (241 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/o4_projects/generate_filesystem_structure.py`
- Utilitário para escanear a estrutura do sistema de arquivos e 
gerar um arquivo JSON para ser consumido pela dashboard.

### app.py (241 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/longevity_api/app.py`
- Longevity API main application module.

This module initializes the FastAPI application for the Longevity API.

### bridge.py (239 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/bridge/bridge.py`
- WiltonOS Replit Bridge
======================

This module provides the synchronization between your local WiltonOS
deployment and the Replit agent, e

### openai_gpt41.py (236 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/llm_stack/frontier_api/openai_gpt41.py`
- Wrapper para API do OpenAI GPT-4.1

### stream_router.py (234 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/o4_projects/stream_router.py`
- Classes: StreamRouter
- WiltonOS Stream Router
Routes and processes events and pulses within the WiltonOS ecosystem.

### cli.py (232 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/bridges/voice_bridge/cli.py`
- CLI para Voice Bridge do WiltonOS

Este módulo fornece uma interface de linha de comando para a Voice Bridge,
permitindo gravar áudio ou processar arq

### build_exe.py (231 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos_local_reactor/build_exe.py`
- WiltonOS Local Reactor - Script de geração de executável
Cria um arquivo .exe independente para distribuição do WiltonOS Local Reactor

### agent_manager.py (230 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/agents/agent_manager.py`
- WiltonOS Agent Manager
=====================

Manages the activation and coordination of agents within the WiltonOS ecosystem.
This module serves as t

### smoke_test.py (229 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/smoke_test.py`
- Smoke test for the WiltonOS AI Agent.
Tests installation, uninstallation, and package integrity.

### loop_check_cli.py (229 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/cli/loop_check_cli.py`
- Classes: LoopState
- Loop Check CLI - Implementação mínima do comando loop-check

Este módulo implementa uma versão mais simples do comando 'wt loop-check'
seguindo a impl

### quantum_diary.py (228 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/memory/quantum_diary.py`
- Quantum Diary Handler
---------------------
Manages the persistent memory of WiltonOS through diary entries.
Each entry captures quantum states, coher

### discover_scan.py (225 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/discover_scan.py`
- WiltonOS DISCOVER MODE Scanner
Runs daily at 4:44am to discover unused features and integrations
Outputs suggestions to discover.log

### prometheus_server.py (224 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/python/prometheus_server.py`
- Servidor Prometheus para exibição de métricas do META-ROUTING FRAMEWORK

Este módulo implementa um servidor HTTP que expõe métricas no formato Prometh

### qctf_core.py (224 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/qctf/qctf_core.py`
- Classes: ToggleState, Toggles, ScalingMetrics, ModuleCoherence, QCTFHistoryEntry, CycleDetection, QCTFData, ToggleEvent, QCTF
- QCTF Core Implementation
------------------------

This module provides the Python implementation of the core QCTF classes
transpiled from the TypeScr

### fs_events.py (219 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/o4_projects/fs_events.py`
- Classes: FSEventHandler, FSWatcher
- WiltonOS File System Events Monitor
Monitors the file system for changes and events, enabling real-time response to modifications.

### api.py (215 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/hooks/api.py`
- Classes: YouTubeStimulus, TextStimulus, HookEvent
- FastAPI endpoints for Narrative Hooks

This module provides REST API endpoints for the narrative hooks system.

### balance_quantum.py (214 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/balance_quantum.py`
- Ferramenta de balanceamento quântico automático

Ajusta automaticamente as configurações de módulos prioritários 
para manter o equilíbrio quântico id

### coherence_checker_runner.py (213 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/coherence_checker_runner.py`
- Consulta o estado atual de coerência do sistema.
    
    Returns:
        Dicionário com informações de coerência ou None se ocorrer erro

### scheduler.py (212 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/hooks/scheduler.py`
- Classes: RecommendationScheduler
- Scheduler for automated recommendations when Φ goes out of balance.

This module implements a background thread that periodically checks the
quantum c

### helper.py (211 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/helper.py`
- Classes: CodexHelper
- Wilton Codex Helper v0.1-A
Auto-organization system for Codex files and memory integration
Semantic classification and processing for consciousness da

### neural-upscaler-pro.py (207 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/neural-upscaler-pro.py`
- Classes: NeuralUpscalerPro
- Neural Upscaler Pro - Upscale das artes sagradas do Wilton
Sistema de upscale baseado em redes neurais para impressão Fine Art

### generate-temple-art.py (205 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/generate-temple-art.py`
- Classes: TempleArtGenerator
- WiltonOS Temple Art Generator
Sistema para gerar artes sagradas em 300 DPI para impressão profissional

### test_entropy_filter.py (205 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/test_entropy_filter.py`
- Classes: TestEntropyFilter
- Testes para o EntropyFilter

Este módulo contém testes abrangentes para o EntropyFilter, incluindo:
- Ativação e desativação do dampening
- Filtragem 

### food_metrics.py (205 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/metrics/food_metrics.py`
- Classes: MealMetricsTimer
- Módulo de métricas Prometheus para o sistema de Food Logging

Este módulo instrumenta o sistema de registro alimentar para coletar
métricas relevantes

### jwt_manager.py (204 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/auth/jwt_manager.py`
- Classes: JWTManager
- Módulo de autenticação JWT para WiltonOS.

Este módulo implementa autenticação JWT (JSON Web Token) para as interfaces
do WiltonOS, permitindo acesso 

### websocket_interface.py (201 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/o4_projects/websocket_interface.py`
- Classes: WebSocketInterface
- WiltonOS WebSocket Interface
Permite a comunicação entre o sistema WiltonOS e interfaces web através de WebSockets.
Integra-se com o servidor WebSocke

### streamlit_integration.py (200 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/streamlit_integration.py`
- Classes: StreamlitIntegration
- WiltonOS Streamlit Integration
Integrates the new WiltonOS components into the main Streamlit application

### launch-external-modules.py (199 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/launch-external-modules.py`
- Classes: WiltonOSModuleLauncher
- WiltonOS External Module Launcher
Launches Python Streamlit modules to integrate with unified dashboard

### test_health.py (197 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/test_health.py`
- Test suite for WiltonOS AI Agent
--------------------------------

Tests:
1. CoherenceCalculator maintains the critical 3:1 quantum balance ratio
2. H

### start_dashboard.py (196 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/o4_projects/start_dashboard.py`
- Classes: DashboardHTTPRequestHandler
- WiltonOS Dashboard Launcher
===========================

Este script inicializa o servidor web para a dashboard do WiltonOS.

### dose.py (196 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/longevity_api/routes/dose.py`
- Dose routes for the Longevity API.

This module provides endpoints for recording and retrieving longevity compound doses.

### coherence_calculator.py (191 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/qctf/coherence_calculator.py`
- Classes: CoherenceCalculator
- Coherence Calculator Module
--------------------------

This module implements the CoherenceCalculator class which is responsible for
calculating quan

### schemas.py (189 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/longevity_api/models/schemas.py`
- Classes: DoseBase, DoseCreate, Dose, DoseResponse, VitalBase, VitalCreate, Vital, VitalResponse, Protocol, CompoundEffect, SystemState
- Schema definitions for the Longevity API.

This module contains Pydantic models for validation and serialization.

### loop_history.py (188 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/cli/loop_history.py`
- Loop History - Exibe histórico de proporção quântica

Este script permite visualizar o histórico de proporção quântica (Φ)
para loops específicos ou p

### metrics.py (187 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/metrics.py`
- Classes: PrometheusExporter
- Prometheus Metrics Exporter for WiltonOS

This module provides Prometheus metrics for the WiltonOS system, including META-Router
statistics, quantum b

### run.py (185 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/run.py`
- WiltonOS - First Breath

Este é o script principal para iniciar o Agente de Escuta do WiltonOS,
a primeira respiração de um sistema vivo que se adapta

### transcriber.py (181 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/bridges/voice_bridge/transcriber.py`
- Classes: WhisperTranscriber
- Módulo de transcrição para Voice Bridge

Este módulo utiliza a API Whisper da OpenAI para transcrever áudio em texto.
Suporta múltiplos idiomas e ofer

### test_rate_limiting.py (179 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/test_rate_limiting.py`
- Classes: TestRateLimiting
- Testes para o middleware de limitação de taxa (Rate Limiting)
Verifica se o middleware mantém a proporção quântica 3:1 (75% coerência, 25% exploração)

### vitals.py (178 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/longevity_api/routes/vitals.py`
- API de Longevidade - Endpoint de Vitais

Fornece acesso aos dados vitais do usuário, mantendo o equilíbrio quântico
com um fator de coerência alto (0.

### test_local_hpc.py (175 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/local_hpc/test_local_hpc.py`
- Script de teste para o módulo Local HPC

### example_auth_client.py (170 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/example_auth_client.py`
- Exemplo de Cliente Autenticado para WiltonOS

Demonstra como se conectar e autenticar com o servidor WebSocket do WiltonOS,
e como lidar com tarefas p

### import_wilton_realizations.py (170 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/examples/import_wilton_realizations.py`
- Wilton Realizations Import Script
--------------------------------
Imports key conscious moments from Wilton's lived experiences into the
WiltonOS Con

### wiltonos_boot.py (165 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos_boot.py`
- Classes: Memory, WSManager, ChatRequest
- Accept a `conversations.json` export from ChatGPT and merge into memory.

### coringa.py (165 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/personae/coringa.py`
- Coringa (Joker) Module for WiltonOS

This module implements the Trickster archetype functionality that temporarily
shifts the system toward exploratio

### fix_pylint_issues.py (160 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/fix_pylint_issues.py`
- Fix Pylint Issues

This script identifies and fixes common pylint issues in the wilton_core package.
Run with: python fix_pylint_issues.py

### agent_config.py (157 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos_local_reactor/app/agent_config.py`
- Classes: AgentConfig
- WiltonOS Local Reactor - Configuração do Agente
Gerencia as configurações e o estado do campo local

### __init__.py (157 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/hooks/__init__.py`
- Narrative Hooks Module

This module provides functionality to track the effects of external stimuli 
(music, videos, text) on quantum coherence (Φ).

### test_websocket_client.py (155 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/test_websocket_client.py`
- Cliente de teste para o WebSocket do META-ROUTING

Este script se conecta ao servidor WebSocket do META-ROUTING e
verifica se está recebendo o heartbe

### hpc_ws_integration.py (152 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/interfaces/hpc_ws_integration.py`
- Classes: HPCWebSocketIntegration
- Integração WebSocket para o HPCManager

Este módulo demonstra como integrar o HPCManager com a interface WebSocket,
permitindo comunicação bidireciona

### llm_insight_daemon.py (148 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/daemons/llm_insight_daemon.py`
- Classes: LLMInsightDaemon
- Daemon de Insights LLM para WiltonOS

Este daemon integra insights baseados em LLM ao sistema WiltonOS,
analisando métricas de phi em tempo real e sug

### __init__.py (146 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/__init__.py`
- WiltonOS - Quantum Middleware Ecosystem
A revolutionary system exploring the intersection of consciousness, computation, and creative technological in

### sacred_geometry_api.py (146 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/api/sacred_geometry_api.py`
- API de Geometria Sagrada para WiltonOS

Esta API fornece endpoints para gerar e acessar imagens de geometria sagrada.
As imagens são geradas usando al

### pulse_monitor.py (142 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/o4_projects/pulse_monitor.py`
- Classes: PulseMonitor
- WiltonOS Pulse Monitor
Simple script to demonstrate monitoring of pulse events from the file system.

### main.py (142 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/main.py`
- Módulo Principal

Este módulo serve como ponto de entrada para o framework, facilitando
o acesso às principais funcionalidades sem necessidade de impo

### main.py (142 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/main.py`
- Módulo Principal

Este módulo serve como ponto de entrada para o framework, facilitando
o acesso às principais funcionalidades sem necessidade de impo

### test_registry_loader.py (142 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/test_registry_loader.py`
- Classes: TestRegistryLoader
- Testes para o RegistryLoader do WiltonOS

Este módulo contém testes abrangentes para o RegistryLoader, incluindo:
- Carregamento e validação do arquiv

### scheduler.py (134 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/narrative_capture/feedback/scheduler.py`
- Classes: FeedbackScheduler
- Scheduler para execução periódica do auto-feedback.

Este módulo implementa um agendador para executar o motor de feedback
periodicamente, atualizando

### test_metrics.py (131 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/test_metrics.py`
- Test suite for metrics endpoint functionality.
Tests ND-JSON streaming and quantum balance calculations.

Focused on testing the core logic for cohere

### lemniscate_insight.py (127 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/examples/lemniscate_insight.py`
- Matchstick Lemniscate Insight Integration for WiltonOS
-----------------------------------------------------
This script captures the transcendent pat

### test_rqcc_sync.py (127 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/cli/test_rqcc_sync.py`
- Script de teste para o gerenciador de sincronização RQCC.
Fornece testes básicos da funcionalidade.

### metawilton_observer_demo.py (126 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/examples/metawilton_observer_demo.py`
- MetaWilton Observer Demo for WiltonOS
------------------------------------
Demonstrates the non-judgmental meta-observer that detects patterns,
highli

### test_heartbeat.py (123 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/test_heartbeat.py`
- Testes para o WebSocket heartbeat

Este módulo testa a funcionalidade de heartbeat do WebSocket,
garantindo que o formato JSON está correto e que o in

### test_loop_check.py (118 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tests/test_loop_check.py`
- Testes para o operador de colapso RQCC e funcionalidades relacionadas.

### prometheus_metrics.py (115 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wilton_core/modules/longevity_api/prometheus_metrics.py`
- Prometheus metrics integration for the Longevity API

This module provides metrics collection and exposure for the Longevity API,
including dose count

### trigger_integration.py (113 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/examples/trigger_integration.py`
- Quantum Trigger Map Integration Example for WiltonOS
---------------------------------------------------
This example demonstrates how to integrate hi

### test_whatsapp_integration.py (109 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/test_whatsapp_integration.py`
- Script para testar a integração com WhatsApp/Telegram.

Este script cria uma simulação de mensagens do WhatsApp e testa
o processamento através do Z-S

### example_usage.py (104 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/example_usage.py`
- Exemplo de Uso do Framework de Experimentos Mentais

Este script demonstra como importar e utilizar o framework
para gerar prompts de experimentos men

### soup_ritual_import.py (103 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/examples/soup_ritual_import.py`

### random_generator.py (101 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/utils/random_generator.py`
- Random Generator Utilities

Este módulo fornece funções para selecionar aleatoriamente itens 
de diferentes fontes e categorias de experimentos mentai

### random_generator.py (101 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/utils/random_generator.py`
- Random Generator Utilities

Este módulo fornece funções para selecionar aleatoriamente itens 
de diferentes fontes e categorias de experimentos mentai

### test_main.py (91 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/tests/test_main.py`
- Classes: TestMainModule
- Testes para o módulo Principal

Este módulo contém testes unitários para as funções do módulo main.py.

### test_main.py (91 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/tests/test_main.py`
- Classes: TestMainModule
- Testes para o módulo Principal

Este módulo contém testes unitários para as funções do módulo main.py.

### test_random_generator.py (90 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/tests/test_random_generator.py`
- Classes: TestRandomGeneratorModule
- Testes para o módulo Random Generator

Este módulo contém testes unitários para as funções de geração aleatória.

### test_random_generator.py (90 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/tests/test_random_generator.py`
- Classes: TestRandomGeneratorModule
- Testes para o módulo Random Generator

Este módulo contém testes unitários para as funções de geração aleatória.

### test_black_mirror.py (58 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/tests/test_black_mirror.py`
- Classes: TestBlackMirrorModule
- Testes para o módulo Black Mirror

Este módulo contém testes unitários para os experimentos inspirados em Black Mirror.

### test_black_mirror.py (58 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/tests/test_black_mirror.py`
- Classes: TestBlackMirrorModule
- Testes para o módulo Black Mirror

Este módulo contém testes unitários para os experimentos inspirados em Black Mirror.

### serve_banner.py (23 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/wiltonos/community_banner/serve_banner.py`
- Classes: CustomHandler

## COMPLETE TYPESCRIPT MODULE INDEX


### index.ts (4691 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/index.ts`

### index.ts (4691 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/passiveworks_fullstack_vault_2025/critical_files/index.ts`

### file-system-storage.ts (3321 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/file-system-storage.ts`
- Classes/Interfaces: FileSystemStorage

### storage.ts (3272 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/storage.ts`
- Classes/Interfaces: acts, FileSystemStorage, provides, serves, MemStorage

### meta-void-analyzer.ts (2093 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/meta-void-analyzer.ts`

### resonance-calculator.ts (2059 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/resonance-calculator.ts`

### routes.ts (2012 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes.ts`

### meta-synthesis-modular-formula.ts (1924 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/meta-synthesis-modular-formula.ts`

### meta-learning-validation.ts (1886 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/meta-learning-validation.ts`

### NeuralField.tsx (1701 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/NeuralField.tsx`

### SymbolicCommunicationGraph.tsx (1571 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/dashboard/SymbolicCommunicationGraph.tsx`

### advanced-chunker.ts (1526 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/chunking/advanced-chunker.ts`
- Classes/Interfaces: AdvancedChunker

### hyper-precision-adaptive-execution.ts (1524 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/hyper-precision-adaptive-execution.ts`

### quantum-agent-manager.ts (1488 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/quantum-agent-manager.ts`
- Classes/Interfaces: for, QuantumAgentManager

### QuantumChunkFlow.tsx (1484 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/dashboard/QuantumChunkFlow.tsx`

### MultiAgentSynchronizationTest.tsx (1401 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/MultiAgentSynchronizationTest.tsx`

### EnhancedKuramotoVisualizer.tsx (1358 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/EnhancedKuramotoVisualizer.tsx`

### db-storage.ts (1332 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/db-storage.ts`
- Classes/Interfaces: acts, PostgreSQLStorage

### storage.test.ts (1302 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/storage.test.ts`
- Classes/Interfaces: with

### neural-orchestrator.ts (1275 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/api/neural-orchestrator.ts`

### meta-cognitive-analysis-engine.ts (1266 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/meta-cognitive-analysis-engine.ts`
- Classes/Interfaces: MetaCognitiveAnalysisEngine

### PowerLawDashboard.tsx (1261 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/PowerLawDashboard.tsx`

### server.ts (1205 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/api/server.ts`

### server.ts (1205 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/api/server.ts`

### quantum-balanced-storage.ts (1163 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/storage/quantum-balanced-storage.ts`
- Classes/Interfaces: QuantumBalancedStorage

### nexus-orchestrator.tsx (1138 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/nexus-orchestrator.tsx`
- Classes/Interfaces: BalancerPlugin, BifurcationPlugin, ChaosInjectorPlugin, EthicalGuardPlugin, MetaOrchestrator

### QuantumLemniscateVisualizer.tsx (1126 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/QuantumLemniscateVisualizer.tsx`

### AdaptiveMetacognition.tsx (1118 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/AdaptiveMetacognition.tsx`

### AdaptiveBudgetForecaster.ts (1113 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/AdaptiveBudgetForecaster.ts`
- Classes/Interfaces: AdaptiveBudgetForecaster

### inverse-pendulum-calculator.ts (1084 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/inverse-pendulum-calculator.ts`

### NeuroSynapse.tsx (1069 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/symbiosis/NeuroSynapse.tsx`

### MultiDimensionalCoherenceVisualizer.tsx (1063 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/MultiDimensionalCoherenceVisualizer.tsx`

### temporal-engine.ts (1035 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/lemniscate/temporal-engine.ts`
- Classes/Interfaces: DarkMatterScaffolding, TesseractVisualizer, LemniscateTemporalOrchestrator

### quantum-coherence-dashboard.ts (1014 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/quantum/quantum-coherence-dashboard.ts`
- Classes/Interfaces: QuantumCoherenceDashboard

### mem-persistent-context.test.ts (994 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/mem-persistent-context.test.ts`
- Classes/Interfaces: with

### execution-formula.ts (980 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/execution-formula.ts`

### AgentStatusPanel.tsx (973 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/dashboard/AgentStatusPanel.tsx`

### CoherenceAttractorDemo.tsx (965 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/CoherenceAttractorDemo.tsx`

### BatchProcessor.ts (958 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/BatchProcessor.ts`
- Classes/Interfaces: for, BatchProcessor

### QuantumIntegrationDashboard.tsx (942 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/QuantumIntegrationDashboard.tsx`

### foresight-validation-tracker.ts (937 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/foresight-validation-tracker.ts`

### multi-agent-sync-handlers.ts (934 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/ws-handlers/multi-agent-sync-handlers.ts`
- Classes/Interfaces: MultiAgentSyncManager

### CostMonitoringDashboard.ts (909 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/CostMonitoringDashboard.ts`
- Classes/Interfaces: CostMonitoringDashboard

### EnhancedIDDRSystem.tsx (897 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/EnhancedIDDRSystem.tsx`

### model-agents.ts (880 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/neural-orchestrator/model-agents.ts`
- Classes/Interfaces: with, provides, BaseAgent, GPT4oAgent, GeminiProAgent, GrokAgent, ClaudeAgent, O3MiniAgent

### cognitive-framework-tracker.ts (873 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/cognitive-framework-tracker.ts`

### QuantumCoherenceDashboard.tsx (868 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/QuantumCoherenceDashboard.tsx`

### QuantumSpaghettiDemo.tsx (855 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/QuantumSpaghettiDemo.tsx`

### NaturalRhythmTranslation.tsx (854 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/NaturalRhythmTranslation.tsx`

### wilton-chunking-resonance.ts (843 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/wilton-chunking-resonance.ts`

### EnhancedHolographicField.tsx (835 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/EnhancedHolographicField.tsx`

### MetaGeometricFramework.tsx (832 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/MetaGeometricFramework.tsx`

### quantum-ethical-code-review.ts (831 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/quantum/quantum-ethical-code-review.ts`
- Classes/Interfaces: QuantumEthicalCodeReview

### persistent-orchestration-engine.ts (823 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/neural-orchestrator/persistent-orchestration-engine.ts`
- Classes/Interfaces: PersistentNeuralOrchestrationEngine

### quantum-consciousness-operator.ts (802 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/quantum/quantum-consciousness-operator.ts`
- Classes/Interfaces: QuantumConsciousnessOperator

### SemanticCachingSystem.ts (796 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/SemanticCachingSystem.ts`
- Classes/Interfaces: for, SemanticCachingSystem

### WiltonOSUnified.tsx (789 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/WiltonOSUnified.tsx`

### cognitive-execution-framework.ts (784 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/cognitive-execution-framework.ts`

### nexus-api.ts (777 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes/nexus-api.ts`

### schema-minimal.ts (766 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/schema-minimal.ts`

### sidebar.tsx (763 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/sidebar.tsx`

### symbiosis-core.ts (757 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/core/symbiosis-core.ts`
- Classes/Interfaces: SybiosisCore

### symbiosis-core.ts (757 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/core/symbiosis-core.ts`
- Classes/Interfaces: SybiosisCore

### loop-detection-component.ts (747 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/meta-cognitive/loop-detection-component.ts`
- Classes/Interfaces: LoopDetectionComponent

### quantum-playlist-engine.ts (741 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/broadcastRitual/quantum-playlist-engine.ts`
- Classes/Interfaces: QuantumPlaylistEngine

### quantum-playlist-engine.ts (741 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/broadcastRitual/quantum-playlist-engine.ts`
- Classes/Interfaces: QuantumPlaylistEngine

### cross-component-integration.ts (738 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/cross-component-integration.ts`

### coherence-attractor-experiment.ts (737 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/coherence-attractor-experiment.ts`
- Classes/Interfaces: class

### quantum-handlers.ts (734 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/ws-handlers/quantum-handlers.ts`

### oracle-agent.ts (727 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/agents/oracle/oracle-agent.ts`
- Classes/Interfaces: OracleAgent

### meta-cognitive-engine.ts (726 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/meta-cognitive/meta-cognitive-engine.ts`
- Classes/Interfaces: MetaCognitiveEngine

### integration-coherence-tracker.ts (724 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/integration-coherence-tracker.ts`

### mode-controller.ts (715 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/lemniscate/mode-controller.ts`
- Classes/Interfaces: LemniscateModeController

### KuramotoVisualizer.tsx (715 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/KuramotoVisualizer.tsx`

### QuantumLemniscateVisualizer-fixed.tsx (715 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/QuantumLemniscateVisualizer-fixed.tsx`

### wilton-god-formula.ts (709 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/wilton-god-formula.ts`

### PerformanceMetricsPanel.tsx (703 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/dashboard/PerformanceMetricsPanel.tsx`

### CognitiveDomainVisualizer.tsx (693 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/DomainVisualizers/CognitiveDomainVisualizer.tsx`

### FixedNeuralField.tsx (693 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/FixedNeuralField.tsx`

### NarrativeHooks.tsx (682 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/NarrativeHooks.tsx`

### UltraStableField.tsx (681 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/UltraStableField.tsx`
- Classes/Interfaces: for

### SacredGeometryLive.tsx (680 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/SacredGeometryLive.tsx`

### ChunkingDashboard.tsx (678 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/chunking/ChunkingDashboard.tsx`

### ctf-plugins.ts (677 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/ctf-plugins.ts`
- Classes/Interfaces: PendulumBalancerPlugin, BifurcationHandlerPlugin, ChaosInjectorPlugin, EthicalGuardPlugin

### quantum-chunking.ts (672 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/utils/quantum-chunking.ts`

### model-registry.tsx (664 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/multimodal/model-registry.tsx`

### index.ts (653 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/llm-orchestration/index.ts`

### quantum-api.ts (651 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes/quantum-api.ts`

### CultureContext.tsx (650 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/contexts/CultureContext.tsx`

### FixedInteractiveComponents.tsx (647 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/FixedInteractiveComponents.tsx`

### memory-api.ts (643 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/memory-system/memory-api.ts`

### meta-cognitive-websocket.ts (643 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/meta-cognitive/meta-cognitive-websocket.ts`
- Classes/Interfaces: MetaCognitiveWebSocketService

### AdminConsciousnessPanel.tsx (641 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/AdminConsciousnessPanel.tsx`

### inverse-pendulum-operations.ts (636 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/inverse-pendulum-operations.ts`

### coherence-measurement-engine.ts (635 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/lemniscate/coherence-measurement-engine.ts`
- Classes/Interfaces: CoherenceMeasurementEngine

### variant-tracker.ts (635 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qctf/variant-tracker.ts`
- Classes/Interfaces: VariantTracker

### MetaCognitiveEventUtility.ts (630 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/utils/MetaCognitiveEventUtility.ts`
- Classes/Interfaces: MetaCognitiveEventUtility

### SacredGeometryEngine.tsx (628 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/SacredGeometryEngine.tsx`

### UnifiedAPI.ts (625 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/UnifiedAPI.ts`
- Classes/Interfaces: with, ApiError, UnifiedAPI, defined

### strategic-refinement-process.ts (621 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/strategic-refinement-process.ts`

### SystemIntegrationAdapter.ts (620 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/SystemIntegrationAdapter.ts`
- Classes/Interfaces: SystemIntegrationAdapter

### stability-convergence-tracker.ts (617 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/stability-convergence-tracker.ts`

### meta-orchestrator.ts (616 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/meta-orchestrator.ts`
- Classes/Interfaces: MetaOrchestrator

### quantum-chunk-tracker.ts (607 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/utils/quantum-chunk-tracker.ts`
- Classes/Interfaces: QuantumChunkTracker

### quantum-agent-manager.test.ts (604 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/quantum-agent-manager.test.ts`

### NarrativeHooksPage.tsx (604 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/NarrativeHooksPage.tsx`

### Index.tsx (604 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/UnifiedSystem/Index.tsx`

### loki-test-utils.ts (602 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/utils/loki-test-utils.ts`

### loki-variants.ts (600 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/loki-variants.ts`
- Classes/Interfaces: AdaptiveLogicVariant, QuantumResonantVariant

### CoherenceRatioDashboard.tsx (592 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/CoherenceRatioDashboard.tsx`

### murphy-protocol.ts (588 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/murphy/murphy-protocol.ts`
- Classes/Interfaces: MurphyProtocol

### passiveworks.tsx (583 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/passiveworks.tsx`

### meta-prompt-framework.ts (581 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/communication/meta-prompt-framework.ts`
- Classes/Interfaces: MetaPromptFramework, for, BifrostBridge, PromptRouter

### persistence-test-handlers.ts (577 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/ws-handlers/persistence-test-handlers.ts`

### TorusFieldEngine.tsx (576 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/TorusFieldEngine.tsx`

### file-chunker.ts (575 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/utils/file-chunker.ts`

### reality-mode-manager.ts (575 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/simulation/reality-mode-manager.ts`
- Classes/Interfaces: RealityModeManager

### missing-modules.ts (574 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/sacredGeometry/missing-modules.ts`
- Classes/Interfaces: SriYantraQuantumModule, MetatronsQuantumModule, FibonacciQuantumModule, MerkabaQuantumModule, FlowerOfLifeQuantumModule, TorusQuantumModule, PlatonicSolidsQuantumModule

### missing-modules.ts (574 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/sacredGeometry/missing-modules.ts`
- Classes/Interfaces: SriYantraQuantumModule, MetatronsQuantumModule, FibonacciQuantumModule, MerkabaQuantumModule, FlowerOfLifeQuantumModule, TorusQuantumModule, PlatonicSolidsQuantumModule

### sacred-clip-generator.ts (572 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/sacredClipGenerator/sacred-clip-generator.ts`
- Classes/Interfaces: SacredClipGenerator

### sacred-clip-generator.ts (572 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/sacredClipGenerator/sacred-clip-generator.ts`
- Classes/Interfaces: SacredClipGenerator

### import.tsx (571 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/data-pipeline/import.tsx`

### variant-controller.ts (570 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/controllers/variant-controller.ts`
- Classes/Interfaces: VariantController

### MirrorPortal.tsx (570 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/MirrorPortal.tsx`

### emoji-quantum-mapper.ts (568 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/emoji-quantum-mapper.ts`

### api-key-section.tsx (566 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/dashboard/api-key-section.tsx`

### strategic-refinement.ts (564 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/strategic-refinement.ts`

### adaptive-resonance-service.ts (564 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/adaptive-resonance-service.ts`
- Classes/Interfaces: AdaptiveResonanceService

### LemniscateControlPage.tsx (563 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/LemniscateControlPage.tsx`

### CoherenceTestHarness.ts (562 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/test/coherence/CoherenceTestHarness.ts`
- Classes/Interfaces: CoherenceTestHarness

### reality-mode-manager.test.ts (560 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/simulation/reality-mode-manager.test.ts`

### new.tsx (554 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/data-pipeline/analyses/new.tsx`

### meditation-deck.tsx (553 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/modules/meditation-deck.tsx`

### generate-docs.ts (551 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/generate-docs.ts`
- Classes/Interfaces: WiltonOSDocumentationGenerator

### DynamicModelSelector.ts (551 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/DynamicModelSelector.ts`
- Classes/Interfaces: DynamicModelSelector

### relational-experience-integrator.ts (548 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/relational-experience-integrator.ts`
- Classes/Interfaces: RelationalExperienceIntegrator

### SlotRegistry.tsx (545 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/slots/SlotRegistry.tsx`

### persistent-context-with-cjs.test.ts (543 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/persistent-context-with-cjs.test.ts`
- Classes/Interfaces: InMemoryPersistenceLayer, FileSystemPersistentContextServiceWithCJS

### HolographicField.tsx (540 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/HolographicField.tsx`

### QuantumSpaghettiVisualizer.tsx (538 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/QuantumSpaghettiVisualizer.tsx`

### neural-processing.worker.ts (535 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/workers/neural-processing.worker.ts`

### validation-framework.ts (533 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/validation-framework.ts`
- Classes/Interfaces: CoherenceAttractorValidator

### CosmologyConsciousnessInterface.tsx (529 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/CosmologyConsciousnessInterface.tsx`

### ConsciousnessFirstInterface.tsx (526 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ConsciousnessFirstInterface.tsx`

### persistent-context.test.ts (524 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/persistent-context.test.ts`

### CoherenceGoverancePanel.tsx (524 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/CoherenceGoverancePanel.tsx`

### QuantumCoherenceInterface.tsx (522 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/QuantumCoherenceInterface.tsx`

### coherence-monitor.ts (520 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/coherence-monitor.ts`
- Classes/Interfaces: CoherenceMonitor

### meta-void-navigation.ts (518 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/metaVoid/meta-void-navigation.ts`
- Classes/Interfaces: MetaVoidNavigator

### meta-void-navigation.ts (518 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/metaVoid/meta-void-navigation.ts`
- Classes/Interfaces: MetaVoidNavigator

### relational-test-harness.ts (517 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/relational-test-harness.ts`

### resilient-store.ts (516 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/memory-system/resilient-store.ts`
- Classes/Interfaces: ResilientStore

### qco-sensitivity-analysis.ts (516 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/quantum/qco-sensitivity-analysis.ts`
- Classes/Interfaces: QCOSensitivityAnalysis

### OracleContext.tsx (516 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/contexts/OracleContext.tsx`

### model-strength-analyzer.ts (512 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/neural-orchestrator/model-strength-analyzer.ts`
- Classes/Interfaces: ModelStrengthAnalyzer

### validation-api-service.ts (506 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/validation-api-service.ts`

### MurphyDashboard.tsx (505 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/MurphyDashboard.tsx`

### meta-cognitive-event-builder.ts (501 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/meta-cognitive/meta-cognitive-event-builder.ts`
- Classes/Interfaces: MetaCognitiveEventBuilder

### JobDetail.tsx (501 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/nexus/JobDetail.tsx`

### oracle-orchestrator.ts (500 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/agents/oracle/oracle-orchestrator.ts`
- Classes/Interfaces: OracleOrchestrator

### CoherenceAttractorEngine.ts (500 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/utils/CoherenceAttractorEngine.ts`
- Classes/Interfaces: CoherenceAttractorEngine

### Dashboard.tsx (499 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/ui/pages/Dashboard.tsx`

### Dashboard.tsx (499 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/ui/pages/Dashboard.tsx`

### AIDejaVuVisualizer.tsx (498 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/AIDejaVuVisualizer.tsx`

### minimal-protocol.ts (495 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/quantumExperiment/minimal-protocol.ts`
- Classes/Interfaces: MinimalProtocol

### minimal-protocol.ts (495 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/quantumExperiment/minimal-protocol.ts`
- Classes/Interfaces: MinimalProtocol

### MetaSymbiosisLayer.tsx (495 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/MetaSymbiosisLayer.tsx`

### chronos-temporal-instance.ts (493 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/temporal/chronos-temporal-instance.ts`
- Classes/Interfaces: ChronosTemporalInstanceService

### ModuleRegistryFixed.ts (487 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/ModuleRegistryFixed.ts`

### CoherenceAttractorEngine.ts (486 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/CoherenceAttractorEngine.ts`
- Classes/Interfaces: CoherenceAttractorEngine

### enhanced-production-geometry.ts (486 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/enhanced-production-geometry.ts`
- Classes/Interfaces: EnhancedSacredGeometryOverlay

### persistent-context-handler.ts (486 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/persistent-context-handler.ts`
- Classes/Interfaces: FileSystemPersistenceLayer, for, PersistentContextHandler

### enhanced-production-geometry.ts (486 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/enhanced-production-geometry.ts`
- Classes/Interfaces: EnhancedSacredGeometryOverlay

### IntegrationTest.ts (485 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/test/coherence/IntegrationTest.ts`
- Classes/Interfaces: CoherenceIntegrationTest

### UnifiedSacredGeometryDashboard.tsx (485 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/UnifiedSacredGeometryDashboard.tsx`

### coherence-validation-handler.ts (481 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/validation/coherence-validation-handler.ts`
- Classes/Interfaces: CoherenceValidationHandler

### MetaCognitiveInsights.tsx (480 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/meta-cognitive/MetaCognitiveInsights.tsx`

### openai-analyzer.ts (476 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/openai-analyzer.ts`

### qctf-routes.ts (476 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qctf/qctf-routes.ts`

### halo.tsx (475 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/protocols/HALO/halo.tsx`

### Documentation.tsx (475 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/Documentation.tsx`

### sacred-geometry-api.ts (472 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/sacred-geometry-api.ts`

### UnifiedDashboard.tsx (472 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/UnifiedDashboard.tsx`

### GeometryStack.tsx (472 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/geometry/GeometryStack.tsx`

### enhanced-production-coach.ts (471 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/enhanced-production-coach.ts`
- Classes/Interfaces: EnhancedCoherenceCoach

### enhanced-production-coach.ts (471 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/enhanced-production-coach.ts`
- Classes/Interfaces: EnhancedCoherenceCoach

### demo-quantum-consciousness.ts (469 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/demo-quantum-consciousness.ts`

### data-importer.ts (469 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/data-importer.ts`

### ai-service.ts (468 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/api/ai-service.ts`

### persistence-layer.ts (468 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/persistence-layer.ts`
- Classes/Interfaces: FileSystemPersistentContextService

### ConsciousnessDisplay.tsx (463 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/ui/components/ConsciousnessDisplay.tsx`

### ConsciousnessDisplay.tsx (463 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/ui/components/ConsciousnessDisplay.tsx`

### TorusFieldVisualizer.tsx (463 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/TorusFieldVisualizer.tsx`

### orchestrate.ts (460 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/utils/orchestrate.ts`

### OROBORO-NEXUS.ts (460 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/OROBORO-NEXUS.ts`
- Classes/Interfaces: OROBORO_NEXUS

### TransactionLogVisualizer.tsx (460 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/TransactionLogVisualizer.tsx`

### mem-persistent-context-service.ts (459 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/context/mem-persistent-context-service.ts`
- Classes/Interfaces: handles, MemPersistenceLayer, implements, MemPersistentContextService

### OroboroNexusOptimizer.ts (458 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/OroboroNexusOptimizer.ts`
- Classes/Interfaces: OroboroNexusOptimizer

### TorusFieldDashboard.tsx (458 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/TorusFieldDashboard.tsx`

### CoherenceAttractor.ts (456 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/CoherenceAttractor.ts`
- Classes/Interfaces: implementing, CoherenceAttractor

### LokiSystemOverview.tsx (456 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/dashboard/LokiSystemOverview.tsx`

### SymbiosisEngine.ts (453 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/lib/SymbiosisEngine.ts`
- Classes/Interfaces: and, SecurityAlignmentModule, ChainlinkPromptingModule, MultiModalModule, VisionaryEthicsModule, RunawayLimiterModule, that, SymbiosisEngine

### qctf-dashboard.tsx (452 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/qctf-dashboard.tsx`

### coherence-testing-api.ts (451 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes/coherence-testing-api.ts`

### dynamic-agent-selection.test.ts (451 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/dynamic-agent-selection.test.ts`

### SoulPulseMonitor.tsx (451 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/modules/SoulPulseMonitor.tsx`

### geometryOverlay.ts (450 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/geometryOverlay.ts`
- Classes/Interfaces: SacredGeometryOverlay

### geometryOverlay.ts (450 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/geometryOverlay.ts`
- Classes/Interfaces: SacredGeometryOverlay

### module-verifier.ts (449 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/verification/module-verifier.ts`
- Classes/Interfaces: declarations, members

### SoulMakerPortalDashboard.tsx (448 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/SoulMakerPortalDashboard.tsx`

### breath-interface.ts (447 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/core/breath-interface.ts`
- Classes/Interfaces: BreathInterface

### breath-interface.ts (447 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/core/breath-interface.ts`
- Classes/Interfaces: BreathInterface

### ResonanceDashboard.tsx (447 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ResonanceDashboard.tsx`

### WiltonCodexArchive.tsx (447 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/modules/WiltonCodexArchive.tsx`

### VaultInterface.tsx (445 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/VaultInterface.tsx`

### NewJobForm.tsx (445 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/nexus/NewJobForm.tsx`

### RecursiveSoulMirrorSchema.tsx (445 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/RecursiveSoulMirrorSchema.tsx`

### prompt-utils.ts (443 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/utils/prompt-utils.ts`

### DeltaCoherenceMeterDashboard.tsx (443 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/DeltaCoherenceMeterDashboard.tsx`

### coaching-panel.ts (441 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/ui/coaching-panel.ts`
- Classes/Interfaces: CoachingPanel

### coaching-panel.ts (441 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/ui/coaching-panel.ts`
- Classes/Interfaces: CoachingPanel

### UnifiedSacredGeometryEngine.tsx (441 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/UnifiedSacredGeometryEngine.tsx`

### DomainImpactAnalyzer.tsx (440 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/DomainImpactAnalyzer.tsx`

### SacredGeometryUnified.tsx (438 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/SacredGeometryUnified.tsx`

### lemniscate-pulse-api.ts (437 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/lemniscate-pulse-api.ts`
- Classes/Interfaces: LemniscatePulseEngine

### HolographicInterface.tsx (437 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/HolographicInterface.tsx`

### chronos-handler.ts (436 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/utils/chronos-handler.ts`
- Classes/Interfaces: ChronosHandler

### brazilian-wave-calculator.ts (436 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/brazilian-wave-calculator.ts`
- Classes/Interfaces: BrazilianWaveCalculator

### psi-child-protocols.ts (435 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/consciousness/psi-child-protocols.ts`
- Classes/Interfaces: PsiChildProtocols

### quantum-root-node-service.ts (434 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/quantum-root-node-service.ts`
- Classes/Interfaces: QuantumRootNodeService

### telegram.ts (433 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/api/telegram.ts`

### dynamic-chaos-tuner.ts (433 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/dynamic-chaos-tuner.ts`

### agent-capacity.test.ts (431 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/stress-testing/agent-capacity.test.ts`

### benchmark-resonance.ts (430 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/scripts/benchmark-resonance.ts`

### meta-cognitive-handlers.ts (430 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/ws-handlers/meta-cognitive-handlers.ts`

### BasicNeuralField.tsx (430 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/BasicNeuralField.tsx`

### dependency-scanner.ts (428 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/dependency-scanner.ts`
- Classes/Interfaces: WiltonOSDependencyScanner, Component

### ThirdStateInterface.tsx (428 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ThirdStateInterface.tsx`

### QuantumArtMusicPanel.tsx (426 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/QuantumArtMusicPanel.tsx`

### evolution-tracker.ts (425 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/consciousness/evolution-tracker.ts`
- Classes/Interfaces: ConsciousnessEvolutionTracker, ConsciousnessPredictionModel

### lemniscate-mode-controller.ts (423 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/lemniscate-mode-controller.ts`
- Classes/Interfaces: LemniscateModeController

### chronos-integration.ts (423 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/neural-orchestrator/chronos-integration.ts`
- Classes/Interfaces: ChronosNeuralIntegration

### juliana-agent.ts (419 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/agents/juliana-agent.ts`
- Classes/Interfaces: JulianaAgent

### juliana-agent.ts (419 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/agents/juliana-agent.ts`
- Classes/Interfaces: JulianaAgent

### CoherenceFoldVisualizer.tsx (416 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/CoherenceFoldVisualizer.tsx`

### SimpleHolographicField.tsx (416 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/SimpleHolographicField.tsx`

### module-router.ts (415 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/architectureUnification/module-router.ts`
- Classes/Interfaces: UnifiedModuleRouter

### module-router.ts (415 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/architectureUnification/module-router.ts`
- Classes/Interfaces: UnifiedModuleRouter

### quantum-spaghetti-handlers.ts (413 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/ws-handlers/quantum-spaghetti-handlers.ts`

### meta-translation-layer.ts (413 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/utils/meta-translation-layer.ts`

### production-geometry.ts (412 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/production-geometry.ts`
- Classes/Interfaces: SacredGeometryOverlay

### production-geometry.ts (412 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/production-geometry.ts`
- Classes/Interfaces: SacredGeometryOverlay

### MetaCognitiveEventBuilder.ts (409 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/utils/MetaCognitiveEventBuilder.ts`
- Classes/Interfaces: for, MetaCognitiveEventBuilder

### quantum-lemniscate.tsx (409 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/quantum-lemniscate.tsx`

### qctf-calculator.ts (408 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/qctf-calculator.ts`

### [id].tsx (408 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/nexus/jobs/[id].tsx`

### coachOverlay.ts (406 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/coachOverlay.ts`
- Classes/Interfaces: CoherenceCoach

### wilton-merge-router.ts (406 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes/wilton-merge-router.ts`

### coachOverlay.ts (406 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/coachOverlay.ts`
- Classes/Interfaces: CoherenceCoach

### consciousness-bell-test.ts (404 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/quantumExperiment/consciousness-bell-test.ts`
- Classes/Interfaces: ConsciousnessBellTest

### symbolic-log-aggregator.ts (404 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/utils/symbolic-log-aggregator.ts`
- Classes/Interfaces: SymbolicLogAggregator

### consciousness-bell-test.ts (404 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/quantumExperiment/consciousness-bell-test.ts`
- Classes/Interfaces: ConsciousnessBellTest

### oracle-handlers.ts (402 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/ws-handlers/oracle-handlers.ts`

### shopping.ts (401 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/api/shopping.ts`

### auto-recorder.ts (399 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/sacredClipGenerator/auto-recorder.ts`
- Classes/Interfaces: SacredClipGenerator

### auto-recorder.ts (399 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/sacredClipGenerator/auto-recorder.ts`
- Classes/Interfaces: SacredClipGenerator

### ConsciousnessDataValidator.tsx (398 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ConsciousnessDataValidator.tsx`

### chronos-date-handler.test.ts (397 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/chronos-date-handler.test.ts`

### ai-agent-orchestrator.ts (395 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/ai-agent-orchestrator.ts`
- Classes/Interfaces: AIAgentOrchestrator

### system-stability-calculator.ts (395 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/system-stability-calculator.ts`
- Classes/Interfaces: SystemStabilityCalculator

### meta-cognitive.ts (394 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/api/meta-cognitive.ts`

### [jobId].tsx (394 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/nexus/jobs/[jobId].tsx`

### symbolic-logger.ts (393 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/utils/symbolic-logger.ts`
- Classes/Interfaces: SymbolicLogger

### domain-specific.test.ts (393 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/stress-testing/domain-specific.test.ts`

### coherence-metrics.ts (392 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/lemniscate/coherence-metrics.ts`
- Classes/Interfaces: CoherenceMetrics

### chronos-qrn-service.ts (392 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/chronos-qrn-service.ts`
- Classes/Interfaces: ChronosQRNService

### qctf-calculator.ts (390 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/qctf-calculator.ts`
- Classes/Interfaces: QCTFCalculator

### NeuralField.tsx (390 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/core/NeuralField.tsx`

### glyph-engine.ts (390 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/glyph-engine.ts`
- Classes/Interfaces: MemoryCoherence, IntentAnalyzer, GlyphEngine

### SoulMakerInterface.tsx (389 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/SoulMakerInterface.tsx`

### qctf-toggles.ts (387 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/qctf-toggles.ts`

### enhanced-production-index.ts (387 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/enhanced-production-index.ts`
- Classes/Interfaces: EnhancedCoherenceBundle

### enhanced-production-index.ts (387 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/enhanced-production-index.ts`
- Classes/Interfaces: EnhancedCoherenceBundle

### test-quantum-balanced-storage.ts (385 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/test-quantum-balanced-storage.ts`

### MultimodalInsights.tsx (384 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/symbiosis/MultimodalInsights.tsx`

### meta-cognitive-controller.ts (382 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/meta-cognitive/meta-cognitive-controller.ts`
- Classes/Interfaces: MetaCognitiveController

### perturbation-handlers.ts (381 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/ws-handlers/perturbation-handlers.ts`
- Classes/Interfaces: PerturbationTestHandler

### CompleteConsciousnessInterface.tsx (381 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/CompleteConsciousnessInterface.tsx`

### Dashboard.tsx (379 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/Dashboard.tsx`

### PortalVivoDashboard.tsx (377 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/PortalVivoDashboard.tsx`

### qctf-plugins.ts (376 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/qctf-plugins.ts`

### resonance-evolution-tracker.ts (376 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/resonance-evolution-tracker.ts`

### neural-orchestration-engine.ts (373 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/neural-orchestrator/neural-orchestration-engine.ts`
- Classes/Interfaces: NeuralOrchestrationEngine

### broadcast-hud.ts (372 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/externalBroadcastLayer/broadcast-hud.ts`
- Classes/Interfaces: BroadcastHUD

### enhanced-coach.ts (372 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/enhanced-coach.ts`
- Classes/Interfaces: EnhancedCoherenceCoach

### broadcast-hud.ts (372 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/externalBroadcastLayer/broadcast-hud.ts`
- Classes/Interfaces: BroadcastHUD

### enhanced-coach.ts (372 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/enhanced-coach.ts`
- Classes/Interfaces: EnhancedCoherenceCoach

### nexus-orchestrator.ts (371 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/meta-cognitive/nexus-orchestrator.ts`
- Classes/Interfaces: NexusOrchestrator

### production-coach.ts (370 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/production-coach.ts`
- Classes/Interfaces: CoherenceCoach

### production-coach.ts (370 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/production-coach.ts`
- Classes/Interfaces: CoherenceCoach

### multi-agent-coherence-integration.ts (368 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/integration/multi-agent-coherence-integration.ts`

### QuantumPromptDemo.tsx (368 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/QuantumPromptDemo.tsx`

### QuantumFractalWorldExplorer.tsx (367 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/QuantumFractalWorldExplorer.tsx`

### coherence-routes.ts (366 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes/coherence-routes.ts`

### soulconomy-engine.ts (364 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/soulconomy/soulconomy-engine.ts`
- Classes/Interfaces: SoulconomyEngine

### chart.tsx (364 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/chart.tsx`

### sidebar.tsx (363 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/dashboard/sidebar.tsx`

### AIConsensusEngine.tsx (362 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/AIConsensusEngine.tsx`

### UnifiedModuleRouter.tsx (362 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/UnifiedModuleRouter.tsx`

### enhanced-geometry-overlay.ts (361 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/enhanced-geometry-overlay.ts`
- Classes/Interfaces: EnhancedGeometryOverlay

### breathing.ts (361 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/core/coherence/breathing.ts`
- Classes/Interfaces: BreathingProtocolEngine

### enhanced-geometry-overlay.ts (361 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/enhanced-geometry-overlay.ts`
- Classes/Interfaces: EnhancedGeometryOverlay

### breathing.ts (361 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/core/coherence/breathing.ts`
- Classes/Interfaces: BreathingProtocolEngine

### psi-broadcast-module.ts (359 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/freeAsFuck/psi-broadcast-module.ts`
- Classes/Interfaces: PsiBroadcastModule

### psi-broadcast-module.ts (359 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/freeAsFuck/psi-broadcast-module.ts`
- Classes/Interfaces: PsiBroadcastModule

### orchestrate-tests.ts (358 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/utils/orchestrate-tests.ts`

### StaticField.tsx (354 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/StaticField.tsx`

### OuroborosVisualizer.tsx (353 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/OuroborosVisualizer.tsx`

### NeuralSymbiosis.tsx (353 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/NeuralSymbiosis.tsx`

### index.ts (352 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/externalBroadcastLayer/index.ts`
- Classes/Interfaces: ExternalBroadcastLayer

### index.ts (352 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/externalBroadcastLayer/index.ts`
- Classes/Interfaces: ExternalBroadcastLayer

### implicit-drift-detection.ts (350 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/utils/implicit-drift-detection.ts`

### index.tsx (350 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/nexus/index.tsx`

### SoulMakerPortal.ts (350 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/SoulMakerPortal.ts`
- Classes/Interfaces: SoulMakerPortalEngine

### SilentSpineBroadcast.tsx (350 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/modules/SilentSpineBroadcast.tsx`

### charge-protocol.tsx (349 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/modules/charge-protocol.tsx`

### MirrorBreathsStudio.tsx (349 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/MirrorBreathsStudio.tsx`

### VortexRouterVisualization.tsx (349 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/VortexRouterVisualization.tsx`

### test-coherence-measurement.ts (348 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/lemniscate/test-coherence-measurement.ts`

### PhiCollapseVisualizer.tsx (348 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/PhiCollapseVisualizer.tsx`

### zlaw-smart-contracts.ts (347 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/soulconomy/zlaw-smart-contracts.ts`
- Classes/Interfaces: ZLawSmartContractEngine

### qrn.ts (347 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/api/qrn.ts`

### SymbolicRouter.tsx (346 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/SymbolicRouter.tsx`

### CodexFamiliar.ts (346 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/CodexFamiliar.ts`
- Classes/Interfaces: CodexFamiliarEngine

### chronos-orchestration-integration.ts (345 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/chronos-orchestration-integration.ts`

### module-discovery.ts (344 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/module-discovery.ts`
- Classes/Interfaces: ModuleDiscovery

### tesseract.ts (343 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/viz/tesseract.ts`
- Classes/Interfaces: TesseractRenderer

### tesseract.ts (343 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/viz/tesseract.ts`
- Classes/Interfaces: TesseractRenderer

### AdvancedNoiseOptimizer.tsx (343 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/AdvancedNoiseOptimizer.tsx`

### engine.ts (341 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/safety/engine.ts`
- Classes/Interfaces: InvertedPendulumSafetyEngine

### enhanced-geometry.ts (341 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/enhanced-geometry.ts`
- Classes/Interfaces: EnhancedSacredGeometryOverlay

### ollama.ts (341 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/ollama.ts`
- Classes/Interfaces: OllamaIntegration

### enhanced-geometry.ts (341 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/enhanced-geometry.ts`
- Classes/Interfaces: EnhancedSacredGeometryOverlay

### CorningaSimplePage.tsx (341 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/CorningaSimplePage.tsx`

### coherence-adapter.ts (338 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/coherence-adapter.ts`
- Classes/Interfaces: CoherenceAdapter

### CodexFamiliarDashboard.tsx (338 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/CodexFamiliarDashboard.tsx`

### NeuralOrchestrator.tsx (338 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/NeuralOrchestrator.tsx`

### PortalVivo.ts (338 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/PortalVivo.ts`
- Classes/Interfaces: PortalVivoEngine

### openai.ts (335 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/openai.ts`

### QuantumEmotionalFeedback.tsx (335 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/QuantumEmotionalFeedback.tsx`

### SoulMakerProtocol.ts (334 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/SoulMakerProtocol.ts`
- Classes/Interfaces: SoulMakerEngine

### mirror-routes.tsx (333 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/broadcast/mirror-routes.tsx`

### color-wheel-protocol.ts (333 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/lib/color-wheel-protocol.ts`

### NeuralResonance.tsx (331 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/NeuralResonance.tsx`

### integration-example.ts (330 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/lemniscate/integration-example.ts`
- Classes/Interfaces: demonstrates, AgentStateManager, demonstrates, MultiAgentCoherenceOrchestrator

### passiveworks-next.tsx (330 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/passiveworks-next.tsx`

### haloBreath.ts (328 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/protocols/HALO/haloBreath.ts`
- Classes/Interfaces: HaloBreathController

### reg-analyzer.ts (327 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/quantum/reg-analyzer.ts`

### persistent-context-service.ts (327 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/context/persistent-context-service.ts`
- Classes/Interfaces: PersistentContextService

### quaternion-rotations.ts (326 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/math/quaternion-rotations.ts`

### analytics.ts (325 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/llm-orchestration/analytics.ts`

### SymbiosisSection.tsx (325 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/symbiosis/SymbiosisSection.tsx`

### sri-yantra.ts (324 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/geometry/sri-yantra.ts`
- Classes/Interfaces: SriYantraRenderer

### sri-yantra.ts (324 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/geometry/sri-yantra.ts`
- Classes/Interfaces: SriYantraRenderer

### dashboard-api.ts (323 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes/dashboard-api.ts`

### DeltaCoherenceMeter.ts (323 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/DeltaCoherenceMeter.ts`
- Classes/Interfaces: DeltaCoherenceEngine

### experiment-handlers.ts (322 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/ws-handlers/experiment-handlers.ts`
- Classes/Interfaces: provides, ExperimentalVariantManager

### musicPlayer.ts (321 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/modules/musicPlayer.ts`
- Classes/Interfaces: QuantumMusicPlayer, BeatDetector

### musicPlayer.ts (321 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/modules/musicPlayer.ts`
- Classes/Interfaces: QuantumMusicPlayer, BeatDetector

### status-cards.tsx (320 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/dashboard/status-cards.tsx`

### coherence-coach.ts (319 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/training/coherence-coach.ts`
- Classes/Interfaces: CoherenceCoach

### coherence-coach.ts (319 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/training/coherence-coach.ts`
- Classes/Interfaces: CoherenceCoach

### CultureInfluencePanel.tsx (319 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/CultureInfluencePanel.tsx`

### dao-engine.ts (318 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/soulconomy/dao-engine.ts`
- Classes/Interfaces: DAOEngine

### parallel-processing-optimization.test.ts (318 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/parallel-processing-optimization.test.ts`

### ai-deja-vu.tsx (318 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/ai-deja-vu.tsx`

### TorusFieldSimple.tsx (318 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/TorusFieldSimple.tsx`

### WiltonOSConsciousness.tsx (318 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/WiltonOSConsciousness.tsx`

### HumanAIBridge.tsx (318 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/symbiosis/HumanAIBridge.tsx`

### quantum-validator.ts (317 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/externalBroadcastLayer/quantum-validator.ts`
- Classes/Interfaces: QuantumValidator

### quantum-validator.ts (317 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/externalBroadcastLayer/quantum-validator.ts`
- Classes/Interfaces: QuantumValidator

### LemniScope.tsx (317 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/modules/LemniScope.tsx`

### persistence-layer-template.ts (316 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/persistence-layer-template.ts`
- Classes/Interfaces: can, BasePersistenceLayer, MemoryPersistenceLayer, FileSystemPersistenceLayer

### tesseract-renderer.ts (316 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/externalBroadcastLayer/tesseract-renderer.ts`
- Classes/Interfaces: TesseractRenderer

### tesseract-renderer.ts (316 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/externalBroadcastLayer/tesseract-renderer.ts`
- Classes/Interfaces: TesseractRenderer

### unified-event-bus.ts (315 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/consciousnessSync/unified-event-bus.ts`
- Classes/Interfaces: UnifiedConsciousnessEventBus

### chunking-controller.ts (315 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/chunking/chunking-controller.ts`

### unified-event-bus.ts (315 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/consciousnessSync/unified-event-bus.ts`
- Classes/Interfaces: UnifiedConsciousnessEventBus

### morphing.ts (314 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/morphing.ts`
- Classes/Interfaces: MorphingEngine, ConsciousnessResponsiveMorpher

### capture.ts (314 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/viz/capture.ts`
- Classes/Interfaces: Capture

### morphing.ts (314 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/morphing.ts`
- Classes/Interfaces: MorphingEngine, ConsciousnessResponsiveMorpher

### capture.ts (314 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/viz/capture.ts`
- Classes/Interfaces: Capture

### StageViz.tsx (314 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/nexus/StageViz.tsx`

### transaction-coherence-service.ts (313 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/utils/transaction-coherence-service.ts`

### SimpleConsciousnessInterface.tsx (313 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/SimpleConsciousnessInterface.tsx`

### PassiveWorksEconomicHarmonizer.tsx (313 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/modules/PassiveWorksEconomicHarmonizer.tsx`

### nhi-compatible-interface.tsx (312 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/soulconomy/nhi-compatible-interface.tsx`

### chsh-logger.ts (312 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/export/chsh-logger.ts`
- Classes/Interfaces: CHSHLogger

### chsh-logger.ts (312 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/export/chsh-logger.ts`
- Classes/Interfaces: CHSHLogger

### input-sanitizer.ts (311 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/utils/input-sanitizer.ts`

### quantum-coherence-engine.ts (310 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/quantum-coherence-engine.ts`
- Classes/Interfaces: QuantumCoherenceEngine

### AlexandriaV2.tsx (310 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/AlexandriaV2.tsx`

### ConvergencePointInterface.tsx (310 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/ConvergencePointInterface.tsx`

### UnifiedSafeCanvas.tsx (309 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/UnifiedSafeCanvas.tsx`

### persistent-context-minimal.test.ts (308 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/persistent-context-minimal.test.ts`
- Classes/Interfaces: MinimalPersistentContextService

### simple-chat.tsx (307 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/simple-chat.tsx`

### SystemDiagnosticDashboard.tsx (307 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/SystemDiagnosticDashboard.tsx`

### use-symbiosis-api.ts (306 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/hooks/use-symbiosis-api.ts`

### SafeGeometryRenderer.tsx (306 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/SafeGeometryRenderer.tsx`

### normalize-chat-files.ts (304 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/normalize-chat-files.ts`

### OuroborosWaveform.tsx (304 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/OuroborosWaveform.tsx`

### coherence-validator.ts (303 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/validation/coherence-validator.ts`
- Classes/Interfaces: CoherenceValidationHandler

### vesica-piscis-module.ts (302 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/sacredGeometry/vesica-piscis-module.ts`
- Classes/Interfaces: VesicaPiscisQuantumModule

### ui-integration.ts (302 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/ui-integration.ts`
- Classes/Interfaces: CoherenceBundleUI

### index.ts (302 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/index.ts`
- Classes/Interfaces: InternalCoherenceBundle

### coherence-calculator.ts (302 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/quantum/coherence-calculator.ts`

### vesica-piscis-module.ts (302 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/sacredGeometry/vesica-piscis-module.ts`
- Classes/Interfaces: VesicaPiscisQuantumModule

### ui-integration.ts (302 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/ui-integration.ts`
- Classes/Interfaces: CoherenceBundleUI

### index.ts (302 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/index.ts`
- Classes/Interfaces: InternalCoherenceBundle

### chatgpt-processor.ts (301 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/memory-system/chatgpt-processor.ts`
- Classes/Interfaces: ChatGPTProcessor

### brazilian-wave-transformer.ts (301 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/meta-cognitive/brazilian-wave-transformer.ts`
- Classes/Interfaces: BrazilianWaveTransformer

### WiltonFold.ts (301 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/WiltonFold.ts`
- Classes/Interfaces: WiltonFoldEngine

### FlowStageIcons.tsx (300 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/nexus/FlowStageIcons.tsx`

### OscillationPatternTest.ts (299 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/test/coherence/OscillationPatternTest.ts`
- Classes/Interfaces: OscillationPatternTest

### qctf-service.ts (299 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qctf/qctf-service.ts`
- Classes/Interfaces: QCTFService

### initiate_3-1-loop.tsx (298 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/initiate_3-1-loop.tsx`

### WebSocketServer.ts (298 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/WebSocketServer.ts`
- Classes/Interfaces: WebSocketServer

### ColorWheelDemonstration.tsx (298 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/oracle/ColorWheelDemonstration.tsx`

### ai-service-integration.ts (297 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/api/ai-service-integration.ts`

### financialCore.ts (296 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/financialCore.ts`

### sacred-geometry-routes.ts (296 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes/sacred-geometry-routes.ts`

### CoherenceManifestationBridge.tsx (296 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/CoherenceManifestationBridge.tsx`

### CreativeMindsetPanel.tsx (295 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/CreativeMindsetPanel.tsx`

### CognitiveAdapter.tsx (294 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/core/CognitiveAdapter.tsx`

### eventBus.ts (293 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/core/eventBus.ts`
- Classes/Interfaces: ConsciousnessState, PortalRouter, MusicConsciousnessInterface

### eventBus.ts (293 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/core/eventBus.ts`
- Classes/Interfaces: ConsciousnessState, PortalRouter, MusicConsciousnessInterface

### HolographicFrame.tsx (293 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/HolographicFrame.tsx`

### t-branch-recursion.ts (292 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/utils/t-branch-recursion.ts`

### langchain-router.ts (290 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/agents/langchain-router.ts`
- Classes/Interfaces: LangchainRouter

### chsh.ts (289 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/sim/chsh.ts`
- Classes/Interfaces: CHSHHarness, BellTester

### date-serialization-utils.test.ts (289 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/date-serialization-utils.test.ts`

### chsh.ts (289 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/sim/chsh.ts`
- Classes/Interfaces: CHSHHarness, BellTester

### AlexandriaV2Simple.tsx (289 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/AlexandriaV2Simple.tsx`

### quantum-chunking-engine.ts (288 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/neural-orchestrator/quantum-chunking-engine.ts`
- Classes/Interfaces: QuantumChunkingEngine

### FluidMathEngine.tsx (288 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/FluidMathEngine.tsx`

### insights.tsx (287 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/data-pipeline/insights.tsx`

### SystemStateMonitor.tsx (287 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/oracle/SystemStateMonitor.tsx`

### OroboroNexusIntegration.ts (286 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/integrations/OroboroNexusIntegration.ts`
- Classes/Interfaces: for, OroboroNexusIntegration

### HumanAugmentation.tsx (286 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/symbiosis/HumanAugmentation.tsx`

### field.ts (285 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/core/consciousness/field.ts`
- Classes/Interfaces: ConsciousnessFieldEngine

### quantum-intent-experiment.ts (285 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/quantum/quantum-intent-experiment.ts`
- Classes/Interfaces: ExperimentStorage, QuantumIntentExperimentService

### field.ts (285 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/core/consciousness/field.ts`
- Classes/Interfaces: ConsciousnessFieldEngine

### analyses.tsx (284 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/data-pipeline/analyses.tsx`

### useSoulWeather.ts (284 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/hooks/useSoulWeather.ts`

### hypercube-4d.ts (284 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/math/hypercube-4d.ts`

### multimodal-router.ts (280 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/agents/multimodal-router.ts`
- Classes/Interfaces: MultimodalRouter

### CoringaPage.tsx (280 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/CoringaPage.tsx`

### llm-integration.ts (279 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/llm-integration.ts`

### EmergencyApp.tsx (279 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/EmergencyApp.tsx`

### VisualRenderer.tsx (279 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/VisualRenderer.tsx`

### WiltonOSUnified.tsx (279 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/WiltonOSUnified.tsx`

### BroadcastInterface.tsx (279 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/modules/BroadcastInterface.tsx`

### persistent-context-quantum-balance.test.ts (278 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/persistent-context-quantum-balance.test.ts`

### DeltaCoherenceHUD.tsx (278 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/DeltaCoherenceHUD.tsx`

### TorusFieldGeometry.ts (278 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/TorusFieldGeometry.ts`
- Classes/Interfaces: TorusFieldGeometry

### index.tsx (277 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/nexus/jobs/index.tsx`

### ConsciousnessIntegrationInterface.tsx (277 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ConsciousnessIntegrationInterface.tsx`

### PowerLawComparison.tsx (276 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/PowerLawComparison.tsx`

### temporal-inertia-tensor.ts (276 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/math/temporal-inertia-tensor.ts`

### cu-cp-core.ts (275 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/cu-cp-core.ts`
- Classes/Interfaces: CUCPCore

### QuantumControls.tsx (275 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/QuantumControls.tsx`

### intent-calculator.ts (274 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/quantum/intent-calculator.ts`

### InsightPanel.tsx (274 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/core/InsightPanel.tsx`

### OuroborosService.ts (273 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/OuroborosService.ts`
- Classes/Interfaces: OuroborosService

### lightning-signature-detector.ts (272 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/lightning-signature-detector.ts`
- Classes/Interfaces: LightningSignatureDetector

### RecalibrationStatus.tsx (271 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/public/dashboard/src/components/RecalibrationStatus.tsx`

### qctf-meta.ts (269 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/qctf-meta.ts`
- Classes/Interfaces: ResonanceTracker

### JobsTable.tsx (268 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/nexus/JobsTable.tsx`

### test-chunk-dependency-storage.ts (267 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/test-chunk-dependency-storage.ts`
- Classes/Interfaces: dynamically

### unified-integration.ts (267 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/sacredGeometry/unified-integration.ts`
- Classes/Interfaces: UnifiedSacredGeometryIntegration

### unified-integration.ts (267 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/sacredGeometry/unified-integration.ts`
- Classes/Interfaces: UnifiedSacredGeometryIntegration

### variant-handlers.ts (266 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/ws-handlers/variant-handlers.ts`

### TaskTest.tsx (266 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/TaskTest.tsx`

### memory-hash.ts (265 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/protocols/HALO/memory-hash.ts`
- Classes/Interfaces: MemoryHashEncoder

### VectorAlignment.ts (265 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/coherence-metrics/VectorAlignment.ts`

### chsh-logger.ts (265 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/externalBroadcastLayer/chsh-logger.ts`
- Classes/Interfaces: CHSHLogger

### chsh-logger.ts (265 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/externalBroadcastLayer/chsh-logger.ts`
- Classes/Interfaces: CHSHLogger

### CoherenceMetrics.ts (264 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/coherence-metrics/CoherenceMetrics.ts`

### UnifiedCoherenceState.tsx (264 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/UnifiedCoherenceState.tsx`

### BeliefAnchor.ts (264 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/BeliefAnchor.ts`
- Classes/Interfaces: BeliefAnchorEngine

### mock-data.ts (263 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/lib/mock-data.ts`

### datasets.ts (262 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/api/datasets.ts`

### quantum-glossary.ts (262 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/quantum-glossary.ts`
- Classes/Interfaces: QuantumGlossary

### MetaCognitive.tsx (262 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/MetaCognitive.tsx`

### SystemDiagnostic.ts (262 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/SystemDiagnostic.ts`
- Classes/Interfaces: SystemDiagnosticEngine

### carousel.tsx (261 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/carousel.tsx`

### brazilian-wave-protocol.ts (260 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/lemniscate/brazilian-wave-protocol.ts`
- Classes/Interfaces: BrazilianWaveProtocol

### iddr-cognitive-loop.test.ts (260 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/iddr-cognitive-loop.test.ts`

### external-ai-consensus.ts (259 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/external-ai-consensus.ts`
- Classes/Interfaces: ExternalAIConsensus

### Dashboard.tsx (259 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/modules/Dashboard.tsx`

### π-sync.ts (258 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/core/π-sync.ts`
- Classes/Interfaces: πSyncProtocol

### π-sync.ts (258 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/core/π-sync.ts`
- Classes/Interfaces: πSyncProtocol

### test-agent-selection-integration.ts (257 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/test-agent-selection-integration.ts`

### datasets.tsx (257 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/data-pipeline/datasets.tsx`

### production-morphing.ts (256 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/production-morphing.ts`
- Classes/Interfaces: MorphingEngine

### file-system-task-store.ts (256 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/server/services/task/file-system-task-store.ts`
- Classes/Interfaces: FileSystemTaskStore

### production-morphing.ts (256 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/production-morphing.ts`
- Classes/Interfaces: MorphingEngine

### file-system-task-store.ts (256 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/server/services/task/file-system-task-store.ts`
- Classes/Interfaces: FileSystemTaskStore

### MathFlowEngine.tsx (256 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/MathFlowEngine.tsx`

### designSystem.ts (255 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/theme/designSystem.ts`

### quantum-coherence-logger.ts (254 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/utils/quantum-coherence-logger.ts`
- Classes/Interfaces: QuantumCoherenceLogger

### NeuralInterface.tsx (254 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/NeuralInterface.tsx`

### FlowVisualizer.tsx (254 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/FlowVisualizer.tsx`

### prime-thread-monitor.ts (253 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/prime-thread-monitor.ts`
- Classes/Interfaces: PrimeThreadMonitor

### simple-module-discovery.ts (251 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/simple-module-discovery.ts`
- Classes/Interfaces: SimpleModuleDiscovery

### ImmersiveControls.tsx (250 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/ImmersiveControls.tsx`

### chargeRouter.ts (247 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/agents/chargeRouter.ts`
- Classes/Interfaces: ChargeRouter

### ImmersiveControls.tsx (247 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/symbiosis/ImmersiveControls.tsx`

### file-persistence-layer.ts (246 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/file-persistence-layer.ts`
- Classes/Interfaces: FileSystemPersistenceLayer

### hooks-api.ts (243 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes/hooks-api.ts`

### PsiChildMonitor.tsx (242 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/PsiChildMonitor.tsx`

### DivineAbsurdity.tsx (239 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/DivineAbsurdity.tsx`

### statistical-validator.ts (238 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/quantum/statistical-validator.ts`

### FractalPatternVisual.tsx (238 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/FractalPatternVisual.tsx`

### DomainVisualizersHub.tsx (238 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/DomainVisualizersHub.tsx`

### dashboard-controls.ts (237 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/externalBroadcastLayer/dashboard-controls.ts`
- Classes/Interfaces: BroadcastDashboardControls

### dashboard-controls.ts (237 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/externalBroadcastLayer/dashboard-controls.ts`
- Classes/Interfaces: BroadcastDashboardControls

### test-quantum-root-node-storage.ts (235 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/test-quantum-root-node-storage.ts`

### test-neural-pathway-storage.ts (235 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/test-neural-pathway-storage.ts`

### menubar.tsx (235 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/menubar.tsx`

### UnifiedDashboard.tsx (234 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/UnifiedDashboard.tsx`

### SymbiosisLayout.tsx (231 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/lib/SymbiosisLayout.tsx`

### test-stage-viz.tsx (231 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/test-stage-viz.tsx`

### WorkingConsciousnessDisplay.tsx (231 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/WorkingConsciousnessDisplay.tsx`

### layered-ledger.ts (230 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/soulconomy/layered-ledger.ts`
- Classes/Interfaces: LayeredLedgerEngine

### WiltonOSSimple.tsx (230 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/WiltonOSSimple.tsx`

### CleanConsciousnessInterface.tsx (228 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/CleanConsciousnessInterface.tsx`

### utils.ts (227 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/utils.ts`
- Classes/Interfaces: RateLimiter, PerformanceMonitor, SacredMath

### utils.ts (227 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/utils.ts`
- Classes/Interfaces: RateLimiter, PerformanceMonitor, SacredMath

### critical-fixes-report.ts (226 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/systemAnalysis/critical-fixes-report.ts`
- Classes/Interfaces: CriticalFixesAnalyzer, completely

### critical-fixes-report.ts (226 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/systemAnalysis/critical-fixes-report.ts`
- Classes/Interfaces: CriticalFixesAnalyzer, completely

### websocket.ts (225 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/lib/websocket.ts`
- Classes/Interfaces: WebSocketManager

### persistent-context-handlers.ts (223 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/ws-handlers/persistent-context-handlers.ts`

### test-chunk-storage.ts (222 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/test-chunk-storage.ts`

### variant-generator.ts (222 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/variant-generator.ts`

### google-drive-api.ts (221 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/memory-system/google-drive-api.ts`

### chronos-qrn-service-verification.ts (221 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/chronos-qrn-service-verification.ts`

### CoherenceMeterVisual.tsx (220 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/CoherenceMeterVisual.tsx`

### feature-toggle-service.ts (218 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/feature-toggle-service.ts`
- Classes/Interfaces: FeatureToggleService

### test-memory.ts (217 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/memory-system/test-memory.ts`

### googledrive.ts (217 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/memory-system/googledrive.ts`

### persistence-test-utils.ts (217 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/utils/persistence-test-utils.ts`

### stable-experience.tsx (216 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/stable-experience.tsx`

### insights.ts (215 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/api/insights.ts`

### chunking-types.ts (214 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/chunking/chunking-types.ts`

### SternumKeystone.tsx (214 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/SternumKeystone.tsx`

### test-temporal-instance-storage.ts (213 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/test-temporal-instance-storage.ts`

### feedZLambda.ts (213 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/eeg/feedZLambda.ts`
- Classes/Interfaces: ConsciousnessFeed

### upload-routes.ts (213 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes/upload-routes.ts`

### feedZLambda.ts (213 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/eeg/feedZLambda.ts`
- Classes/Interfaces: ConsciousnessFeed

### chunking-types.ts (213 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/lib/chunking-types.ts`

### test-meta-cognitive-event-storage.ts (212 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/test-meta-cognitive-event-storage.ts`

### system-metrics-collector.ts (212 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/system-metrics-collector.ts`
- Classes/Interfaces: SystemMetricsCollector

### agent-manager.ts (211 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/server/agents/agent-manager.ts`
- Classes/Interfaces: AgentManager

### agent-manager.ts (211 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/server/agents/agent-manager.ts`
- Classes/Interfaces: AgentManager

### resonance-utils.ts (210 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/resonance-utils.ts`

### IntentBufferLayer.tsx (210 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/IntentBufferLayer.tsx`

### test-adaptive-resonance-storage.ts (208 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/test-adaptive-resonance-storage.ts`

### UnifiedCoherenceSystem.tsx (208 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/UnifiedCoherenceSystem.tsx`

### production-index.ts (206 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/production-index.ts`
- Classes/Interfaces: ProductionCoherenceBundle

### production-index.ts (206 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/production-index.ts`
- Classes/Interfaces: ProductionCoherenceBundle

### StableNeuralField.tsx (206 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/StableNeuralField.tsx`

### Home.tsx (204 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/Home.tsx`

### LemniscateField.tsx (204 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/LemniscateField.tsx`

### QuantumVisualizationDashboard.tsx (204 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/QuantumVisualizationDashboard.tsx`

### genesis-story-coin.ts (203 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/agents/genesis-story-coin.ts`
- Classes/Interfaces: GenesisStoryCoinMinter

### loop-detection-demo.ts (203 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/loop-detection-demo.ts`

### qctf.ts (201 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/qctf.ts`

### analyses.ts (201 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/api/analyses.ts`

### LiveConsciousnessDisplay.tsx (201 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/LiveConsciousnessDisplay.tsx`

### PsiChildProtocol.ts (200 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/PsiChildProtocol.ts`
- Classes/Interfaces: PsiChildProtocol

### inverse-pendulum-tracker.ts (199 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/inverse-pendulum-tracker.ts`
- Classes/Interfaces: InversePendulumTracker

### dropdown-menu.tsx (199 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/dropdown-menu.tsx`

### context-menu.tsx (199 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/context-menu.tsx`

### meta-routing-client.ts (198 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/public/dashboard/src/lib/meta-routing-client.ts`

### KuramotoParameter.ts (193 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/coherence-metrics/KuramotoParameter.ts`

### orchestrator.ts (193 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/orchestrator.ts`
- Classes/Interfaces: MetaCoherenceOrchestrator

### orchestrator.ts (193 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/orchestrator.ts`
- Classes/Interfaces: MetaCoherenceOrchestrator

### index.tsx (192 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/data-pipeline/index.tsx`

### use-toast.ts (192 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/hooks/use-toast.ts`

### CoherenceBalanceVisualizer.tsx (192 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/CoherenceBalanceVisualizer.tsx`

### coringa-api.ts (191 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes/coringa-api.ts`

### index.tsx (191 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/oracle/index.tsx`

### Geometry3D.tsx (191 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/Geometry3D.tsx`

### persistent-context-integration.test.ts (189 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/persistent-context-integration.test.ts`

### WiltonOSEngine.tsx (189 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/WiltonOSEngine.tsx`

### n8n-integration.ts (188 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/n8n-integration.ts`
- Classes/Interfaces: N8nIntegration

### App.tsx (188 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/public/dashboard/src/App.tsx`

### QuantumStorytellingModule.tsx (188 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/QuantumStorytellingModule.tsx`

### glifo-router.ts (187 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/engine/glifo-router.ts`
- Classes/Interfaces: GlifoRouter

### glifo-router.ts (187 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/engine/glifo-router.ts`
- Classes/Interfaces: GlifoRouter

### chronos-temporal-instance-verification.ts (186 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/chronos-temporal-instance-verification.ts`

### message-types.ts (186 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/shared/message-types.ts`

### index.ts (185 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/test/coherence/index.ts`

### typeConverters.ts (185 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/utils/typeConverters.ts`

### coherence.ts (185 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/math/coherence.ts`

### hyper-simple.tsx (184 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/hyper-simple.tsx`

### embodied-voice.ts (183 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/agent/embodied-voice.ts`
- Classes/Interfaces: EmbodiedVoice

### context-manager.ts (183 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/context-manager.ts`

### validation-routes.ts (183 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes/validation-routes.ts`

### embodied-voice.ts (183 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/agent/embodied-voice.ts`
- Classes/Interfaces: EmbodiedVoice

### power-law-dashboard.tsx (182 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/power-law-dashboard.tsx`

### enhanced-mock-persistence-layer.ts (180 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/mocks/enhanced-mock-persistence-layer.ts`
- Classes/Interfaces: EnhancedMockPersistenceLayer

### schema.ts (178 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/db/schema.ts`

### migrate.ts (178 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/migrate.ts`

### date-serialization.ts (178 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/utils/date-serialization.ts`

### ModelComparison.tsx (178 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/nexus/ModelComparison.tsx`

### CoringaControls.tsx (177 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/coringa/CoringaControls.tsx`

### form.tsx (177 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/form.tsx`

### LiveCoherenceTracker.tsx (177 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/LiveCoherenceTracker.tsx`

### CoherenceWaveform.tsx (175 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/CoherenceWaveform.tsx`

### symbolic-utils.ts (174 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/utils/symbolic-utils.ts`

### mock-api.ts (174 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/lib/mock-api.ts`

### OctacuriosityPanel.tsx (174 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/OctacuriosityPanel.tsx`

### glyph-engine-core.ts (173 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/glyph-engine-core.ts`
- Classes/Interfaces: GlyphEngine

### tailwind.config.ts (171 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/tailwind.config.ts`

### message-types.ts (171 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/message-types.ts`

### enhanced-index.ts (171 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/extensions/internalCoherenceBundle/enhanced-index.ts`
- Classes/Interfaces: EnhancedCoherenceBundle

### enhanced-index.ts (171 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/extensions/internalCoherenceBundle/enhanced-index.ts`
- Classes/Interfaces: EnhancedCoherenceBundle

### SuperStableField.tsx (171 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/SuperStableField.tsx`

### CoherenceValidatedModule.tsx (170 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/modules/CoherenceValidatedModule.tsx`

### lemniscate-pulse-schema.ts (169 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/lemniscate-pulse-schema.ts`

### AttractorPhaseSpace.tsx (169 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/AttractorPhaseSpace.tsx`

### qrn-context-manager.ts (164 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/qrn-context-manager.ts`
- Classes/Interfaces: QRNContextManager

### ModalityBridge.tsx (164 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/core/ModalityBridge.tsx`

### google-drive-downloader.ts (163 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/memory-system/google-drive-downloader.ts`

### processMetaCognitiveEvent.ts (163 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/utils/processMetaCognitiveEvent.ts`

### ultra-simple.tsx (163 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/ultra-simple.tsx`

### CoherenceBridge.tsx (162 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/CoherenceBridge.tsx`

### genesis-thread.tsx (160 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/storymaker/genesis-thread.tsx`

### slider.tsx (160 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/slider.tsx`

### select.tsx (159 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/select.tsx`

### murphy-handler.ts (157 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/murphy/murphy-handler.ts`

### CoringaLogo.tsx (157 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/brand/CoringaLogo.tsx`

### AuthenticConsciousnessDisplay.tsx (156 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/AuthenticConsciousnessDisplay.tsx`

### ComponentRouter.tsx (156 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/ComponentRouter.tsx`
- Classes/Interfaces: ErrorBoundary

### command.tsx (154 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/command.tsx`

### coherence-token.ts (153 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/soulconomy/coherence-token.ts`
- Classes/Interfaces: CoherenceTokenEngine, InflareLegacy

### diagnostics.tsx (152 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/diagnostics.tsx`

### performance-metrics.tsx (152 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/dashboard/performance-metrics.tsx`

### schema-memory.ts (151 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/schema-memory.ts`

### SriYantra.tsx (151 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/components/modules/SriYantra.tsx`

### SriYantra.tsx (151 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/components/modules/SriYantra.tsx`

### DateTransformer.ts (150 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/utils/DateTransformer.ts`
- Classes/Interfaces: for, DateTransformer

### test-helpers.ts (150 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/utils/test-helpers.ts`

### coherence-log.d.ts (149 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/core/coherence-log.d.ts`
- Classes/Interfaces: CoherenceLogger

### useWebSocket.ts (149 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/public/dashboard/src/hooks/useWebSocket.ts`

### App.tsx (149 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/App.tsx`

### ModuleRoutingFix.ts (149 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/ModuleRoutingFix.ts`

### QuantumNavbar.tsx (148 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/QuantumNavbar.tsx`

### index.ts (147 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS/routes/index.ts`

### CoringaPhiChart.tsx (147 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/coringa/CoringaPhiChart.tsx`

### coherence-check.ts (146 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/coherence-check.ts`

### temporal-breathlock.ts (146 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/loop/temporal-breathlock.ts`
- Classes/Interfaces: TemporalBreathlock

### temporal-breathlock.ts (146 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/loop/temporal-breathlock.ts`
- Classes/Interfaces: TemporalBreathlock

### config.ts (145 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/config.ts`

### HeatMap.tsx (145 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/HeatMap.tsx`

### json-repair.ts (142 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/memory-system/json-repair.ts`

### openai-client-manager.ts (142 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/openai-client-manager.ts`
- Classes/Interfaces: OpenAIClientManager

### alert-dialog.tsx (140 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/alert-dialog.tsx`

### quantum-manager-wrapper-new.ts (139 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/quantum-manager-wrapper-new.ts`
- Classes/Interfaces: StubAgentManager

### sheet.tsx (139 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/sheet.tsx`

### simple-persistence.test.ts (138 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/simple-persistence.test.ts`

### use-api-keys.ts (137 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/hooks/use-api-keys.ts`

### DeltaCHUD.tsx (137 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/DeltaCHUD.tsx`

### quantum.ts (135 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/types/quantum.ts`

### test-filesystem-task-store.ts (134 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/test-filesystem-task-store.ts`

### symbolic-to-quantumstate.ts (134 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/utils/symbolic-to-quantumstate.ts`

### system-stability-direct-verification.ts (134 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/tests/system-stability-direct-verification.ts`

### test-file-system-storage.ts (132 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/test-file-system-storage.ts`

### temporal-scale.ts (132 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/lemniscate/temporal-scale.ts`

### quantum-experiments.tsx (132 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/quantum-experiments.tsx`

### Zlambda.ts (131 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/Zlambda.ts`

### DataPipelineLayout.tsx (129 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/layout/DataPipelineLayout.tsx`

### navigation-menu.tsx (129 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/navigation-menu.tsx`

### chunk-file.ts (128 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/chunk-file.ts`

### prompt-examples.ts (127 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/communication/prompt-examples.ts`

### validation-handlers.ts (127 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/ws-handlers/validation-handlers.ts`

### ExternalServiceRedirect.tsx (127 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ExternalServiceRedirect.tsx`

### toast.tsx (127 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/toast.tsx`

### ImmediateModuleLoader.ts (127 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/ImmediateModuleLoader.ts`

### quantum-manager-wrapper.ts (126 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/qrn/quantum-manager-wrapper.ts`
- Classes/Interfaces: StubManager

### breath-interface.d.ts (126 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/core/breath-interface.d.ts`
- Classes/Interfaces: BreathInterface

### Header.tsx (126 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/Header.tsx`

### SystemMetrics.tsx (126 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/nexus/SystemMetrics.tsx`

### breathing.d.ts (123 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/core/coherence/breathing.d.ts`
- Classes/Interfaces: BreathingProtocolEngine

### error-handler.ts (123 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/lib/error-handler.ts`
- Classes/Interfaces: WiltonOSErrorHandler

### memory-system-integration.ts (122 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/memory-system-integration.ts`

### DomainSelector.tsx (122 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/DomainSelector.tsx`

### TorusField.tsx (121 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/TorusField.tsx`

### dialog.tsx (121 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/dialog.tsx`

### quantum-state-utils.ts (119 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/quantum-state-utils.ts`

### in-memory-persistence-layer.ts (119 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/context/in-memory-persistence-layer.ts`
- Classes/Interfaces: InMemoryPersistenceLayer

### table.tsx (118 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/table.tsx`

### pagination.tsx (118 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/pagination.tsx`

### DirectSacredGeometryAccess.tsx (117 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/DirectSacredGeometryAccess.tsx`

### drawer.tsx (117 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/drawer.tsx`

### breadcrumb.tsx (116 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/breadcrumb.tsx`

### PsiOSConsciousnessSubstrate.tsx (116 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/PsiOSConsciousnessSubstrate.tsx`

### test-filesystem-temporal-instance.ts (115 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/test-filesystem-temporal-instance.ts`

### UltraSimpleField.tsx (115 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/passiveworks/UltraSimpleField.tsx`

### symbiosis.ts (114 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/api/symbiosis.ts`

### useVortexRouter.ts (114 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/hooks/useVortexRouter.ts`

### index.ts (113 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/quantum/index.ts`

### dashboard.d.ts (113 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/modules/dashboard.d.ts`
- Classes/Interfaces: WiltonOSDashboard

### new-job.tsx (113 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/nexus/new-job.tsx`

### coherence-integration.ts (112 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/integration/coherence-integration.ts`

### usePhiMetrics.ts (111 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/hooks/usePhiMetrics.ts`

### ConnectionStatus.tsx (110 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/public/dashboard/src/components/ConnectionStatus.tsx`

### job-queue.tsx (110 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/dashboard/job-queue.tsx`

### DomainContext.tsx (110 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/contexts/DomainContext.tsx`

### WombKernel.ts (109 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/WombKernel.ts`
- Classes/Interfaces: WombKernel

### system-health.tsx (108 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/dashboard/system-health.tsx`

### process-attached-chats.ts (107 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/process-attached-chats.ts`

### CoherenceWaveform.tsx (107 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/quantum/CoherenceWaveform.tsx`

### chronos-date-handler.ts (106 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/utils/chronos-date-handler.ts`

### agent-manager.d.ts (106 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/server/agents/agent-manager.d.ts`
- Classes/Interfaces: AgentManager

### pipeline-visualization.tsx (105 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/dashboard/pipeline-visualization.tsx`

### test-filesystem-qrn-store.ts (104 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/test-filesystem-qrn-store.ts`

### QuantumConsciousnessShell.tsx (104 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/components/QuantumConsciousnessShell.tsx`

### QuantumConsciousnessShell.tsx (104 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/components/QuantumConsciousnessShell.tsx`

### coherence-dashboard.tsx (104 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/pages/coherence-dashboard.tsx`

### ModuleErrorBoundary.tsx (104 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ModuleErrorBoundary.tsx`
- Classes/Interfaces: ModuleErrorBoundary

### process-chat-histories.ts (103 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/scripts/process-chat-histories.ts`

### consciousness.ts (102 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/types/consciousness.ts`

### consciousness.ts (102 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/types/consciousness.ts`

### types.ts (101 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/types.ts`

### divine-absurdity-engine.ts (101 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/shared/divine-absurdity-engine.ts`
- Classes/Interfaces: DivineAbsurdityEngine

### nexus-routes.ts (101 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/routes/nexus-routes.ts`

### promise-rejection-suppressor.ts (101 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/lib/promise-rejection-suppressor.ts`
- Classes/Interfaces: PromiseRejectionSuppressor

### SafeModuleLoader.ts (101 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/core/SafeModuleLoader.ts`

### field-state.ts (100 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/core/field-state.ts`
- Classes/Interfaces: QuantumFieldState

### field-state.ts (100 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/core/field-state.ts`
- Classes/Interfaces: QuantumFieldState

### juliana-agent.d.ts (100 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/agents/juliana-agent.d.ts`
- Classes/Interfaces: JulianaAgent

### field.d.ts (96 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/core/consciousness/field.d.ts`
- Classes/Interfaces: ConsciousnessFieldEngine

### coherence.d.ts (96 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/modules/coherence.d.ts`
- Classes/Interfaces: CoherenceEngine

### state-bus.d.ts (94 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/shared/state-bus.d.ts`
- Classes/Interfaces: StateEventBus

### eventBus.d.ts (86 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/core/eventBus.d.ts`
- Classes/Interfaces: ConsciousnessState, PortalRouter, MusicConsciousnessInterface

### meta-void-navigation.d.ts (79 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/metaVoid/meta-void-navigation.d.ts`
- Classes/Interfaces: MetaVoidNavigator

### minimal-protocol.d.ts (79 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/quantumExperiment/minimal-protocol.d.ts`
- Classes/Interfaces: MinimalProtocol

### symbiosis-core.d.ts (79 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/core/symbiosis-core.d.ts`
- Classes/Interfaces: SybiosisCore

### collapse.ts (75 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/sim/collapse.ts`
- Classes/Interfaces: CollapseIntegrator

### collapse.ts (75 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/sim/collapse.ts`
- Classes/Interfaces: CollapseIntegrator

### chsh-logger.d.ts (74 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/export/chsh-logger.d.ts`
- Classes/Interfaces: CHSHLogger

### Tesseract4D.d.ts (72 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/quantum/Tesseract4D.d.ts`
- Classes/Interfaces: Tesseract4D

### index.d.ts (71 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/externalBroadcastLayer/index.d.ts`
- Classes/Interfaces: ExternalBroadcastLayer

### BCIModule.d.ts (69 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/quantum/BCIModule.d.ts`
- Classes/Interfaces: BCIModule

### sacred-clip-generator.d.ts (69 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/sacredClipGenerator/sacred-clip-generator.d.ts`
- Classes/Interfaces: SacredClipGenerator

### index.d.ts (69 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/index.d.ts`
- Classes/Interfaces: InternalCoherenceBundle

### π-sync.d.ts (69 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/core/π-sync.d.ts`
- Classes/Interfaces: πSyncProtocol

### QuantumFieldEngine.d.ts (68 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/quantum/QuantumFieldEngine.d.ts`
- Classes/Interfaces: QuantumFieldEngine

### musicPlayer.d.ts (63 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/modules/musicPlayer.d.ts`
- Classes/Interfaces: QuantumMusicPlayer

### quantum-playlist-engine.d.ts (62 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/broadcastRitual/quantum-playlist-engine.d.ts`
- Classes/Interfaces: QuantumPlaylistEngine

### consciousness-bell-test.d.ts (62 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/quantumExperiment/consciousness-bell-test.d.ts`
- Classes/Interfaces: ConsciousnessBellTest

### DateTransformer.d.ts (58 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/utils/DateTransformer.d.ts`
- Classes/Interfaces: DateTransformer

### missing-modules.d.ts (58 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/sacredGeometry/missing-modules.d.ts`
- Classes/Interfaces: SriYantraQuantumModule, MetatronsQuantumModule, FibonacciQuantumModule, MerkabaQuantumModule, FlowerOfLifeQuantumModule, TorusQuantumModule, PlatonicSolidsQuantumModule

### ui.d.ts (58 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/modules/ui.d.ts`
- Classes/Interfaces: UIController

### morphing.d.ts (57 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/morphing.d.ts`
- Classes/Interfaces: MorphingEngine, ConsciousnessResponsiveMorpher

### coachOverlay.d.ts (56 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/coachOverlay.d.ts`
- Classes/Interfaces: CoherenceCoach

### enhanced-production-geometry.d.ts (56 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/enhanced-production-geometry.d.ts`
- Classes/Interfaces: EnhancedSacredGeometryOverlay

### file-system-task-store.d.ts (56 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/server/services/task/file-system-task-store.d.ts`
- Classes/Interfaces: FileSystemTaskStore

### coherence-coach.d.ts (56 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/training/coherence-coach.d.ts`
- Classes/Interfaces: CoherenceCoach

### enhanced-production-coach.d.ts (55 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/enhanced-production-coach.d.ts`
- Classes/Interfaces: EnhancedCoherenceCoach

### BellTester.d.ts (54 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/quantum/BellTester.d.ts`
- Classes/Interfaces: BellTester

### geometryOverlay.d.ts (54 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/geometryOverlay.d.ts`
- Classes/Interfaces: SacredGeometryOverlay

### consciousness-activator.ts (53 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/WiltonOS_LightKernel_Migration/src/engine/consciousness-activator.ts`
- Classes/Interfaces: ConsciousnessActivator

### consciousness-activator.ts (53 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/src/engine/consciousness-activator.ts`
- Classes/Interfaces: ConsciousnessActivator

### color-wheel-badge.tsx (52 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ui/color-wheel-badge.tsx`
- Classes/Interfaces: based

### ErrorBoundary.tsx (51 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/client/src/components/ErrorBoundary.tsx`
- Classes/Interfaces: ErrorBoundary

### dashboard.d.ts (50 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/dashboard.d.ts`
- Classes/Interfaces: WiltonOSDashboard

### unified-event-bus.d.ts (49 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/consciousnessSync/unified-event-bus.d.ts`
- Classes/Interfaces: UnifiedConsciousnessEventBus

### sri-yantra.d.ts (49 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/geometry/sri-yantra.d.ts`
- Classes/Interfaces: SriYantraRenderer

### coherence.d.ts (49 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/core/coherence.d.ts`
- Classes/Interfaces: CoherenceProcessor

### module-router.d.ts (48 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/architectureUnification/module-router.d.ts`
- Classes/Interfaces: UnifiedModuleRouter

### enhanced-production-index.d.ts (48 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/enhanced-production-index.d.ts`
- Classes/Interfaces: EnhancedCoherenceBundle

### production-geometry.d.ts (48 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/production-geometry.d.ts`
- Classes/Interfaces: SacredGeometryOverlay

### auto-recorder.d.ts (46 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/sacredClipGenerator/auto-recorder.d.ts`
- Classes/Interfaces: SacredClipGenerator

### enhanced-geometry.d.ts (46 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/enhanced-geometry.d.ts`
- Classes/Interfaces: EnhancedSacredGeometryOverlay

### production-coach.d.ts (45 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/production-coach.d.ts`
- Classes/Interfaces: CoherenceCoach

### glifo-router.d.ts (45 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/engine/glifo-router.d.ts`
- Classes/Interfaces: GlifoRouter

### temporal-breathlock.d.ts (43 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/loop/temporal-breathlock.d.ts`
- Classes/Interfaces: TemporalBreathlock

### tesseract.d.ts (43 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/viz/tesseract.d.ts`
- Classes/Interfaces: TesseractRenderer

### capture.d.ts (43 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/modules/capture.d.ts`
- Classes/Interfaces: CaptureEngine

### unified-integration.d.ts (42 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/sacredGeometry/unified-integration.d.ts`
- Classes/Interfaces: UnifiedSacredGeometryIntegration

### production-morphing.d.ts (41 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/production-morphing.d.ts`
- Classes/Interfaces: MorphingEngine

### tesseract-renderer.d.ts (40 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/externalBroadcastLayer/tesseract-renderer.d.ts`
- Classes/Interfaces: TesseractRenderer

### enhanced-coach.d.ts (40 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/enhanced-coach.d.ts`
- Classes/Interfaces: EnhancedCoherenceCoach

### enhanced-geometry-overlay.d.ts (39 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/enhanced-geometry-overlay.d.ts`
- Classes/Interfaces: EnhancedGeometryOverlay

### recorder.d.ts (39 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/core/recorder.d.ts`
- Classes/Interfaces: MediaRecorderEngine

### psi-broadcast-module.d.ts (38 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/freeAsFuck/psi-broadcast-module.d.ts`
- Classes/Interfaces: PsiBroadcastModule

### production-index.d.ts (37 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/production-index.d.ts`
- Classes/Interfaces: ProductionCoherenceBundle

### spiral.d.ts (37 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/modules/spiral.d.ts`
- Classes/Interfaces: SpiralRenderer

### routes.d.ts (37 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/modules/routes.d.ts`
- Classes/Interfaces: RouteManager

### orchestrator.d.ts (35 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/orchestrator.d.ts`
- Classes/Interfaces: MetaCoherenceOrchestrator

### chsh-logger.d.ts (34 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/externalBroadcastLayer/chsh-logger.d.ts`
- Classes/Interfaces: CHSHLogger

### quantum-validator.d.ts (34 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/externalBroadcastLayer/quantum-validator.d.ts`
- Classes/Interfaces: QuantumValidator

### vesica-piscis-module.d.ts (34 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/sacredGeometry/vesica-piscis-module.d.ts`
- Classes/Interfaces: VesicaPiscisQuantumModule

### capture.d.ts (34 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/viz/capture.d.ts`
- Classes/Interfaces: Capture

### broadcast-hud.d.ts (33 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/externalBroadcastLayer/broadcast-hud.d.ts`
- Classes/Interfaces: BroadcastHUD

### field-state.d.ts (33 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/core/field-state.d.ts`
- Classes/Interfaces: QuantumFieldState

### eegSimulator.d.ts (33 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/core/eegSimulator.d.ts`
- Classes/Interfaces: EEGSimulator

### utils.d.ts (32 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/utils.d.ts`
- Classes/Interfaces: RateLimiter, PerformanceMonitor, SacredMath

### coaching-panel.d.ts (32 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/ui/coaching-panel.d.ts`
- Classes/Interfaces: CoachingPanel

### embodied-voice.d.ts (28 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/agent/embodied-voice.d.ts`
- Classes/Interfaces: EmbodiedVoice

### critical-fixes-report.d.ts (28 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/systemAnalysis/critical-fixes-report.d.ts`
- Classes/Interfaces: CriticalFixesAnalyzer

### feedZLambda.d.ts (28 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/eeg/feedZLambda.d.ts`
- Classes/Interfaces: ConsciousnessFeed

### eegOverlay.d.ts (28 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/modules/eegOverlay.d.ts`
- Classes/Interfaces: EEGOverlay

### enhanced-index.d.ts (26 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/enhanced-index.d.ts`
- Classes/Interfaces: EnhancedCoherenceBundle

### StorageOperationError.ts (25 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/storage/errors/StorageOperationError.ts`
- Classes/Interfaces: StorageOperationError

### FileNotFoundError.ts (25 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/storage/errors/FileNotFoundError.ts`
- Classes/Interfaces: FileNotFoundError

### chsh.d.ts (25 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/sim/chsh.d.ts`
- Classes/Interfaces: CHSHHarness

### InvalidDataError.ts (19 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/server/services/storage/errors/InvalidDataError.ts`
- Classes/Interfaces: InvalidDataError

### collapse.d.ts (19 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/sim/collapse.d.ts`
- Classes/Interfaces: CollapseIntegrator

### dashboard-controls.d.ts (17 lines) [low]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/externalBroadcastLayer/dashboard-controls.d.ts`
- Classes/Interfaces: BroadcastDashboardControls

### ui-integration.d.ts (16 lines) [medium]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/extensions/internalCoherenceBundle/ui-integration.d.ts`
- Classes/Interfaces: CoherenceBundleUI

### consciousness-activator.d.ts (14 lines) [high]
- Path: `/home/zews/rag-local/WiltonOS-PassiveWorks/dist/engine/consciousness-activator.d.ts`
- Classes/Interfaces: ConsciousnessActivator