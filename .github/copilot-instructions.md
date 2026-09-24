# Copilot Instructions

## Commands

### Build
- `cd frontend && npm run build` — builds the Vue SPA into `frontend_dist\`, which is what the PyQt app loads.

### Run
- `new_run_main_ui.bat` — creates/uses `venv\` (bootstraps pip) and starts the app.
- `venv\Scripts\python.exe new_main_ui.py`
- Missing packages from `requirements.txt` are installed at app startup by `core\check_requirements.py` (via `core\app_startup.py`), whichever entry point starts the app — the launchers themselves install nothing.
- `cd frontend && npm run dev` — useful for working on the Vue app in isolation; `frontend\src\bridge.js` falls back to a mock backend when QWebChannel is unavailable.

### Tests and type checks
Always use the repository venv for Python — the first `python` on PATH here is a 3.10 without pandas/PyQt6 and produces dozens of fake ImportErrors.
- `venv\Scripts\python.exe run_tests.py` — full backend suite (standard `unittest`, `tests\test_*.py`; no pytest). `--quick` skips the slow integration modules listed in `run_tests.SLOW_MODULES` and the tests marked `@requires_torch` (`--with-torch` keeps them); `--durations 15` prints per-module timings. If started with a non-venv interpreter it re-executes itself with the venv.
- Never `import torch` at the top of a test module: mark torch tests with `tests\_optional_deps.requires_torch` and import torch lazily via `load_torch()` / `bind_torch(globals())` (enforced by `tests\test_optional_deps.py`).
- `venv\Scripts\python.exe -m py_compile <files>` — syntax check for the Python files you touched.
- `cd frontend && npm run test` — vitest (`src/**/*.test.ts`, pure frontend logic).
- `cd frontend && npm run test:node` — node:test suites in `src/studio/*.test.mjs` (outside vitest).
- `cd frontend && npm run type-check` — `vue-tsc --noEmit`; the baseline is **0 errors** (includes the type-level contract `src/types/bridgePayloads.test-d.ts`).
- Bridge contract guards: `tests\test_bridge_contract.py` (action/event/slot/signal names, both directions) and `tests\test_web_mode_security.py` (web-mode whitelists in `web_main_ui.py`).
- There is no ESLint/flake8 configuration.

## Architecture

- This is a **Windows-first PyQt6 desktop app with a Vue 3 SPA frontend**. `new_main_ui.py` starts `GeneratorMainUI`, which is assembled from multiple mixins in `ui\generator_main.py`.
- `ui\generator_ui_setup.py` creates a `QStackedWidget` with three workspaces:
  - index 0: the main Vue SPA in a `QWebEngineView`
  - index 1: a native `BrowserTab`
  - index 2: a native `BackendUITab`
- The main frontend entry is the built file `frontend_dist\index.html`, not the Vite dev server. Vue routing uses `createMemoryHistory()` in `frontend\src\router.js`.
- **Python↔Vue communication is centered on QWebChannel**:
  - `ui\vue_bridge.py` exposes signals, widget synchronization hooks, and callable slots.
  - `frontend\src\bridge.js` initializes the QWebChannel backend object.
  - `frontend\src\stores\widgetStore.js` maintains the shared widget state and synchronizes changes in both directions.
- **Most legacy Python business logic still runs unchanged** through proxy widgets. `ui\generator_ui_setup.py` creates `LineEditProxy`, `TextEditProxy`, `ComboBoxProxy`, `CheckBoxProxy`, etc. from `ui\widget_proxies.py`, so older mixins like `ui\generator_settings.py` and `ui\generator_prompts.py` can keep reading and writing widget-like objects.
- Vue user actions are not handled by direct HTTP calls. Components call `requestAction(...)`, which reaches `VueBridge.onAction()` and is dispatched by `GeneratorMainUI._handle_vue_action()`. That action handler is the main integration point for tab switches, image transfer, editor commands, queue actions, search/event workflows, and generation requests.
- Persistent UI/application state is split across JSON config files such as `prompt_settings.json`, `config\ui_prefs.json`, and `config\tab_defaults.json`. Search/event data also depends on local parquet datasets configured in `config.py`.

## Key conventions

- **Do not wire new UI state directly to real PyQt widgets.** Add or reuse a proxy in `ui\generator_ui_setup.py`, then bind the same widget ID in Vue through `widgetStore`.
- **Widget IDs must match exactly across Python and Vue.** Examples: `widgets.character_input` ↔ `LineEditProxy(b, 'character_input')`, `widgets.model_combo` ↔ `ComboBoxProxy(b, 'model_combo')`.
- Use the right integration path for each kind of change:
  - **field/state sync**: add or update a proxy widget
  - **button/action behavior**: send `requestAction(...)` from Vue and handle it in `_handle_vue_action()`
  - **backend event to Vue**: add a signal in `ui\vue_bridge.py` and subscribe with `onBackendEvent(...)`
- When sending an image to another tab, preserve the existing sequencing: emit `tabChanged` first, then emit the image-load signal after the 100 ms delay. Several flows rely on that render timing.
- UI styling now lives primarily in `frontend\src\`. PyQt QSS/theme changes usually do not affect the visible app because the main workspace is rendered by Vue inside `QWebEngineView`.
- Use `core\error_handler.py` error codes/toast integration for user-facing backend failures instead of adding silent fallbacks.
- Prompt exclusion rules are a repository-specific feature with multiple prefix/suffix/exception forms. If you touch exclude filtering, keep the Vue help text in `frontend\src\components\PromptPanel.vue` and the parsing logic in `ui\generator_prompts.py` aligned.
- Some SPA preferences intentionally live in browser storage (`tabOrder`, `tagBlockMode`, `ratingFilter`, `loraStack`) and are also mirrored back to Python through actions such as `save_ui_prefs`. Check both sides before changing persistence behavior.
