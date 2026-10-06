#!/usr/bin/env bash
# Start the Laya decision playground on a Modalix DevKit.
#
#   ./run.sh                     web app on port 8095
#   ./run.sh --port 9000         another port
#   ./run.sh --seq-lens 128      which compiled sequence lengths the app offers
#                                (default: all of them)
#   ./run.sh --preload NAME      the model to put on the MLA at startup, or "none" (default:
#                                general). One model is loaded at a time; another is loaded
#                                in its place from the pages.
#   ./run.sh --hub REPO          the Hugging Face repository the Models page downloads compiled
#                                models from, or "none" (default: TDoSiMa/sima-laya)
#   ./run.sh --no-games          leave the game model out altogether
#   ./run.sh --stop              stop a running app
#   ./run.sh --reset-mla         reset the board's MLA runtime first (see below)
#   ./run.sh cli <args...>       the command-line runtime instead, e.g.
#                                ./run.sh cli bench model --iters 100
#
# --reset-mla restarts the board-wide MLA services, which unloads every model on the MLA,
# including other applications'. It never happens unless asked for. Use it when the model
# fails to load with MLA_LOAD_FAILED: the MLA memory pool is then full or fragmented.
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${LAYA_MODEL_DIR:-${APP_DIR}/model}"
GAME_MODEL_DIR="${LAYA_GAME_MODEL_DIR:-${APP_DIR}/model-dino}"
PORT="${LAYA_PORT:-8095}"
MLA_SUDO_PASSWORD="${MLA_SUDO_PASSWORD:-edgeai}"   # the DevKit image's stock password
seq_lens="${LAYA_SEQ_LENS:-all}"
reset=0
games=1
preload="${LAYA_PRELOAD:-general}"
hub="${LAYA_HUB:-TDoSiMa/sima-laya}"

fail() { echo "run.sh: $*" >&2; exit 1; }

[[ -x "${APP_DIR}/laya" ]] || fail "the runtime is not built yet; run ./setup.sh first"

if [[ "${1:-}" == "cli" ]]; then
  shift
  cd "${APP_DIR}"
  exec "${APP_DIR}/laya" "$@"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --seq-lens) seq_lens="${2:?--seq-lens needs a value}"; shift 2 ;;
    --reset-mla) reset=1; shift ;;
    --no-games) games=0; shift ;;
    --preload) preload="${2:?--preload needs a value}"; shift 2 ;;
    --hub) hub="${2:?--hub needs a value}"; shift 2 ;;
    --stop)
      # SIGTERM lets the server close the runtimes, which releases the models on the MLA.
      pids="$(pgrep -f 'python3 .*webapp/server[.]py' || true)"
      [[ -n "$pids" ]] && kill -TERM $pids && echo "stopped" || echo "not running"
      exit 0 ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) fail "unknown option: $1 (see --help)" ;;
  esac
done

# With no compiled model on the board the app still starts: its Models page downloads them.
[[ -f "${MODEL_DIR}/laya_config.json" ]] \
  || echo "No compiled model in ${MODEL_DIR} yet: download one on the Models page once the app is up."

mla_sudo() {
  if sudo -n true 2> /dev/null; then sudo -n "$@"; else echo "${MLA_SUDO_PASSWORD}" | sudo -S -p '' "$@"; fi
}

if [[ $reset -eq 1 ]]; then
  echo "Resetting the MLA runtime (this unloads every model on the MLA)"
  if [[ -x /usr/bin/fix_devkit_runtime.sh ]]; then
    mla_sudo /usr/bin/fix_devkit_runtime.sh | tail -n 3
  else
    mla_sudo systemctl restart simaai-appcomplex.service
  fi
  sleep 3
fi

# Loaded graphs live in the kernel's CMA pool, and so does page cache until something needs the
# room. Right after large files were copied to the board the pool can be full of it, and the
# load then fails. Dropping the page cache is harmless and frees it.
cma_free_mb="$(awk '/CmaFree/ {print int($2 / 1024)}' /proc/meminfo)"
if [[ "${cma_free_mb:-0}" -lt 1000 ]]; then
  sync
  mla_sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches' 2> /dev/null || true
fi

if ss -ltn 2> /dev/null | grep -q ":${PORT} "; then
  fail "port ${PORT} is already in use. Is the app already running? (./run.sh --stop)
  Or pick another port: ./run.sh --port 9000"
fi

address="$(hostname -I 2> /dev/null | awk '{print $1}')"
echo "Starting; the demo will be at http://${address:-<board-ip>}:${PORT} (models: /models)"
args=(--model "${MODEL_DIR}" --laya "${APP_DIR}/laya" --port "${PORT}" --preload "${preload}" --root "${APP_DIR}"
      --hub "${hub}")
[[ -n "${seq_lens}" && "${seq_lens}" != "all" ]] && args+=(--seq-lens "${seq_lens}")
# Every other model-<name> directory is a question model the page can switch to.
for dir in "${APP_DIR}"/model-*/; do
  name="$(basename "$dir")"; name="${name#model-}"
  [[ "$name" == "dino" || ! -f "${dir}laya_config.json" ]] && continue
  args+=(--extra-model "${name}=${dir%/}")
done
if [[ $games -eq 1 && -f "${GAME_MODEL_DIR}/laya_config.json" ]]; then
  args+=(--game-model "${GAME_MODEL_DIR}")
  echo "Games: http://${address:-<board-ip>}:${PORT}/games"
fi
status=0
python3 "${APP_DIR}/webapp/server.py" "${args[@]}" || status=$?
if [[ $status -ne 0 && $status -ne 130 ]]; then
  echo >&2
  echo "run.sh: the app stopped (exit ${status}). If the message above mentions MLA_LOAD_FAILED," >&2
  echo "  the MLA memory pool is full or fragmented: stop other MLA applications, or rerun with" >&2
  echo "  ./run.sh --reset-mla   (CmaFree now: $(awk '/CmaFree/ {print int($2 / 1024) " MB"}' /proc/meminfo))" >&2
fi
exit $status
