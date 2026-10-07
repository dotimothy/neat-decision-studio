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
#   ./run.sh --llm URL           an OpenAI-compatible chat server the games can set Laya against,
#                                or "none" (default: NEAT GenAI Studio's, http://127.0.0.1:9998)
#   ./run.sh --open-browser      also open the demo in this machine's browser once it is up; if
#                                the app is already running, only open the browser
#   ./run.sh --cli               the demo in this terminal instead of a browser: a prompt that
#                                answers questions and manages models (/help inside). It uses
#                                the running app, or starts one for as long as the prompt is open.
#                                --model NAME loads a model first; --ask "QUESTION" answers
#                                one yes-or-no question and exits; --json prints raw answers
#   ./run.sh --no-games          leave the game model out altogether
#   ./run.sh --stop              stop a running app
#   ./run.sh update              update to the latest published version and rebuild the runtime;
#                                models, the log and the reset token are kept (see below)
#   ./run.sh --version           the Neat logo and the versions: firmware, Neat, neat-llima,
#                                neat-runtime and Neat Decision Studio itself
#   ./run.sh --reset-mla         reset the board's MLA runtime first (see below)
#   ./run.sh --reset-token       the token the Models page asks for when the accelerator is
#                                reset from another machine's browser
#   ./run.sh runtime <args...>   the C++ runtime's own command line instead, e.g.
#                                ./run.sh runtime bench model --iters 100
#
# update pulls with git in a clone of the repository; anywhere else it downloads the published
# source and puts it in place of this directory's. It never goes back to an older version than
# the one installed (UPDATE_FORCE=1 does), and it rebuilds the runtime unless UPDATE_BUILD=0.
# DECISION_STUDIO_BRANCH names another branch than main. A running app is left running: restart
# it to use what was installed.
#
# --reset-mla restarts the board-wide MLA services, which unloads every model on the MLA,
# including other applications'. It never happens unless asked for. Use it when the model
# fails to load with MLA_LOAD_FAILED: the MLA memory pool is then full or fragmented.
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${LAYA_MODEL_DIR:-${APP_DIR}/model}"
GAME_MODEL_DIR="${LAYA_GAME_MODEL_DIR:-${APP_DIR}/model-dino}"
PORT="${LAYA_PORT:-8095}"
export MLA_SUDO_PASSWORD="${MLA_SUDO_PASSWORD:-edgeai}"   # the DevKit image's stock password; the app's
                                                          # own Reset Accelerator button needs it too
seq_lens="${LAYA_SEQ_LENS:-all}"
reset=0
games=1
preload="${LAYA_PRELOAD:-general}"
hub="${LAYA_HUB:-TDoSiMa/sima-laya}"
llm="${LAYA_LLM:-http://127.0.0.1:9998}"

cli=0              # the terminal client (webapp/cli.py) instead of a browser
cli_args=()
preload_set=0
open_browser=0
desktop=0          # started from the desktop icon (setup.sh installs it): --open-browser, and
                   # the terminal window stays open on a failure so the message can be read

# The Neat sparkle and what is installed, shown when the app or its prompt is started in a
# terminal, as the other Neat demos do. Plain text when not a terminal, or NO_COLOR is set.
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  P_TEAL=$'\033[38;2;61;179;138m'; P_GREEN=$'\033[38;2;74;168;54m'; P_LIME=$'\033[38;2;154;190;30m'
  P_BLUE=$'\033[38;2;58;125;216m'; P_ORANGE=$'\033[38;2;223;108;30m'; P_INK=$'\033[38;2;60;66;74m'
  C_ACCENT=$'\033[38;2;47;212;192m'; C_MUTED=$'\033[38;2;140;150;160m'
else
  C_RESET=''; C_BOLD=''; C_DIM=''; P_TEAL=''; P_GREEN=''; P_LIME=''; P_BLUE=''; P_ORANGE=''; P_INK=''; C_ACCENT=''; C_MUTED=''
fi
banner() {
  printf '\n'
  printf '        %s▲%s\n'           "${P_TEAL}" "${C_RESET}"
  printf '       %s███%s\n'          "${P_GREEN}" "${C_RESET}"
  printf '      %s█████%s\n'         "${P_GREEN}" "${C_RESET}"
  printf '   %s◀████%s█%s████▶%s\n'  "${P_BLUE}" "${P_INK}" "${P_LIME}" "${C_RESET}"
  printf '      %s█████%s\n'         "${P_ORANGE}" "${C_RESET}"
  printf '       %s███%s\n'          "${P_ORANGE}" "${C_RESET}"
  printf '        %s▼%s\n'           "${P_ORANGE}" "${C_RESET}"
  printf '\n'
  printf '   %sNEAT%s %sDecision Studio%s\n' "${C_BOLD}" "${C_RESET}" "${C_ACCENT}${C_BOLD}" "${C_RESET}"
  printf '   %sSystem-1 decision models on your SiMa.ai Modalix MLSoC%s\n\n' "${C_MUTED}" "${C_RESET}"
}
# This app's version: the number in VERSION and, in a clone, the branch and commit it is at
# (bin/laya-deploy stamps the same into the copy it makes).
laya_version() {
  local base; base="$(cat "${APP_DIR}/VERSION" 2> /dev/null || echo unknown)"
  if [[ "$base" != *+* ]] && git -C "${APP_DIR}" rev-parse --short=12 HEAD > /dev/null 2>&1; then
    base+="+$(git -C "${APP_DIR}" rev-parse --abbrev-ref HEAD).$(git -C "${APP_DIR}" rev-parse --short=12 HEAD)"
  fi
  printf '%s' "$base"
}
# Versions come from the installed packages and the image's build file, which answer at once;
# the `neat` command would say the same after an online update check of several seconds.
system_info() {
  local kv='   %s%-13s%s %s\n' package
  version_of() { dpkg-query -W -f '${Version}' "$1" 2> /dev/null || true; }
  local firmware; firmware="$(grep -m1 -oE '[0-9]+\.[0-9]+\.[0-9]+_[A-Za-z0-9_]+' /etc/buildinfo 2> /dev/null || true)"
  [[ -n "$firmware" ]] && printf "$kv" "${C_MUTED}" "firmware" "${C_RESET}" "$firmware"
  printf "$kv" "${C_MUTED}" "neat" "${C_RESET}" "$(version_of sima-neat | grep . || echo 'not installed')"
  printf "$kv" "${C_MUTED}" "neat-llima" "${C_RESET}" "$(version_of sima-lmm-dev | grep . || echo 'not installed')"
  printf "$kv" "${C_MUTED}" "neat-runtime" "${C_RESET}" "$(version_of neat-runtime | grep . || echo 'not installed')"
  printf "$kv" "${C_MUTED}" "neat-decision" "${C_RESET}" "$(laya_version)"
  printf "$kv" "${C_MUTED}" "python" "${C_RESET}" "$(python3 --version 2>&1 | cut -d' ' -f2)"
  printf "$kv" "${C_MUTED}" "host" "${C_RESET}" "$(uname -sm 2> /dev/null || echo unknown)"
  printf '\n'
}

fail() {
  echo "run.sh: $*" >&2
  [[ $desktop -eq 1 ]] && { read -r -p "Press Enter to close this window... " _ || true; }
  exit 1
}

# Whether the app answers on a URL, and opening one in the desktop's browser.
answers() {
  python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1] + "/api/info", timeout=2)' "$1" 2> /dev/null
}
open_url() {
  local opener
  for opener in xdg-open x-www-browser sensible-browser; do
    if command -v "$opener" > /dev/null 2>&1; then
      (setsid "$opener" "$1" > /dev/null 2>&1 &)
      return 0
    fi
  done
  echo "run.sh: no browser opener (xdg-open) found; open $1 yourself" >&2
}

# ------------------------------------------------------------------------------------ update
u_step() { printf '\n%s\n' "${C_ACCENT}${C_BOLD}▸${C_RESET} $*"; }
u_info() { printf '%s\n' "${C_MUTED}·${C_RESET} $*"; }
u_ok()   { printf '%s\n' "${P_TEAL}✔${C_RESET} $*"; }
u_warn() { printf '%s\n' "${P_ORANGE}⚠${C_RESET} $*" >&2; }
u_err()  { printf '%s\n' "${P_ORANGE}✘${C_RESET} $*" >&2; }

# Whether version $1 is older than version $2 (as `sort -V` orders them). A version with a
# "+branch.commit" tail is compared without it.
older_than() {
  local a="${1%%+*}" b="${2%%+*}"
  [[ "$a" != "$b" && "$(printf '%s\n%s\n' "$a" "$b" | sort -V | head -n 1)" == "$a" ]]
}

do_update() {
  local branch="${DECISION_STUDIO_BRANCH:-main}" before after running=0
  before="$(cat "${APP_DIR}/VERSION" 2> /dev/null || echo unknown)"
  banner
  pgrep -f "python3 ${APP_DIR}/webapp/server[.]py" > /dev/null 2>&1 && running=1
  u_step "Updating Neat Decision Studio ${C_DIM}(installed: ${before})${C_RESET}"

  if command -v git > /dev/null 2>&1 \
       && git -C "${APP_DIR}" ls-files --error-unmatch run.sh > /dev/null 2>&1; then
    # A clone of the repository: pulling is the cleanest update.
    u_info "This is a git checkout: pulling ${C_BOLD}${branch}${C_RESET}"
    if ! git -C "${APP_DIR}" pull --ff-only origin "${branch}"; then
      u_err "git pull failed (local changes, or a branch that has diverged). Resolve it by hand."
      return 1
    fi
  else
    # A copy (bin/laya-deploy, or an unpacked download): fetch the published source and put it
    # in place of this one.
    local tool repo repo_url archive tmp root published
    for tool in tar rsync; do
      command -v "$tool" > /dev/null 2>&1 || { u_err "$tool is needed to update a copy of the app."; return 1; }
    done
    repo_url="${DECISION_STUDIO_REPO_URL:-https://github.com/dotimothy/neat-decision-studio.git}"
    repo="${repo_url%.git}"; repo="${repo#https://github.com/}"; repo="${repo#git@github.com:}"
    archive="${DECISION_STUDIO_ARCHIVE_URL:-https://github.com/${repo}/archive/refs/heads/${branch}.tar.gz}"
    tmp="$(mktemp -d)" || { u_err "mktemp failed."; return 1; }
    u_info "Fetching ${C_DIM}${archive}${C_RESET}"
    if [[ -f "${archive}" ]]; then
      cp "${archive}" "${tmp}/src.tar.gz" || { u_err "Cannot read ${archive}."; rm -rf "${tmp}"; return 1; }
    elif command -v curl > /dev/null 2>&1; then
      curl -fsSL "${archive}" -o "${tmp}/src.tar.gz" || { u_err "The download failed. Is the board online?"; rm -rf "${tmp}"; return 1; }
    elif command -v wget > /dev/null 2>&1; then
      wget -qO "${tmp}/src.tar.gz" "${archive}" || { u_err "The download failed. Is the board online?"; rm -rf "${tmp}"; return 1; }
    else
      u_err "curl or wget is needed to update."; rm -rf "${tmp}"; return 1
    fi
    # The archive's one top directory. Not `| head -n 1`: tar would be cut short, and with
    # pipefail that ends the script.
    root="$(tar -tzf "${tmp}/src.tar.gz" 2> /dev/null || true)"; root="${root%%/*}"
    if [[ -z "$root" ]] || ! tar -xzf "${tmp}/src.tar.gz" -C "${tmp}" 2> /dev/null \
         || [[ ! -f "${tmp}/${root}/run.sh" || ! -f "${tmp}/${root}/webapp/server.py" ]]; then
      u_err "What was downloaded is not this app's source."; rm -rf "${tmp}"; return 1
    fi
    # Never backwards: a copy made from a development host may be ahead of what is published.
    published="$(cat "${tmp}/${root}/VERSION" 2> /dev/null || true)"
    if [[ "${UPDATE_FORCE:-0}" != "1" ]] && { [[ -z "$published" ]] || older_than "$published" "$before"; }; then
      u_ok "Nothing to do: what is published (${published:-no version}) is not newer than what is installed (${before})."
      u_info "UPDATE_FORCE=1 ./run.sh update installs it all the same."
      rm -rf "${tmp}"; return 0
    fi
    u_info "Putting ${published} in place ${C_DIM}(keeping the models, the log and the reset token)${C_RESET}"
    # A mirror of the source, so that files a release dropped go here too; everything that
    # belongs to this installation and not to the source is left out of it.
    rsync -a --delete-delay --delay-updates \
      --exclude='/model/' --exclude='/model-*/' --exclude='/models/' --exclude='/build/' \
      --exclude='/third_party/' --exclude='/runtime/build/' --exclude='/laya' \
      --exclude='/.reset-token' --exclude='*.log' --exclude='/.git/' --exclude='/.venv*/' \
      --exclude='__pycache__/' \
      "${tmp}/${root}/" "${APP_DIR}/" \
      || { u_err "The source could not be put in place."; rm -rf "${tmp}"; return 1; }
    rm -rf "${tmp}"
  fi
  chmod +x "${APP_DIR}/run.sh" "${APP_DIR}/setup.sh" 2> /dev/null || true
  after="$(cat "${APP_DIR}/VERSION" 2> /dev/null || echo unknown)"
  u_ok "Source updated: ${before} → ${C_BOLD}${after}${C_RESET}"

  # The runtime is C++ and has to match the source it came with.
  if [[ "${UPDATE_BUILD:-1}" == "0" ]]; then
    u_info "Runtime not rebuilt (UPDATE_BUILD=0); ./setup.sh does it."
  elif ! "${APP_DIR}/setup.sh" --runtime-only; then
    u_err "The runtime did not build; the app keeps the one it had. ./setup.sh shows why."
    return 1
  fi
  u_ok "Update complete."
  [[ $running -eq 1 ]] && u_warn "The app is running the version it was started with: restart it (./run.sh --stop, then ./run.sh)."
  return 0
}

# Before anything that needs the runtime: an update is also how a broken build gets mended.
case "${1:-}" in
  update|--update|upgrade) do_update; exit $? ;;
  -h|--help) sed -n '2,43p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

[[ -x "${APP_DIR}/laya" ]] || fail "the runtime is not built yet; run ./setup.sh first"

# `cli` followed by arguments is the older spelling of `runtime`; alone it is `--cli`.
if [[ "${1:-}" == "runtime" || ( "${1:-}" == "cli" && $# -gt 1 ) ]]; then
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
    --preload) preload="${2:?--preload needs a value}"; preload_set=1; shift 2 ;;
    --hub) hub="${2:?--hub needs a value}"; shift 2 ;;
    --llm) llm="${2:?--llm needs a URL, or none}"; shift 2 ;;
    --open-browser) open_browser=1; shift ;;
    --cli|cli) cli=1; shift ;;
    --model) cli=1; cli_args+=(--model "${2:?--model needs a name}"); shift 2 ;;
    --ask) cli=1; cli_args+=(--ask "${2:?--ask needs a question}"); shift 2 ;;
    --budget) cli=1; cli_args+=(--budget "${2:?--budget needs a number}"); shift 2 ;;
    --json) cli=1; cli_args+=(--json); shift ;;
    --desktop) open_browser=1; desktop=1; shift ;;
    --version) banner; system_info; exit 0 ;;
    --reset-token)
      # What the Models page asks for before it resets the accelerator from another machine.
      cat "${APP_DIR}/.reset-token" 2> /dev/null || echo "no token yet: it is made the first time the app starts"
      exit 0 ;;
    --stop)
      # SIGTERM lets the server close the runtimes, which releases the models on the MLA.
      pids="$(pgrep -f 'python3 .*webapp/server[.]py' || true)"
      [[ -n "$pids" ]] && kill -TERM $pids && echo "stopped" || echo "not running"
      exit 0 ;;
    -h|--help) sed -n '2,43p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) fail "unknown option: $1 (see --help)" ;;
  esac
done

# With no compiled model on the board the app still starts: its Models page downloads them.
[[ -f "${MODEL_DIR}/laya_config.json" || $cli -eq 1 ]] \
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

# In a terminal, a start opens with the logo and the versions; a one-line answer does not.
one_shot=0
for arg in "${cli_args[@]}"; do [[ "$arg" == "--ask" ]] && one_shot=1; done
if [[ -t 1 && $one_shot -eq 0 && ( $cli -eq 0 || -t 0 ) ]]; then
  banner; system_info
  cli_args+=(--no-banner)             # the prompt would otherwise draw the logo again
fi

url="http://localhost:${PORT}"
if [[ $cli -eq 1 ]] && answers "$url"; then
  # The app is already up: the prompt joins it, and leaves it running.
  exec python3 "${APP_DIR}/webapp/cli.py" --url "$url" "${cli_args[@]}"
fi
if ss -ltn 2> /dev/null | grep -q ":${PORT} "; then
  if [[ $open_browser -eq 1 ]] && answers "$url"; then
    echo "The app is already running; opening ${url}"
    open_url "$url"
    exit 0
  fi
  fail "port ${PORT} is already in use. Is the app already running? (./run.sh --stop)
  Or pick another port: ./run.sh --port 9000"
fi

address="$(hostname -I 2> /dev/null | awk '{print $1}')"
[[ $cli -eq 1 ]] || echo "Starting; the demo will be at http://${address:-<board-ip>}:${PORT} (models: /models)"
# For the terminal prompt nothing is loaded up front unless asked for: the prompt shows a
# model loading, and lets one be chosen.
[[ $cli -eq 1 && $preload_set -eq 0 ]] && preload="none"
args=(--model "${MODEL_DIR}" --laya "${APP_DIR}/laya" --port "${PORT}" --preload "${preload}" --root "${APP_DIR}"
      --hub "${hub}" --llm "${llm}")
[[ -n "${seq_lens}" && "${seq_lens}" != "all" ]] && args+=(--seq-lens "${seq_lens}")
# Every other model-<name> directory is a question model the page can switch to.
for dir in "${APP_DIR}"/model-*/; do
  name="$(basename "$dir")"; name="${name#model-}"
  [[ "$name" == "dino" || ! ( -f "${dir}laya_config.json" || -f "${dir}clm_config.json" ) ]] && continue
  args+=(--extra-model "${name}=${dir%/}")
done
if [[ $games -eq 1 && -f "${GAME_MODEL_DIR}/laya_config.json" ]]; then
  args+=(--game-model "${GAME_MODEL_DIR}")
  [[ $cli -eq 1 ]] || echo "Games: http://${address:-<board-ip>}:${PORT}/games"
fi
if [[ $cli -eq 1 ]]; then
  # The prompt owns the app: started quietly here, stopped when the prompt is left.
  python3 "${APP_DIR}/webapp/server.py" "${args[@]}" > "${APP_DIR}/webapp.log" 2>&1 &
  server=$!
  trap 'kill -TERM "$server" 2> /dev/null; wait "$server" 2> /dev/null' EXIT
  until answers "$url"; do
    kill -0 "$server" 2> /dev/null || { tail -n 5 "${APP_DIR}/webapp.log" >&2; fail "the app did not start (log: ${APP_DIR}/webapp.log)"; }
    sleep 0.3
  done
  status=0
  python3 "${APP_DIR}/webapp/cli.py" --url "$url" "${cli_args[@]}" || status=$?
  exit $status
fi

waiter=""
if [[ $open_browser -eq 1 ]]; then
  # The browser opens once the app answers, which is after the preloaded model is on the MLA.
  ( for _ in $(seq 1 240); do
      if answers "$url"; then open_url "$url"; break; fi
      sleep 0.5
    done ) &
  waiter=$!
fi
status=0
python3 "${APP_DIR}/webapp/server.py" "${args[@]}" || status=$?
[[ -n "$waiter" ]] && kill "$waiter" 2> /dev/null || true
if [[ $status -ne 0 && $status -ne 130 ]]; then
  echo >&2
  echo "run.sh: the app stopped (exit ${status}). If the message above mentions MLA_LOAD_FAILED," >&2
  echo "  the MLA memory pool is full or fragmented: stop other MLA applications, or rerun with" >&2
  echo "  ./run.sh --reset-mla   (CmaFree now: $(awk '/CmaFree/ {print int($2 / 1024) " MB"}' /proc/meminfo))" >&2
  [[ $desktop -eq 1 ]] && { read -r -p "Press Enter to close this window... " _ || true; }
fi
exit $status
