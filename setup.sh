#!/usr/bin/env bash
# One-time setup of Neat Decision Studio on a Modalix DevKit: check the board has what the
# runtime links against, build the runtime, see to the models, and add the `neat-decision` alias
# and a desktop icon.
#
#   ./setup.sh                    build and verify
#   ./setup.sh --check            also load the general model on the MLA and answer one question
#   ./setup.sh --runtime-only     check the prerequisites and build the runtime, nothing else
#                                 (what `./run.sh update` runs)
#   ./setup.sh --no-alias         leave the shell's startup file alone      (or CREATE_ALIAS=0)
#   ./setup.sh --no-desktop-icon  install no desktop icon            (or CREATE_DESKTOP_ICON=0)
#
# Environment:
#   LAYA_MODELS   compiled models to download from Hugging Face now, by name, e.g.
#                 "general dino", or "all". Unset: in a terminal, with no model on the board,
#                 setup offers the list; otherwise models are downloaded later, on the app's
#                 Models page or with `neat-decision --cli`.
#   LAYA_HUB      the Hugging Face repository they come from (default: TDoSiMa/sima-laya)
#
# `neat-decision` is an alias for run.sh (`neat-decision`, `neat-decision --cli`, `neat-decision --stop`). The
# desktop icon, installed when the board has a desktop, starts the app and opens it in the
# board's browser.
#
# Run it on the board, in a clone of the repository or in the directory `bin/laya-deploy`
# copied (default /media/nvme/laya).
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${LAYA_MODEL_DIR:-${APP_DIR}/model}"
HUB="${LAYA_HUB:-TDoSiMa/sima-laya}"
check=0
runtime_only=0
alias="${CREATE_ALIAS:-1}"
icon="${CREATE_DESKTOP_ICON:-1}"

# Output in the manner of the other Neat demos: the Neat sparkle, sections, and one status
# line per thing done. Plain text when not a terminal, when TERM is dumb or NO_COLOR is set.
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  P_TEAL=$'\033[38;2;61;179;138m'; P_GREEN=$'\033[38;2;74;168;54m'; P_LIME=$'\033[38;2;154;190;30m'
  P_BLUE=$'\033[38;2;58;125;216m'; P_ORANGE=$'\033[38;2;223;108;30m'; P_INK=$'\033[38;2;60;66;74m'
  C_ACCENT=$'\033[38;2;47;212;192m'; C_MUTED=$'\033[38;2;140;150;160m'
  C_OK=$'\033[38;2;53;196;137m'; C_WARN=$'\033[38;2;224;173;74m'; C_ERR=$'\033[38;2;239;91;98m'
else
  C_RESET=''; C_BOLD=''; C_DIM=''; P_TEAL=''; P_GREEN=''; P_LIME=''; P_BLUE=''; P_ORANGE=''; P_INK=''
  C_ACCENT=''; C_MUTED=''; C_OK=''; C_WARN=''; C_ERR=''
fi
step() { printf '\n%s\n' "${C_ACCENT}${C_BOLD}▸${C_RESET} $*"; }
info() { printf '%s\n' "${C_MUTED}·${C_RESET} $*"; }
ok()   { printf '%s\n' "${C_OK}✔${C_RESET} $*"; }
warn() { printf '%s\n' "${C_WARN}⚠${C_RESET} $*" >&2; }
fail() { printf '%s\n' "${C_ERR}✘${C_RESET} $1" >&2; shift; for line in "$@"; do info "$line" >&2; done; exit 1; }

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
  printf '   %sNEAT%s %sDecision Studio%s  %s· setup%s\n' \
    "${C_BOLD}" "${C_RESET}" "${C_ACCENT}${C_BOLD}" "${C_RESET}" "${C_MUTED}" "${C_RESET}"
  printf '   %sBuilds the runtime, sees to the models, and adds the shortcuts%s\n' "${C_MUTED}" "${C_RESET}"
}
# A section break: a row of sparkles in the logo's colours, then a heading.
section() {
  local palette=("${P_TEAL}" "${P_GREEN}" "${P_LIME}" "${P_BLUE}" "${P_ORANGE}") line="   " i
  for ((i = 0; i < 12; i++)); do line+="${palette[i % 5]}✦${C_RESET} "; done
  printf '\n%s\n   %s%s%s\n' "${line}" "${C_BOLD}" "$1" "${C_RESET}"
}

for arg in "$@"; do
  case "$arg" in
    --check) check=1 ;;
    --runtime-only) runtime_only=1 ;;
    --no-alias) alias=0 ;;
    --no-desktop-icon) icon=0 ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf '%s\n' "setup.sh: unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

banner

# ------------------------------------------------------------------------------ environment
section "Environment"
if [[ "$(uname -m)" != "aarch64" ]]; then
  fail "This runs on the Modalix DevKit, not on the host." \
       "From the host: bin/laya-deploy [build/laya] --board user@board   (copies the app, runs setup.sh there)"
fi
for tool in cmake g++ python3; do
  command -v "$tool" > /dev/null || fail "$tool is not installed." "sudo apt install cmake g++ python3"
done
ok "cmake, g++ and python3 ${C_DIM}($(python3 --version 2>&1 | cut -d' ' -f2))${C_RESET}"
if ! find /usr/lib /usr/local/lib -name SimaLMMConfig.cmake 2> /dev/null | grep -q .; then
  fail "The LLiMa runtime development package is missing." "Install it with: sima-cli neat install llima"
fi
ok "LLiMa runtime library ${C_DIM}(sima-lmm-dev)${C_RESET}"
[[ -f /usr/include/nlohmann/json.hpp ]] \
  || fail "nlohmann/json.hpp is missing." "sudo apt install nlohmann-json3-dev"
ok "nlohmann JSON headers"

# ---------------------------------------------------------------------------------- runtime
section "Runtime"
step "Building the C++ runtime: ${C_DIM}${APP_DIR}/runtime${C_RESET}"
log="$(mktemp)"
if ! { cmake -S "${APP_DIR}/runtime" -B "${APP_DIR}/runtime/build" -DCMAKE_BUILD_TYPE=Release \
       && cmake --build "${APP_DIR}/runtime/build" -j"$(nproc)"; } > "$log" 2>&1; then
  tail -n 25 "$log" >&2
  rm -f "$log"
  fail "The runtime did not build; the end of the compiler's output is above."
fi
rm -f "$log"
ln -sf "${APP_DIR}/runtime/build/laya" "${APP_DIR}/laya"
ok "Runtime built: ${C_DIM}${APP_DIR}/laya${C_RESET}"
# `./run.sh update` wants no more than this: the models and shortcuts are as they were.
[[ $runtime_only -eq 1 ]] && exit 0

# ----------------------------------------------------------------------------------- models
section "Models"
# Which compiled models are on the board, and whether each is complete.
on_board() {
  python3 - "${APP_DIR}" <<'PY'
import json, os, sys
root = sys.argv[1]
for entry in sorted(os.listdir(root)):
    path = os.path.join(root, entry)
    if not (entry == "model" or entry.startswith("model-")):
        continue
    name = "general" if entry == "model" else entry[len("model-"):]
    if os.path.isfile(os.path.join(path, "clm_config.json")):       # CLM: one graph, in a chain of files
        cfg = json.load(open(os.path.join(path, "clm_config.json")))
        names = [n for files in cfg["elfs"].values() for n in files] + [cfg["token_embeddings"], cfg["tokenizer"], cfg["heads"]]
        graphs = ", ".join(sorted(cfg["elfs"], key=int)) + f" tokens, each in {len(next(iter(cfg['elfs'].values())))} parts"
    elif os.path.isfile(os.path.join(path, "laya_config.json")):
        cfg = json.load(open(os.path.join(path, "laya_config.json")))
        names = list(cfg["elfs"].values()) + [cfg["token_embeddings"], cfg["act_tail"], cfg["tokenizer"]]
        graphs = ", ".join(sorted(cfg["elfs"], key=int)) + " tokens"
    else:
        continue
    missing = [n for n in names if not os.path.isfile(os.path.join(path, n))]
    print(f"{name}\t{'missing ' + ', '.join(missing) if missing else graphs + ' · ' + cfg['precision']}\t{int(bool(missing))}")
PY
}
hub() { python3 "${APP_DIR}/webapp/server.py" --model "${MODEL_DIR}" --root "${APP_DIR}" --hub "${HUB}" "$@"; }

found=0
while IFS=$'\t' read -r name detail broken; do
  [[ -n "$name" ]] || continue
  found=$((found + 1))
  if [[ "$broken" == "1" ]]; then warn "${name}: ${detail}; copy or download it again"; else ok "${name} ${C_DIM}${detail}${C_RESET}"; fi
done < <(on_board)

wanted="${LAYA_MODELS:-}"
if [[ -z "$wanted" && $found -eq 0 && -t 0 && -t 1 ]]; then
  # Nothing on the board and somebody is here to ask: offer what Hugging Face has.
  info "No model on this board yet. On ${C_BOLD}${HUB}${C_RESET}:"
  if listing="$(hub --list-hub 2> /dev/null)" && [[ -n "$listing" ]]; then
    names=()
    while IFS=$'\t' read -r name title size _; do
      names+=("$name")
      printf '  %s%d.%s %s%s%s  %s%s · %s%s\n' "${C_DIM}" "${#names[@]}" "${C_RESET}" "${C_BOLD}" "$title" "${C_RESET}" "${C_MUTED}" "$name" "$size" "${C_RESET}"
    done <<< "$listing"
    read -r -p "${C_MUTED}  Download now? Numbers or names, \"all\", or blank for later ▸ ${C_RESET}" answer || answer=""
    for token in ${answer//,/ }; do
      if [[ "$token" =~ ^[0-9]+$ ]] && (( token >= 1 && token <= ${#names[@]} )); then wanted+=" ${names[token - 1]}"; else wanted+=" ${token}"; fi
    done
  else
    warn "Could not read the list from Hugging Face (is the board online?)."
  fi
fi
if [[ -n "${wanted// /}" ]]; then
  step "Downloading from ${HUB}:${C_DIM}${wanted/#/ }${C_RESET}"
  # shellcheck disable=SC2086
  if hub --fetch ${wanted//,/ }; then ok "Models downloaded."; else warn "A download did not finish; the Models page continues it."; fi
elif [[ $found -eq 0 ]]; then
  info "No model yet. Download one on the app's Models page or with ${C_BOLD}neat-decision --cli${C_RESET},"
  info "or compile one on the host (bin/laya-compile) and copy it over with bin/laya-deploy."
fi

# -------------------------------------------------------------------------------- shortcuts
section "Shortcuts"
# The alias goes where this user's bash looks: ~/.bashrc for terminals on the desktop, and the
# login file (the first that exists of ~/.bash_profile, ~/.bash_login, ~/.profile) for ssh.
install_alias() {
  local line="alias neat-decision='${APP_DIR}/run.sh'" login="" file files=()
  [[ -f "${HOME}/.bashrc" ]] && files+=("${HOME}/.bashrc")
  for file in "${HOME}/.bash_profile" "${HOME}/.bash_login" "${HOME}/.profile"; do
    [[ -f "$file" ]] && { login="$file"; break; }
  done
  files+=("${login:-${HOME}/.bash_profile}")
  for file in "${files[@]}"; do
    if [[ -f "$file" ]] && grep -qxF "$line" "$file"; then
      ok "'neat-decision' alias already in ${C_DIM}${file}${C_RESET}"
    elif [[ -f "$file" ]] && grep -q "^alias neat-decision=" "$file"; then
      sed -i "s|^alias neat-decision=.*|${line}|" "$file"       # set up before from another directory
      ok "'neat-decision' alias updated in ${C_DIM}${file}${C_RESET}"
    else
      printf '\n# Neat Decision Studio, added by setup.sh\n%s\n' "$line" >> "$file"
      ok "Added a 'neat-decision' alias to ${C_DIM}${file}${C_RESET}"
      info "It works in a new shell, or after: ${C_BOLD}source ${file}${C_RESET}"
    fi
    # The app was Neat Laya Studio before, with an alias of that name: it goes, and the
    # comment setup wrote above it. `neat-decision` is the one name now.
    if grep -q "^alias neat-laya=" "$file"; then
      sed -i -e '/^# Neat Laya Studio, added by setup\.sh$/d' -e '/^alias neat-laya=/d' "$file"
      ok "Removed the old 'neat-laya' alias from ${C_DIM}${file}${C_RESET}"
    fi
  done
}

# A launcher on the desktop and in the applications menu. It runs `run.sh --desktop` in a
# terminal window: the app starts, or is found running, and the board's browser opens on it.
# Closing that window stops the app if the window started it.
install_icon() {
  local desktop_dir="${XDG_DESKTOP_DIR:-${HOME}/Desktop}" apps_dir="${HOME}/.local/share/applications"
  if [[ ! -d "$desktop_dir" && ! -d /usr/share/xsessions ]]; then
    info "This board has no desktop; no icon installed."
    return 0
  fi
  local exec_line terminal=true target targets=() trusted=1 bus="${DBUS_SESSION_BUS_ADDRESS:-}"
  exec_line="Exec=\"${APP_DIR}/run.sh\" --desktop"
  if command -v x-terminal-emulator > /dev/null 2>&1; then
    exec_line="Exec=x-terminal-emulator -T \"Neat Decision Studio\" -e \"${APP_DIR}/run.sh --desktop\""
    terminal=false
  fi
  [[ -z "$bus" && -S "/run/user/$(id -u)/bus" ]] && bus="unix:path=/run/user/$(id -u)/bus"
  mkdir -p "$apps_dir"
  rm -f "${apps_dir}/neat-laya.desktop" "${desktop_dir}/neat-laya.desktop"    # the icon under the app's old name
  targets+=("${apps_dir}/neat-decision.desktop")
  [[ -d "$desktop_dir" ]] && targets+=("${desktop_dir}/neat-decision.desktop")
  for target in "${targets[@]}"; do
    printf '%s\n' "[Desktop Entry]" "Version=1.0" "Type=Application" "Name=Neat Decision Studio" \
      "Comment=System-1 decision models answering in about 19 ms on the Modalix MLA" \
      "$exec_line" "Path=${APP_DIR}" "Icon=${APP_DIR}/webapp/static/brand/neat-decision.svg" \
      "Terminal=${terminal}" "Categories=Development;" "Keywords=Laya;Decision;SiMa;Modalix;Neat;" \
      "StartupNotify=false" > "$target"
    chmod 755 "$target"
    # XFCE asks before running a launcher it does not trust. It can be told through the
    # desktop session's bus, when somebody is logged in at the board.
    if [[ -n "$bus" ]] && command -v gio > /dev/null 2>&1 \
       && DBUS_SESSION_BUS_ADDRESS="$bus" gio set -t string "$target" metadata::xfce-exe-checksum \
            "$(sha256sum "$target" | cut -d' ' -f1)" > /dev/null 2>&1; then
      DBUS_SESSION_BUS_ADDRESS="$bus" gio set -t string "$target" metadata::trusted true > /dev/null 2>&1 || true
    else
      trusted=0
    fi
  done
  command -v update-desktop-database > /dev/null 2>&1 && update-desktop-database "$apps_dir" > /dev/null 2>&1 || true
  ok "Desktop icon installed: ${C_DIM}${targets[*]}${C_RESET}"
  info "Double-click ${C_BOLD}Neat Decision Studio${C_RESET} on the desktop or in the applications menu: the app starts and the browser opens on it."
  [[ $trusted -eq 1 ]] || info "The first double-click may ask to trust the launcher; choose ${C_BOLD}Mark Executable${C_RESET}."
}

if [[ "$alias" == "1" ]]; then install_alias; else info "Shell alias not added (--no-alias)."; fi
if [[ "$icon" == "1" ]]; then install_icon; else info "Desktop icon not installed (--no-desktop-icon)."; fi

# ------------------------------------------------------------------------------------ check
if [[ $check -eq 1 ]]; then
  section "Check"
  [[ -f "${MODEL_DIR}/laya_config.json" ]] || fail "There is no general model in ${MODEL_DIR} to check with."
  step "Answering a test question on the MLA"
  "${APP_DIR}/laya" run "${MODEL_DIR}" \
    --state "We were billed twice for March. Please refund the duplicate." \
    --question '{"type": "choice", "instructions": "Which department should handle this?", "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"}}' \
    | sed -n '/^{/,$p' \
    || fail "The model did not run." \
            "If the log above says MLA_LOAD_FAILED, the MLA memory pool is full or fragmented:" \
            "stop other MLA applications, or run ./run.sh --reset-mla"
  ok "The model answers on the MLA."
fi

# ------------------------------------------------------------------------------------- done
section "Done"
ok "Setup complete."
address="$(hostname -I 2> /dev/null | awk '{print $1}')"
info "Start the app with ${C_BOLD}neat-decision${C_RESET} ${C_DIM}(${APP_DIR}/run.sh)${C_RESET}, then open ${C_BOLD}http://${address:-<board-ip>}:${LAYA_PORT:-8095}${C_RESET}"
info "Or stay in the terminal: ${C_BOLD}neat-decision --cli${C_RESET}"
printf '\n'
