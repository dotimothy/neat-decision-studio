#!/usr/bin/env bash
# One-time setup of the Laya app on a Modalix DevKit: check the board has what the runtime
# links against, build the runtime, and check a compiled model is in place.
#
#   ./setup.sh            build and verify
#   ./setup.sh --check    also load the model on the MLA and answer one question
#
# Run it on the board, from the directory `bin/laya-deploy` copied (default /media/nvme/laya).
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${LAYA_MODEL_DIR:-${APP_DIR}/model}"
check=0
for arg in "$@"; do
  case "$arg" in
    --check) check=1 ;;
    -h|--help) sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "setup.sh: unknown option: $arg" >&2; exit 2 ;;
  esac
done

fail() { echo "setup.sh: $*" >&2; exit 1; }

if [[ "$(uname -m)" != "aarch64" ]]; then
  fail "this runs on the Modalix DevKit, not on the host. From the host, use:
    bin/laya-deploy build/laya [user@board]   (it copies everything and runs setup.sh there)"
fi

echo "[1/3] Checking prerequisites"
for tool in cmake g++ python3; do
  command -v "$tool" > /dev/null || fail "$tool is not installed (sudo apt install cmake g++ python3)"
done
if ! find /usr/lib /usr/local/lib -name SimaLMMConfig.cmake 2> /dev/null | grep -q .; then
  fail "the LLiMa runtime development package is missing; install it with:
    sima-cli neat install llima"
fi
[[ -f /usr/include/nlohmann/json.hpp ]] || fail "nlohmann/json.hpp is missing (sudo apt install nlohmann-json3-dev)"

echo "[2/3] Building the runtime"
cmake -S "${APP_DIR}/runtime" -B "${APP_DIR}/runtime/build" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "${APP_DIR}/runtime/build" -j"$(nproc)" | tail -n 1
ln -sf "${APP_DIR}/runtime/build/laya" "${APP_DIR}/laya"

echo "[3/3] Checking the model in ${MODEL_DIR}"
if [[ ! -f "${MODEL_DIR}/laya_config.json" ]]; then
  # No model is not an error: the app's Models page downloads compiled ones from Hugging Face.
  [[ $check -eq 1 ]] && fail "no compiled model in ${MODEL_DIR} to check with"
  echo "  none yet. Start the app and download one on its Models page, or compile one on the"
  echo "  host (bin/laya-compile) and copy it over with bin/laya-deploy."
  echo "Setup complete. Start the app with: ${APP_DIR}/run.sh"
  exit 0
fi
python3 - "${MODEL_DIR}" <<'PY' || fail "the model directory is incomplete; redeploy it"
import json, os, sys
root = sys.argv[1]
cfg = json.load(open(os.path.join(root, "laya_config.json")))
names = list(cfg["elfs"].values()) + [cfg["token_embeddings"], cfg["act_tail"], cfg["tokenizer"]]
missing = [n for n in names if not os.path.isfile(os.path.join(root, n))]
if missing:
    print("  missing:", ", ".join(missing)); sys.exit(1)
print("  graphs:", ", ".join(f"{s} tokens" for s in sorted(cfg["elfs"], key=int)), "| weights:", cfg["precision"])
PY

if [[ $check -eq 1 ]]; then
  echo "Answering a test question on the MLA"
  "${APP_DIR}/laya" run "${MODEL_DIR}" \
    --state "We were billed twice for March. Please refund the duplicate." \
    --question '{"type": "choice", "instructions": "Which department should handle this?", "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"}}' \
    | sed -n '/^{/,$p' \
    || fail "the model did not run. If the log above says MLA_LOAD_FAILED, the MLA memory pool is
  full or fragmented: stop other MLA applications, or run ./run.sh --reset-mla"
fi
echo "Setup complete. Start the app with: ${APP_DIR}/run.sh"
