// laya: run the Laya decision model on the Modalix MLA.
//
//   laya run   <model_dir> --state TEXT --questions FILE    answer a set of questions
//   laya raw   <model_dir> --input FILE                     score pre-tokenized sequences
//   laya bench <model_dir> [--tokens N] [--iters N]         time the forward pass
//   laya serve <model_dir>                                  one JSON request per stdin line
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <fmt/format.h>

#include "laya/runtime.hpp"

namespace {

using laya::json;

const char* kUsage = R"(usage: laya <command> <model_dir> [options]

commands:
  run     answer questions about one state
            --state TEXT | --state-file FILE     the text (or JSON document) to decide on
            --questions FILE | --question JSON   {"id": {"type": ..., "instructions": ...}, ...}
                                                 or one question object
  raw     score pre-tokenized sequences (parity checks against the PyTorch model)
            --input FILE     {"cases": [{"name", "ids", "markers", "qtype"}, ...]}
  bench   time the forward pass on a synthetic sequence
            --tokens N       sequence length in tokens (default: the largest loaded graph)
            --iters N        timed iterations after 3 warm-up runs (default: 50)
  serve   read {"state": ..., "questions": {...}} per stdin line, answer per stdout line
          (optional "seq_len" pins the graph; "max_len" and "head_max_len" set the token budget)
  hidden  diagnostic: dump the encoder hidden state from an encoder-only ELF
            --elf FILE --seq-len N --input FILE --out FILE

options:
  --seq-lens LIST    load only these compiled sequence lengths, e.g. 128,512
  --seq-len N        run/raw/bench: pin the graph instead of using the smallest that fits
  --max-len N        run: token budget for question, options and state together; a longer
                     state is cut (default: the checkpoint's, within the loaded graphs)
  --head-max-len N   run: the part of the budget the question and its options may take
)";

struct Args {
  std::string command, model_dir;
  std::map<std::string, std::string> options;
  bool has(const std::string& key) const { return options.contains(key); }
  std::string get(const std::string& key, const std::string& fallback = "") const {
    const auto found = options.find(key);
    return found == options.end() ? fallback : found->second;
  }
};

Args parse_args(int argc, char** argv) {
  if (argc < 3) throw std::invalid_argument("missing command or model directory");
  Args args{argv[1], argv[2], {}};
  for (int i = 3; i < argc; ++i) {
    const std::string key = argv[i];
    if (key.rfind("--", 0) != 0 || i + 1 >= argc)
      throw std::invalid_argument("expected --option VALUE, got: " + key);
    args.options[key.substr(2)] = argv[++i];
  }
  return args;
}

std::string read_file(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot read " + path);
  std::stringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

std::vector<uint32_t> parse_list(const std::string& text) {
  std::vector<uint32_t> values;
  std::stringstream stream(text);
  for (std::string item; std::getline(stream, item, ',');)
    if (!item.empty()) values.push_back(static_cast<uint32_t>(std::stoul(item)));
  return values;
}

// A state given as text is used as text unless it parses as a JSON object or array.
json parse_state(const std::string& text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first != std::string::npos && (text[first] == '{' || text[first] == '[')) {
    json parsed = json::parse(text, nullptr, false);
    if (!parsed.is_discarded()) return parsed;
  }
  return text;
}

int cmd_run(laya::Runtime& runtime, const Args& args) {
  if (args.has("state") == args.has("state-file"))
    throw std::invalid_argument("run needs exactly one of --state and --state-file");
  if (args.has("questions") == args.has("question"))
    throw std::invalid_argument("run needs exactly one of --questions and --question");
  const json state = parse_state(args.has("state") ? args.get("state") : read_file(args.get("state-file")));
  json questions = json::parse(args.has("questions") ? read_file(args.get("questions")) : args.get("question"));
  if (questions.contains("type") && questions.at("type").is_string())
    questions = json{{"question", questions}};
  const auto number = [&](const char* key) { return static_cast<uint32_t>(std::stoul(args.get(key, "0"))); };
  std::cout << runtime.predict(state, questions, number("seq-len"), number("max-len"), number("head-max-len")).dump(2)
            << std::endl;
  return 0;
}

int cmd_raw(laya::Runtime& runtime, const Args& args) {
  const json input = json::parse(read_file(args.get("input")));
  const uint32_t pinned = static_cast<uint32_t>(std::stoul(args.get("seq-len", "0")));
  json results = json::array();
  for (const auto& item : input.at("cases")) {
    const auto ids = item.at("ids").get<std::vector<uint32_t>>();
    json result = {{"name", item.at("name")}, {"tokens", ids.size()}};
    if (ids.size() > (pinned ? pinned : runtime.max_tokens())) {
      result["skipped"] = "too long for the loaded graphs";
    } else {
      const auto out = runtime.infer(ids, item.at("markers").get<std::vector<uint32_t>>(),
                                     static_cast<laya::QType>(item.at("qtype").get<int>()), pinned);
      result["logits"] = out.logits;
      result["act_logits"] = out.act_logits;
      result["seq_len"] = out.seq_len;
      result["mla_ms"] = out.timing.mla_ms;
    }
    results.push_back(result);
  }
  std::cout << json{{"results", results}}.dump(2) << std::endl;
  return 0;
}

// Diagnostic: write the encoder hidden state of each case as float32 [case][seq_len][hidden].
int cmd_hidden(laya::Runtime& runtime, const Args& args) {
  const json input = json::parse(read_file(args.get("input")));
  const uint32_t seq_len = static_cast<uint32_t>(std::stoul(args.get("seq-len")));
  std::ofstream out(args.get("out"), std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + args.get("out"));
  size_t cases = 0;
  for (const auto& item : input.at("cases")) {
    const auto hidden = runtime.encoder_hidden(args.get("elf"), seq_len, item.at("ids").get<std::vector<uint32_t>>());
    out.write(reinterpret_cast<const char*>(hidden.data()), hidden.size() * sizeof(float));
    ++cases;
  }
  std::cout << json{{"cases", cases}, {"seq_len", seq_len}, {"out", args.get("out")}}.dump() << std::endl;
  return 0;
}

int cmd_bench(laya::Runtime& runtime, const Args& args) {
  const uint32_t pinned = static_cast<uint32_t>(std::stoul(args.get("seq-len", "0")));
  const uint32_t tokens = static_cast<uint32_t>(
      std::stoul(args.get("tokens", std::to_string(pinned ? pinned : runtime.max_tokens()))));
  const int iters = std::stoi(args.get("iters", "50"));
  if (tokens < 16) throw std::invalid_argument("bench needs at least 16 tokens");
  const auto& cfg = runtime.config();
  // A plausible layout: [CLS] head [SEP] [MASK] a [MASK] b [MASK] c [SEP] state... [SEP]
  std::vector<uint32_t> ids(tokens, 1000);
  ids.front() = cfg.cls_token_id;
  ids[5] = cfg.sep_token_id;
  const std::vector<uint32_t> markers = {6, 8, 10};
  for (uint32_t marker : markers) ids[marker] = cfg.mask_token_id;
  ids[12] = ids.back() = cfg.sep_token_id;

  std::vector<double> total, mla, pre, post;
  uint32_t seq_len = 0;
  for (int i = 0; i < iters + 3; ++i) {
    // Vary the length by one token every run so the mask upload is part of what is timed.
    const std::vector<uint32_t> run_ids(ids.begin(), ids.end() - (i % 2));
    const auto out = runtime.infer(run_ids, markers, laya::QType::Choice, pinned);
    if (i < 3) continue;
    seq_len = out.seq_len;
    total.push_back(out.timing.total_ms());
    mla.push_back(out.timing.mla_ms);
    pre.push_back(out.timing.pre_ms);
    post.push_back(out.timing.post_ms);
  }
  auto stats = [](std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    auto ms = [](double value) { return std::round(value * 1e3) / 1e3; };
    return json{{"mean", ms(mean)}, {"p50", ms(values[values.size() / 2])},
                {"p95", ms(values[std::min(values.size() - 1, values.size() * 95 / 100)])},
                {"min", ms(values.front())}, {"max", ms(values.back())}};
  };
  const double mean_total = std::accumulate(total.begin(), total.end(), 0.0) / total.size();
  std::cout << json{{"tokens", tokens}, {"seq_len", seq_len}, {"iters", iters},
                    {"precision", cfg.precision},
                    {"latency_ms", stats(total)}, {"mla_ms", stats(mla)},
                    {"pre_ms", stats(pre)}, {"post_ms", stats(post)},
                    {"decisions_per_second", std::round(1e4 / mean_total) / 10.0}}.dump(2)
            << std::endl;
  return 0;
}

int cmd_serve(laya::Runtime& runtime) {
  runtime.warm_up();
  std::cout << json{{"ready", true}, {"seq_lens", runtime.seq_lens()}}.dump() << std::endl;
  for (std::string line; std::getline(std::cin, line);) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    json response;
    try {
      const json request = json::parse(line);
      response = runtime.predict(request.at("state"), request.at("questions"), request.value("seq_len", 0u),
                                 request.value("max_len", 0u), request.value("head_max_len", 0u));
      if (request.contains("id")) response["id"] = request.at("id");
    } catch (const std::exception& error) {
      response = {{"error", error.what()}};
    }
    std::cout << response.dump() << std::endl;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    laya::Runtime runtime(args.model_dir, parse_list(args.get("seq-lens")));
    if (args.command == "run") return cmd_run(runtime, args);
    if (args.command == "raw") return cmd_raw(runtime, args);
    if (args.command == "bench") return cmd_bench(runtime, args);
    if (args.command == "serve") return cmd_serve(runtime);
    if (args.command == "hidden") return cmd_hidden(runtime, args);
    throw std::invalid_argument("unknown command: " + args.command);
  } catch (const std::invalid_argument& error) {
    std::cerr << "laya: " << error.what() << "\n\n" << kUsage;
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "laya: " << error.what() << std::endl;
    return 1;
  }
}
