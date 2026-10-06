// MLA execution for Laya: buffers, the compiled graphs, and the CPU work on either side.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <fmt/format.h>
#include <sima_lmm/mla_buffer.hpp>
#include <sima_lmm/mla_model.hpp>
#include <sima_lmm/tokenizer.hpp>

#include "laya/runtime.hpp"

namespace fs = std::filesystem;
namespace llima = simaai::llima;

namespace laya {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Round-to-nearest-even float32 -> bfloat16, and back.
uint16_t to_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

float from_bf16(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// Large enough that exp() underflows to exactly 0, small enough to stay finite in bfloat16
// once an attention logit is added to it; see MASK_NEG in laya_sima/hostio.py.
constexpr float kMaskNeg = -30000.0f;

Config load_config(const fs::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot read " + path.string());
  const json j = json::parse(stream);
  Config cfg;
  cfg.hidden_size = j.at("hidden_size");
  cfg.vocab_size = j.at("vocab_size");
  cfg.act_hidden_size = j.at("act_hidden_size");
  cfg.num_qtypes = j.at("num_qtypes");
  cfg.qtype_channels = j.at("qtype_channels");
  cfg.sliding_window = j.at("sliding_window");
  cfg.max_len = j.at("max_len");
  cfg.head_max_len = j.at("head_max_len");
  cfg.pad_token_id = j.at("pad_token_id");
  cfg.cls_token_id = j.at("cls_token_id");
  cfg.sep_token_id = j.at("sep_token_id");
  cfg.mask_token_id = j.at("mask_token_id");
  cfg.mask_token = j.at("mask_token");
  cfg.precision = j.value("precision", "");
  cfg.temperature = j.at("temperature").get<std::array<float, 3>>();
  for (const auto& [bucket, value] : j.at("temperature_by_options").items())
    cfg.temperature_by_options[bucket] = value.get<float>();
  for (const auto& [seq_len, file] : j.at("elfs").items())
    cfg.elfs[static_cast<uint32_t>(std::stoul(seq_len))] = file.get<std::string>();
  cfg.token_embeddings = j.at("token_embeddings");
  cfg.act_tail = j.at("act_tail");
  cfg.tokenizer = j.at("tokenizer");
  if (cfg.num_qtypes != 3 || cfg.qtype_channels < cfg.num_qtypes)
    throw std::runtime_error("laya_config.json: bad num_qtypes / qtype_channels");
  return cfg;
}

// Additive attention masks, written token-major ([query][key]) into the two DRAM buffers.
void upload_masks(llima::MLABuffer& global_mask, llima::MLABuffer& local_mask, uint32_t seq_len,
                  size_t num_tokens, uint32_t window) {
  const uint16_t open = to_bf16(0.0f), closed = to_bf16(kMaskNeg);
  std::vector<uint16_t> global(size_t(seq_len) * seq_len), local(size_t(seq_len) * seq_len);
  for (uint32_t query = 0; query < seq_len; ++query) {
    uint16_t* g = &global[size_t(query) * seq_len];
    uint16_t* l = &local[size_t(query) * seq_len];
    for (uint32_t key = 0; key < seq_len; ++key) {
      const bool real = key < num_tokens;
      const uint32_t distance = query > key ? query - key : key - query;
      g[key] = real ? open : closed;
      l[key] = real && distance <= window ? open : closed;
    }
  }
  global_mask.upload(global.data());
  local_mask.upload(local.data());
}

// Ask the kernel to drop a file's pages from the page cache. Loading an ELF leaves up to a
// gigabyte of it cached, the cache spills into the CMA pool the MLA allocates from, and the
// next model then fails to load with MLA_LOAD_FAILED. Needs no privileges; best effort.
void evict_page_cache(const fs::path& path) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return;
  posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  close(fd);
}

// The MLA dispatcher is process-wide; hold it for as long as any Runtime exists.
int g_runtimes = 0;

}  // namespace

// One compiled sequence length: its ELF and the DRAM buffers the ELF reads and writes.
// Buffers are token-major (HWC with H = 1): [position][channel]. The masks are therefore
// [query][key], the transpose of the graph's NCHW (1, key, 1, query).
struct Runtime::Graph {
  uint32_t seq_len;
  llima::MLABuffer embeds, global_mask, local_mask, qtype, scores, act_pre;
  llima::MLAModelWithBuffer model;
  // What the mask and question-type buffers currently hold; both are reused across requests.
  size_t loaded_tokens = 0;
  int loaded_qtype = -1;

  Graph(const fs::path& elf, uint32_t s, const Config& cfg)
      : seq_len(s),
        embeds("embeds", {s, cfg.hidden_size}, "bfloat16", true),
        global_mask("global_mask", {s, s}, "bfloat16", true),
        local_mask("local_mask", {s, s}, "bfloat16", true),
        qtype("qtype", {s, cfg.qtype_channels}, "bfloat16", true),
        scores("scores", {s, 1}, "bfloat16", true),
        act_pre("act_pre", {1, cfg.act_hidden_size}, "bfloat16", true),
        model(elf,
              {llima::MLABufferSlice(&embeds), llima::MLABufferSlice(&global_mask),
               llima::MLABufferSlice(&local_mask), llima::MLABufferSlice(&qtype)},
              {llima::MLABufferSlice(&scores), llima::MLABufferSlice(&act_pre)}) {
    for (auto* buffer : {&embeds, &global_mask, &local_mask, &qtype, &scores, &act_pre}) {
      buffer->allocate();
      buffer->clear();
    }
    evict_page_cache(elf);   // also before: a previous run may have left it cached
    model.load();
    evict_page_cache(elf);
  }

  ~Graph() { model.free(); }

  void set_masks(size_t num_tokens, uint32_t window) {
    if (num_tokens == loaded_tokens) return;
    upload_masks(global_mask, local_mask, seq_len, num_tokens, window);
    loaded_tokens = num_tokens;
  }

  // One-hot in the first channels of every position; the rest of the row is padding.
  void set_qtype(QType type, uint32_t channels) {
    if (static_cast<int>(type) == loaded_qtype) return;
    std::vector<uint16_t> one_hot(size_t(seq_len) * channels, to_bf16(0.0f));
    for (uint32_t pos = 0; pos < seq_len; ++pos)
      one_hot[size_t(pos) * channels + static_cast<int>(type)] = to_bf16(1.0f);
    qtype.upload(one_hot.data());
    loaded_qtype = static_cast<int>(type);
  }
};

// The encoder-only diagnostic graph: same inputs minus the question type, hidden state out.
struct Runtime::EncoderGraph {
  fs::path elf;
  uint32_t seq_len;
  llima::MLABuffer embeds, global_mask, local_mask, hidden;
  llima::MLAModelWithBuffer model;

  EncoderGraph(const fs::path& path, uint32_t s, const Config& cfg)
      : elf(path), seq_len(s),
        embeds("enc_embeds", {s, cfg.hidden_size}, "bfloat16", true),
        global_mask("enc_global_mask", {s, s}, "bfloat16", true),
        local_mask("enc_local_mask", {s, s}, "bfloat16", true),
        hidden("enc_hidden", {s, cfg.hidden_size}, "bfloat16", true),
        model(path,
              {llima::MLABufferSlice(&embeds), llima::MLABufferSlice(&global_mask),
               llima::MLABufferSlice(&local_mask)},
              {llima::MLABufferSlice(&hidden)}) {
    for (auto* buffer : {&embeds, &global_mask, &local_mask, &hidden}) {
      buffer->allocate();
      buffer->clear();
    }
    model.load();
  }
  ~EncoderGraph() { model.free(); }
};

Runtime::Runtime(const fs::path& model_dir, const std::vector<uint32_t>& seq_lens)
    : _cfg(load_config(model_dir / "laya_config.json")) {
  _tokenizer = llima::Tokenizer::from_hf_json(model_dir / _cfg.tokenizer);

  const fs::path table = model_dir / _cfg.token_embeddings;
  _embeddings_bytes = size_t(_cfg.vocab_size) * _cfg.hidden_size * sizeof(uint16_t);
  const int fd = open(table.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + table.string());
  struct stat st {};
  if (fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) != _embeddings_bytes) {
    close(fd);
    throw std::runtime_error(fmt::format("{}: expected {} bytes", table.string(), _embeddings_bytes));
  }
  void* mapped = mmap(nullptr, _embeddings_bytes, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
  close(fd);
  if (mapped == MAP_FAILED) throw std::runtime_error("cannot map " + table.string());
  _embeddings = static_cast<const uint16_t*>(mapped);

  const size_t h = _cfg.act_hidden_size;
  std::vector<float> tail(h * 4 + 2 * h + 2);
  std::ifstream stream(model_dir / _cfg.act_tail, std::ios::binary);
  stream.read(reinterpret_cast<char*>(tail.data()), tail.size() * sizeof(float));
  if (static_cast<size_t>(stream.gcount()) != tail.size() * sizeof(float))
    throw std::runtime_error("bad act tail file: " + (model_dir / _cfg.act_tail).string());
  _act_w_feats.assign(tail.begin(), tail.begin() + h * 4);
  _act_w_out.assign(tail.begin() + h * 4, tail.begin() + h * 6);
  _act_b_out.assign(tail.begin() + h * 6, tail.end());

  if (g_runtimes++ == 0) llima::connect_mla_rt({});
  for (const auto& [seq_len, file] : _cfg.elfs) {
    if (!seq_lens.empty() && std::find(seq_lens.begin(), seq_lens.end(), seq_len) == seq_lens.end())
      continue;
    _graphs[seq_len] = std::make_unique<Graph>(model_dir / file, seq_len, _cfg);
  }
  if (_graphs.empty()) throw std::runtime_error("no compiled graph was loaded from " + model_dir.string());
}

Runtime::~Runtime() {
  _encoder_graph.reset();
  _graphs.clear();
  if (--g_runtimes == 0) llima::disconnect_mla_rt();
  if (_embeddings) munmap(const_cast<uint16_t*>(_embeddings), _embeddings_bytes);
}

uint32_t Runtime::max_tokens() const { return _graphs.rbegin()->first; }

std::vector<uint32_t> Runtime::seq_lens() const {
  std::vector<uint32_t> result;
  for (const auto& entry : _graphs) result.push_back(entry.first);
  return result;
}

std::vector<uint32_t> Runtime::tokenize(const std::string& text) const {
  return _tokenizer->encode(text, false);
}

Runtime::Graph& Runtime::graph_for(size_t num_tokens, uint32_t force_seq_len) {
  if (force_seq_len) {
    const auto found = _graphs.find(force_seq_len);
    if (found == _graphs.end())
      throw std::invalid_argument(fmt::format("no graph is loaded for seq_len {}", force_seq_len));
    if (num_tokens > force_seq_len)
      throw std::invalid_argument(fmt::format("{} tokens do not fit seq_len {}", num_tokens, force_seq_len));
    return *found->second;
  }
  const auto found = _graphs.lower_bound(static_cast<uint32_t>(num_tokens));
  if (found == _graphs.end())
    throw std::invalid_argument(
        fmt::format("{} tokens exceed the largest loaded graph ({})", num_tokens, max_tokens()));
  return *found->second;
}

void Runtime::warm_up() {
  tokenize("choice question: warm up");
  for (const auto& [seq_len, graph] : _graphs) {
    std::vector<uint32_t> ids(seq_len, _cfg.pad_token_id);
    ids.front() = _cfg.cls_token_id;
    ids[1] = _cfg.mask_token_id;
    infer(ids, {1}, QType::Choice, seq_len);
  }
}

// Embedding lookup straight into a DRAM buffer: hidden_size is a multiple of 16 elements, so
// rows are contiguous there. Positions past the sequence hold the pad token.
void Runtime::upload_embeddings(void* buffer, uint32_t seq_len, const std::vector<uint32_t>& ids) const {
  const uint32_t hidden = _cfg.hidden_size;
  auto* rows = static_cast<uint16_t*>(buffer);
  for (uint32_t pos = 0; pos < seq_len; ++pos) {
    const uint32_t id = pos < ids.size() ? ids[pos] : _cfg.pad_token_id;
    if (id >= _cfg.vocab_size) throw std::invalid_argument(fmt::format("token id {} out of range", id));
    std::memcpy(rows + size_t(pos) * hidden, _embeddings + size_t(id) * hidden, hidden * sizeof(uint16_t));
  }
}

std::vector<float> Runtime::encoder_hidden(const fs::path& elf, uint32_t seq_len,
                                           const std::vector<uint32_t>& ids) {
  if (ids.empty() || ids.size() > seq_len) throw std::invalid_argument("sequence does not fit seq_len");
  if (!_encoder_graph || _encoder_graph->elf != elf || _encoder_graph->seq_len != seq_len) {
    _encoder_graph.reset();
    _encoder_graph = std::make_unique<EncoderGraph>(elf, seq_len, _cfg);
  }
  EncoderGraph& graph = *_encoder_graph;
  upload_embeddings(graph.embeds.get_virtual_addr(), seq_len, ids);
  graph.embeds.flush_cache();
  upload_masks(graph.global_mask, graph.local_mask, seq_len, ids.size(), _cfg.sliding_window);
  graph.model.run();
  std::vector<uint16_t> raw(size_t(seq_len) * _cfg.hidden_size);
  graph.hidden.download(raw.data());
  std::vector<float> out(raw.size());
  std::transform(raw.begin(), raw.end(), out.begin(), from_bf16);
  return out;
}

RawOutput Runtime::infer(const std::vector<uint32_t>& ids, const std::vector<uint32_t>& markers,
                         QType qtype, uint32_t force_seq_len) {
  if (ids.empty() || markers.empty()) throw std::invalid_argument("empty sequence or no option markers");
  Graph& graph = graph_for(ids.size(), force_seq_len);
  const uint32_t s = graph.seq_len;
  RawOutput out;
  out.seq_len = s;

  auto stage = Clock::now();
  upload_embeddings(graph.embeds.get_virtual_addr(), s, ids);
  graph.embeds.flush_cache();
  graph.set_masks(ids.size(), _cfg.sliding_window);
  graph.set_qtype(qtype, _cfg.qtype_channels);
  out.timing.pre_ms = elapsed_ms(stage);

  stage = Clock::now();
  graph.model.run();
  out.timing.mla_ms = elapsed_ms(stage);

  stage = Clock::now();
  std::vector<uint16_t> scores(s), act_pre(_cfg.act_hidden_size);
  graph.scores.download(scores.data());
  graph.act_pre.download(act_pre.data());

  out.logits.reserve(markers.size());
  for (uint32_t marker : markers) {
    if (marker >= ids.size()) throw std::invalid_argument("option marker outside the sequence");
    out.logits.push_back(from_bf16(scores[marker]));
  }

  // The four features the act head takes besides the pooled state, from the untempered
  // softmax over the options: top1, top1 - top2, normalized entropy, option count / 255.
  const float top_logit = *std::max_element(out.logits.begin(), out.logits.end());
  std::vector<double> p(out.logits.size());
  double sum = 0;
  for (size_t i = 0; i < p.size(); ++i) sum += p[i] = std::exp(double(out.logits[i]) - top_logit);
  double top1 = 0, top2 = 0, entropy = 0;
  for (auto& value : p) {
    value /= sum;
    entropy -= value * std::log(std::max(value, 1e-9));
    if (value > top1) { top2 = top1; top1 = value; }
    else if (value > top2) top2 = value;
  }
  const double k = std::max<size_t>(p.size(), 2);
  const float feats[4] = {float(top1), float(top1 - top2), float(entropy / std::log(k)), float(k / 255.0)};

  const size_t h = _cfg.act_hidden_size;
  out.act_logits = {_act_b_out[0], _act_b_out[1]};
  for (size_t i = 0; i < h; ++i) {
    float x = from_bf16(act_pre[i]);
    for (size_t f = 0; f < 4; ++f) x += _act_w_feats[i * 4 + f] * feats[f];
    const float activated = 0.5f * x * (1.0f + std::erf(x * float(M_SQRT1_2)));
    out.act_logits[0] += _act_w_out[i] * activated;
    out.act_logits[1] += _act_w_out[h + i] * activated;
  }
  out.timing.post_ms = elapsed_ms(stage);
  return out;
}

json Runtime::predict(const json& state, const json& questions, uint32_t pinned, uint32_t max_len,
                      uint32_t head_max_len) {
  if (pinned && !_graphs.contains(pinned))
    throw std::invalid_argument(fmt::format("no graph is loaded for seq_len {}", pinned));
  if (max_len && max_len < 16) throw std::invalid_argument("the token budget must be at least 16");
  // The budget in force: the request's, within the pinned graph, the largest loaded graph and
  // what the checkpoint was trained for.
  uint32_t budget = std::min(_cfg.max_len, pinned ? pinned : max_tokens());
  if (max_len) budget = std::min(budget, max_len);
  if (state.is_null()) throw std::invalid_argument("state must not be null");
  if (!questions.is_object() || questions.empty())
    throw std::invalid_argument("questions must be a non-empty object of id -> question");

  auto stage = Clock::now();
  // A conversation list is newest-last: keep its tail when the state has to be cut.
  const bool truncate_left = state.is_array();
  std::string text = state.is_string() ? state.get<std::string>() : python_json(state);
  for (size_t pos = 0; (pos = text.find(_cfg.mask_token, pos)) != std::string::npos; ++pos)
    text.replace(pos, _cfg.mask_token.size(), " ");
  const auto state_ids = tokenize(text);
  double tokenize_ms = elapsed_ms(stage);

  json answers = json::object();
  Timing total;
  size_t max_tokens_used = 0, dropped = 0;
  uint32_t seq_len = 0;
  for (const auto& [qid, qdef] : questions.items()) {
    stage = Clock::now();
    Question question;
    Sequence seq;
    try {
      question = parse_question(qdef);
      seq = build_sequence(state_ids, question, truncate_left, budget, head_max_len);
    } catch (const std::invalid_argument& error) {
      throw std::invalid_argument(fmt::format("question \"{}\": {}", qid, error.what()));
    }
    tokenize_ms += elapsed_ms(stage);
    const RawOutput out = infer(seq.ids, seq.markers, question.type, pinned);
    answers[qid] = decode_answer(_cfg, question, out);
    total.pre_ms += out.timing.pre_ms;
    total.mla_ms += out.timing.mla_ms;
    total.post_ms += out.timing.post_ms;
    max_tokens_used = std::max(max_tokens_used, seq.ids.size());
    dropped = std::max(dropped, seq.state_tokens - seq.state_tokens_used);
    seq_len = std::max(seq_len, out.seq_len);
  }
  total.tokenize_ms = tokenize_ms;
  auto ms = [](double value) { return std::round(value * 1e3) / 1e3; };
  return {
      {"answers", answers},
      {"usage",
       {{"tokens", max_tokens_used},
        {"max_len", budget},
        {"seq_len", seq_len},
        {"state_tokens_dropped", dropped},
        {"truncated", dropped > 0},
        {"decisions", questions.size()},
        {"tokenize_ms", ms(total.tokenize_ms)},
        {"pre_ms", ms(total.pre_ms)},
        {"mla_ms", ms(total.mla_ms)},
        {"post_ms", ms(total.post_ms)},
        {"latency_ms", ms(total.total_ms())}}},
  };
}

}  // namespace laya
