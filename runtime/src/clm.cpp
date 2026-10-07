// CLM on the MLA: the chain of encoder graphs, and the CPU work on either side of it.
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

#include "laya/clm.hpp"

namespace fs = std::filesystem;
namespace llima = simaai::llima;

namespace clm {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

uint16_t to_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// Large enough that exp() underflows to exactly 0, small enough to stay finite in bfloat16
// once an attention logit is added to it; see MASK_NEG in clm_sima/hostio.py.
constexpr float kMaskNeg = -30000.0f;

float from_bf16(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void evict_page_cache(const fs::path& path) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return;
  posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  close(fd);
}

Config load_config(const fs::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot read " + path.string());
  const json j = json::parse(stream);
  Config cfg;
  cfg.hidden_size = j.at("hidden_size");
  cfg.vocab_size = j.at("vocab_size");
  cfg.model = j.value("model", "clm");
  cfg.precision = j.value("precision", "");
  cfg.token_embeddings = j.at("token_embeddings");
  cfg.tokenizer = j.at("tokenizer");
  if (j.at("heads").is_null())
    throw std::runtime_error("clm_config.json names no heads; build the model directory with --heads");
  cfg.heads = j.at("heads");
  for (const auto& [seq_len, files] : j.at("elfs").items())
    cfg.elfs[static_cast<uint32_t>(std::stoul(seq_len))] = files.get<std::vector<std::string>>();
  if (cfg.elfs.empty()) throw std::runtime_error("clm_config.json lists no compiled graphs");
  return cfg;
}

// y = W x + b, W row-major (out x in).
std::vector<float> dense(const Head::Dense& layer, const std::vector<float>& x) {
  std::vector<float> y(layer.out);
  for (size_t row = 0; row < layer.out; ++row) {
    const float* w = &layer.weight[row * layer.in];
    float a = 0, b = 0, c = 0, d = 0;
    size_t i = 0;
    for (; i + 4 <= layer.in; i += 4) {
      a += w[i] * x[i]; b += w[i + 1] * x[i + 1]; c += w[i + 2] * x[i + 2]; d += w[i + 3] * x[i + 3];
    }
    for (; i < layer.in; ++i) a += w[i] * x[i];
    y[row] = a + b + c + d + layer.bias[row];
  }
  return y;
}

void activate(std::vector<float>& x, const std::string& kind) {
  for (float& v : x) {
    if (kind == "relu") v = std::max(v, 0.0f);
    else if (kind == "silu") v = v / (1.0f + std::exp(-v));
    else v = 0.5f * v * (1.0f + std::erf(v * float(M_SQRT1_2)));   // gelu, the exact one
  }
}

void layer_norm(std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& bias) {
  double mean = 0, var = 0;
  for (float v : x) mean += v;
  mean /= x.size();
  for (float v : x) var += (v - mean) * (v - mean);
  const double scale = 1.0 / std::sqrt(var / x.size() + 1e-5);
  for (size_t i = 0; i < x.size(); ++i) x[i] = float((x[i] - mean) * scale) * weight[i] + bias[i];
}

void normalise(std::vector<float>& x) {
  double norm = 0;
  for (float v : x) norm += double(v) * v;
  const float scale = float(1.0 / (std::sqrt(norm) + 1e-12));
  for (float& v : x) v *= scale;
}

Head read_head(const json& tensors, const json& description, std::ifstream& stream) {
  Head head;
  head.activation = description.value("activation", "gelu");
  head.residual = description.value("residual", false);
  auto read = [&](const json& shape) {
    size_t count = 1;
    for (const auto& dim : shape) count *= dim.get<size_t>();
    std::vector<float> values(count);
    stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(float));
    if (static_cast<size_t>(stream.gcount()) != count * sizeof(float))
      throw std::runtime_error("the heads file is shorter than its description");
    return values;
  };
  std::vector<float> pending;   // a weight, until its bias follows
  json pending_shape;
  for (const auto& tensor : tensors) {
    const std::string name = tensor.at("name");
    std::vector<float> values = read(tensor.at("shape"));
    if (name.ends_with(".weight")) { pending = std::move(values); pending_shape = tensor.at("shape"); continue; }
    if (name.rfind("norms.", 0) == 0) { head.norms.emplace_back(std::move(pending), std::move(values)); continue; }
    Head::Dense layer{std::move(pending), std::move(values), pending_shape.at(1), pending_shape.at(0)};
    if (name == "inp.bias") head.inp = std::move(layer);
    else if (name == "out.bias") head.out = std::move(layer);
    else head.hidden.push_back(std::move(layer));
  }
  if (!head.inp.out || !head.out.out || (!head.norms.empty() && head.norms.size() != head.hidden.size()))
    throw std::runtime_error("the heads description is not a layout this runtime knows");
  return head;
}

std::string strip(const std::string& text) {
  const auto begin = text.find_first_not_of(" \t\r\n\f\v");
  if (begin == std::string::npos) return "";
  return text.substr(begin, text.find_last_not_of(" \t\r\n\f\v") - begin + 1);
}

bool is_blank(const json& value) {
  return value.is_null() || (value.is_string() && value.get<std::string>().empty());
}

bool is_nested(const json& value) { return (value.is_object() || value.is_array()) && !value.empty(); }

double round4(double value) { return std::round(value * 1e4) / 1e4; }

// The MLA dispatcher is process-wide; hold it for as long as any Runtime exists.
int g_runtimes = 0;

}  // namespace

std::string to_text(const json& value, int indent) {
  if (value.is_null()) return "";
  if (value.is_string()) return value.get<std::string>();
  if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
  if (value.is_number()) return value.dump();
  const std::string pad(indent, ' ');
  std::string out;
  if (value.is_object()) {
    const char* between = indent == 0 ? "\n\n" : "\n";
    for (auto it = value.begin(); it != value.end(); ++it) {
      if (it != value.begin()) out += between;
      out += is_nested(it.value()) ? pad + it.key() + ":\n" + to_text(it.value(), indent + 2)
                                   : pad + it.key() + ": " + to_text(it.value());
    }
    return out;
  }
  for (auto it = value.begin(); it != value.end(); ++it) {
    if (it != value.begin()) out += "\n";
    out += is_nested(*it) ? pad + "-\n" + to_text(*it, indent + 2) : pad + "- " + to_text(*it);
  }
  return out;
}

Pair build_pair(const json& state, const json& question) {
  Pair pair;
  pair.type = question.at("type").get<std::string>();
  const json none;
  const json& crit = question.contains("criteria") ? question.at("criteria") : none;
  const std::string ins = strip(to_text(question.value("instructions", json())));
  // Context first, question last: the layout the heads were trained on.
  const std::string context = strip(to_text(state));
  pair.state_text = !context.empty() && !ins.empty() ? context + "\n\n" + ins : (context.empty() ? ins : context);
  if (pair.state_text.empty()) throw std::invalid_argument("there is neither a state nor a question to read");

  if (pair.type == "choice") {
    // An option is embedded as its own text: its description when there is one, else its key.
    if (crit.is_object()) {
      for (const auto& [key, description] : crit.items()) {
        pair.keys.push_back(key);
        pair.candidates.push_back(is_blank(description) ? key : to_text(description));
      }
    } else if (crit.is_array()) {
      for (const auto& label : crit) {
        pair.keys.push_back(to_text(label));
        pair.candidates.push_back(pair.keys.back());
      }
    }
    if (pair.keys.empty()) throw std::invalid_argument("a choice question needs a non-empty \"criteria\"");
  } else if (pair.type == "score") {
    if (!crit.is_array() || crit.size() < 2)
      throw std::invalid_argument("a score question needs \"criteria\" as an ordered list of two or more levels");
    for (size_t level = 0; level < crit.size(); ++level) {
      pair.keys.push_back(std::to_string(level));
      pair.candidates.push_back(to_text(crit[level]));
    }
    pair.levels = crit;
  } else if (pair.type == "noul") {
    for (const char* key : {"false", "true"}) {
      const json& given = crit.is_object() && crit.contains(key) ? crit.at(key) : none;
      const bool yes = std::string(key) == "true";
      const std::string described = !is_blank(given) ? to_text(given)
          : ins.empty() ? key : (yes ? "Yes. This is true: " : "No. This is false: ") + ins;
      pair.keys.push_back(key);
      pair.candidates.push_back(std::string(key) + ": " + described);
    }
  } else {
    throw std::invalid_argument("unknown question type: " + pair.type);
  }
  return pair;
}

std::vector<float> Head::project(const std::vector<float>& embedding) const {
  std::vector<float> x = dense(inp, embedding);
  activate(x, activation);
  for (size_t i = 0; i < hidden.size(); ++i) {
    std::vector<float> h = dense(hidden[i], x);
    if (!norms.empty()) layer_norm(h, norms[i].first, norms[i].second);
    activate(h, activation);
    if (residual) for (size_t j = 0; j < x.size(); ++j) x[j] += h[j];
    else x = std::move(h);
  }
  std::vector<float> y = dense(out, x);
  normalise(y);
  return y;
}

// One sequence length: its graphs and the buffers between them. Graph i reads buffer i and
// writes buffer i + 1, so nothing is copied on the way, and every graph reads the one mask.
// Buffers are token-major, [position][channel]; the mask is therefore [query][key], the
// transpose of the graph's NCHW (1, key, 1, query).
struct Runtime::Chain {
  uint32_t seq_len;
  std::vector<std::unique_ptr<llima::MLABuffer>> buffers;
  llima::MLABuffer mask;
  std::vector<std::unique_ptr<llima::MLAModelWithBuffer>> models;
  std::vector<size_t> masked_for;  // the text lengths the mask currently holds

  Chain(const fs::path& dir, const Config& cfg, uint32_t s, const std::vector<std::string>& elfs)
      : seq_len(s), mask(fmt::format("clm_mask_{}", s), {s, s}, "bfloat16", true) {
    mask.allocate();
    mask.clear();
    for (size_t i = 0; i <= elfs.size(); ++i) {
      buffers.push_back(std::make_unique<llima::MLABuffer>(
          fmt::format("clm_hidden_{}_{}", s, i), std::vector<size_t>{s, cfg.hidden_size}, "bfloat16", true));
      buffers.back()->allocate();
      buffers.back()->clear();
    }
    for (size_t i = 0; i < elfs.size(); ++i) {
      const fs::path elf = dir / elfs[i];
      models.push_back(std::make_unique<llima::MLAModelWithBuffer>(
          elf, std::vector<llima::MLABufferSlice>{llima::MLABufferSlice(buffers[i].get()), llima::MLABufferSlice(&mask)},
          std::vector<llima::MLABufferSlice>{llima::MLABufferSlice(buffers[i + 1].get())}));
      evict_page_cache(elf);   // also before: a previous run may have left it cached
      models.back()->load();
      evict_page_cache(elf);
    }
  }
  ~Chain() { for (auto& model : models) model->free(); }

  // Texts of these lengths laid end to end: a query sees the keys of its own text at or
  // before it. A position past the last text sees itself only; nothing reads its output.
  void set_mask(const std::vector<size_t>& lengths) {
    if (lengths == masked_for) return;
    const uint16_t open = to_bf16(0.0f), closed = to_bf16(kMaskNeg);
    std::vector<uint16_t> rows(size_t(seq_len) * seq_len, closed);
    for (uint32_t pos = 0; pos < seq_len; ++pos) rows[size_t(pos) * seq_len + pos] = open;
    size_t start = 0;
    for (size_t length : lengths) {
      for (size_t query = start; query < start + length; ++query)
        for (size_t key = start; key <= query; ++key) rows[query * seq_len + key] = open;
      start += length;
    }
    mask.upload(rows.data());
    masked_for = lengths;
  }
};

Runtime::Runtime(const fs::path& model_dir, const std::vector<uint32_t>& seq_lens)
    : _cfg(load_config(model_dir / "clm_config.json")) {
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
  // Not populated up front: the table is over a gigabyte and a text touches a few rows of it.
  void* mapped = mmap(nullptr, _embeddings_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (mapped == MAP_FAILED) throw std::runtime_error("cannot map " + table.string());
  _embeddings = static_cast<const uint16_t*>(mapped);

  const fs::path heads = model_dir / _cfg.heads;
  std::ifstream description_stream(fs::path(heads).replace_extension(".json"));
  if (!description_stream) throw std::runtime_error("cannot read the description of " + heads.string());
  const json description = json::parse(description_stream);
  std::ifstream stream(heads, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot read " + heads.string());
  _state_head = read_head(description.at("heads").at("state_head"), description, stream);
  _action_head = read_head(description.at("heads").at("action_head"), description, stream);
  _scale = description.at("scale");
  if (_state_head.inp.in != _cfg.hidden_size)
    throw std::runtime_error("the heads were made for another encoder width");

  if (g_runtimes++ == 0) llima::connect_mla_rt({});
  for (const auto& [seq_len, elfs] : _cfg.elfs) {
    if (!seq_lens.empty() && std::find(seq_lens.begin(), seq_lens.end(), seq_len) == seq_lens.end()) continue;
    _chains[seq_len] = std::make_unique<Chain>(model_dir, _cfg, seq_len, elfs);
  }
  if (_chains.empty()) throw std::runtime_error("no compiled chain was loaded from " + model_dir.string());
}

Runtime::~Runtime() {
  _chains.clear();
  if (--g_runtimes == 0) llima::disconnect_mla_rt();
  if (_embeddings) munmap(const_cast<uint16_t*>(_embeddings), _embeddings_bytes);
}

std::vector<uint32_t> Runtime::seq_lens() const {
  std::vector<uint32_t> result;
  for (const auto& entry : _chains) result.push_back(entry.first);
  return result;
}

uint32_t Runtime::max_tokens() const { return _chains.rbegin()->first; }

std::vector<uint32_t> Runtime::tokenize(const std::string& text) const {
  return _tokenizer->encode(text, false);
}

std::vector<std::vector<float>> Runtime::pass(const std::vector<std::vector<uint32_t>>& texts, Cost& cost, uint32_t seq_len) {
  size_t total = 0;
  std::vector<size_t> lengths;
  for (const auto& ids : texts) {
    if (ids.empty()) throw std::invalid_argument("a text to embed is empty");
    lengths.push_back(ids.size());
    total += ids.size();
  }
  const auto found = seq_len ? _chains.find(seq_len) : _chains.lower_bound(static_cast<uint32_t>(total));
  if (found == _chains.end() || total > found->first)
    throw std::invalid_argument(seq_len ? fmt::format("{} tokens do not fit a loaded {}-token chain", total, seq_len)
                                        : fmt::format("{} tokens exceed the longest loaded chain ({})", total, max_tokens()));
  Chain& chain = *found->second;
  const uint32_t width = _cfg.hidden_size;

  auto stage = Clock::now();
  llima::MLABuffer& first = *chain.buffers.front();
  auto* rows = static_cast<uint16_t*>(first.get_virtual_addr());
  size_t pos = 0;
  for (const auto& ids : texts) {
    for (uint32_t id : ids) {
      if (id >= _cfg.vocab_size) throw std::invalid_argument(fmt::format("token id {} out of range", id));
      std::memcpy(rows + pos++ * width, _embeddings + size_t(id) * width, width * sizeof(uint16_t));
    }
  }
  first.flush_cache();
  chain.set_mask(lengths);
  cost.pre_ms += elapsed_ms(stage);

  stage = Clock::now();
  for (auto& model : chain.models) model->run();
  cost.mla_ms += elapsed_ms(stage);
  cost.passes++;
  cost.seq_len = std::max(cost.seq_len, chain.seq_len);

  const llima::MLABuffer& last = *chain.buffers.back();
  last.invalidate_cache();
  const auto* out = static_cast<const uint16_t*>(last.get_virtual_addr());
  std::vector<std::vector<float>> result;
  size_t end = 0;
  for (size_t length : lengths) {
    end += length;
    result.emplace_back(width);
    std::transform(out + (end - 1) * width, out + end * width, result.back().begin(), from_bf16);
  }
  return result;
}

std::vector<std::vector<float>> Runtime::encode(std::vector<std::vector<uint32_t>> texts, Cost& cost, bool packed) {
  const size_t longest = max_tokens();
  for (auto& ids : texts) {
    if (ids.empty()) throw std::invalid_argument("a text to embed is empty");
    if (ids.size() > longest) {            // the question is at the end: keep the end
      cost.dropped = std::max(cost.dropped, ids.size() - longest);
      ids.erase(ids.begin(), ids.end() - longest);
    }
    cost.tokens = std::max(cost.tokens, ids.size());
  }
  cost.texts += texts.size();
  std::vector<std::vector<float>> result(texts.size());
  if (!packed) {
    for (size_t i = 0; i < texts.size(); ++i) result[i] = std::move(pass({texts[i]}, cost)[0]);
    return result;
  }

  // Which texts share a pass. For each chain length that could hold the longest text: put the
  // texts, longest first, each into the first pass it fits; a pass then runs on the shortest
  // chain that holds it. Take the length whose passes cost least by what warm-up measured
  // (or, unmeasured, by their positions).
  std::vector<size_t> order(texts.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return texts[a].size() > texts[b].size(); });
  const auto price = [&](uint32_t seq_len) {
    const auto measured = _pass_ms.find(seq_len);
    return measured != _pass_ms.end() ? measured->second : double(seq_len);
  };
  std::vector<std::vector<size_t>> best;
  double best_cost = 0;
  for (const auto& [capacity, chain] : _chains) {
    if (texts[order.front()].size() > capacity) continue;
    std::vector<std::vector<size_t>> bins;
    std::vector<size_t> used;
    for (size_t i : order) {
      size_t bin = 0;
      while (bin < bins.size() && used[bin] + texts[i].size() > capacity) ++bin;
      if (bin == bins.size()) { bins.emplace_back(); used.push_back(0); }
      bins[bin].push_back(i);
      used[bin] += texts[i].size();
    }
    double total = 0;
    for (size_t tokens : used) total += price(_chains.lower_bound(static_cast<uint32_t>(tokens))->first);
    if (best.empty() || total < best_cost) { best = std::move(bins); best_cost = total; }
  }
  for (const auto& bin : best) {
    std::vector<std::vector<uint32_t>> together;
    for (size_t i : bin) together.push_back(texts[i]);
    auto hidden = pass(together, cost);
    for (size_t k = 0; k < bin.size(); ++k) result[bin[k]] = std::move(hidden[k]);
  }
  return result;
}

void Runtime::warm_up() {
  const std::vector<uint32_t> ids = tokenize("Is this a warm-up?");
  for (const auto& [seq_len, chain] : _chains) {
    Cost first, timed;
    pass({ids}, first, seq_len);
    pass({ids}, timed, seq_len);
    _pass_ms[seq_len] = timed.mla_ms + timed.pre_ms;
  }
}

json Runtime::predict(const json& state, const json& questions, double temperature) {
  if (!questions.is_object() || questions.empty())
    throw std::invalid_argument("questions must be a non-empty object of id -> question");
  if (!(temperature > 0 && temperature <= 100)) throw std::invalid_argument("temperature must be in (0, 100]");

  // What each question needs embedded: its state text for the state head, its candidates for
  // the action head. What is not in memory goes through the encoder together.
  std::vector<std::pair<std::string, Pair>> pairs;
  std::vector<std::string> missing;   // cache keys, each once
  for (const auto& [qid, qdef] : questions.items()) {
    try {
      pairs.emplace_back(qid, build_pair(state, qdef));
    } catch (const std::exception& error) {
      throw std::invalid_argument(fmt::format("question \"{}\": {}", qid, error.what()));
    }
    const Pair& pair = pairs.back().second;
    const auto want = [&](const std::string& key) {
      if (!_cache.contains(key) && std::find(missing.begin(), missing.end(), key) == missing.end()) missing.push_back(key);
    };
    want("s:" + pair.state_text);
    for (const auto& candidate : pair.candidates) want("a:" + candidate);
  }

  Cost cost;
  double post_ms = 0;
  size_t wanted = 0;
  for (const auto& [qid, pair] : pairs) wanted += 1 + pair.candidates.size();
  if (!missing.empty()) {
    auto stage = Clock::now();
    std::vector<std::vector<uint32_t>> ids;
    for (const auto& key : missing) {
      ids.push_back(tokenize(key.substr(2)));
      if (ids.back().empty()) throw std::invalid_argument("a text to embed is empty");
    }
    cost.tokenize_ms = elapsed_ms(stage);
    auto hidden = encode(std::move(ids), cost);
    stage = Clock::now();
    for (size_t i = 0; i < missing.size(); ++i) {
      normalise(hidden[i]);
      std::vector<float> vector = (missing[i][0] == 's' ? _state_head : _action_head).project(hidden[i]);
      // Room is made from what was used longest ago, never from what this request needs.
      while (_cache.size() >= std::max(_cache_capacity, wanted + 1)) {
        _cache.erase(_recent.front().key);
        _recent.pop_front();
      }
      _recent.push_back({missing[i], std::move(vector)});
      _cache[missing[i]] = std::prev(_recent.end());
    }
    post_ms += elapsed_ms(stage);
  }
  const auto vector_of = [&](const std::string& key) -> const std::vector<float>& {
    const auto found = _cache.find(key);
    _recent.splice(_recent.end(), _recent, found->second);
    return found->second->vector;
  };

  const auto stage = Clock::now();
  json answers = json::object();
  for (const auto& [qid, pair] : pairs) {
    const std::vector<float>& zs = vector_of("s:" + pair.state_text);
    const size_t k = pair.candidates.size();
    std::vector<double> p(k);
    for (size_t i = 0; i < k; ++i) {
      const std::vector<float>& za = vector_of("a:" + pair.candidates[i]);
      double cosine = 0;
      for (size_t j = 0; j < zs.size(); ++j) cosine += double(zs[j]) * za[j];
      p[i] = _scale * cosine / temperature;
    }
    const double top = *std::max_element(p.begin(), p.end());
    double sum = 0;
    for (auto& value : p) sum += value = std::exp(value - top);
    for (auto& value : p) value /= sum;

    const size_t best = std::max_element(p.begin(), p.end()) - p.begin();
    // Upstream's confidence: the top probability minus the mean of the rest.
    const double confidence = k < 2 ? 1.0 : std::clamp(p[best] - (1.0 - p[best]) / double(k - 1), 0.0, 1.0);
    json answer;
    if (pair.type == "choice") {
      json probabilities = json::object();
      for (size_t i = 0; i < k; ++i) probabilities[pair.keys[i]] = round4(p[i]);
      answer = {{"type", "choice"}, {"choice", pair.keys[best]}, {"probabilities", probabilities},
                {"confidence", round4(confidence)}};
    } else if (pair.type == "score") {
      double expected = 0;
      json legend = json::object(), probabilities = json::object();
      for (size_t i = 0; i < k; ++i) {
        expected += i * p[i];
        legend[pair.keys[i]] = to_text(pair.levels[i]);
        probabilities[pair.keys[i]] = round4(p[i]);
      }
      answer = {{"type", "score"}, {"score", round4(expected)}, {"legend", legend},
                {"probabilities", probabilities}, {"confidence", round4(confidence)}};
    } else {
      answer = {{"type", "noul"}, {"noul", round4(p[1])}, {"confidence", round4(std::max(p[1], 1.0 - p[1]))}};
    }
    answer["answer_confidence"] = round4(std::clamp(p[best], 0.0, 1.0));
    answers[qid] = answer;
  }
  post_ms += elapsed_ms(stage);

  auto ms = [](double value) { return std::round(value * 1e3) / 1e3; };
  return {
      {"answers", answers},
      {"usage",
       {{"tokens", cost.tokens},
        {"max_len", max_tokens()},
        {"seq_len", cost.seq_len ? cost.seq_len : _chains.begin()->first},
        {"state_tokens_dropped", cost.dropped},
        {"truncated", cost.dropped > 0},
        {"decisions", questions.size()},
        {"encoder_passes", cost.passes},
        {"texts_embedded", cost.texts},
        {"cached", wanted - missing.size()},
        {"tokenize_ms", ms(cost.tokenize_ms)},
        {"pre_ms", ms(cost.pre_ms)},
        {"mla_ms", ms(cost.mla_ms)},
        {"post_ms", ms(post_ms)},
        {"latency_ms", ms(cost.tokenize_ms + cost.pre_ms + cost.mla_ms + post_ms)}}},
  };
}

}  // namespace clm
