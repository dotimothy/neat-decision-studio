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
  cfg.seq_len = j.at("seq_len");
  cfg.model = j.value("model", "clm");
  cfg.precision = j.value("precision", "");
  cfg.token_embeddings = j.at("token_embeddings");
  cfg.tokenizer = j.at("tokenizer");
  if (j.at("heads").is_null())
    throw std::runtime_error("clm_config.json names no heads; build the model directory with --heads");
  cfg.heads = j.at("heads");
  cfg.elfs = j.at("elfs").get<std::vector<std::string>>();
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

// The compiled graphs and the buffers between them: graph i reads buffer i and writes buffer
// i + 1, so nothing is copied on the way. Buffers are token-major, [position][channel].
struct Runtime::Chain {
  std::vector<std::unique_ptr<llima::MLABuffer>> buffers;
  std::vector<std::unique_ptr<llima::MLAModelWithBuffer>> models;

  Chain(const fs::path& dir, const Config& cfg) {
    for (size_t i = 0; i <= cfg.elfs.size(); ++i) {
      buffers.push_back(std::make_unique<llima::MLABuffer>(
          fmt::format("clm_hidden_{}", i), std::vector<size_t>{cfg.seq_len, cfg.hidden_size}, "bfloat16", true));
      buffers.back()->allocate();
      buffers.back()->clear();
    }
    for (size_t i = 0; i < cfg.elfs.size(); ++i) {
      const fs::path elf = dir / cfg.elfs[i];
      models.push_back(std::make_unique<llima::MLAModelWithBuffer>(
          elf, std::vector<llima::MLABufferSlice>{llima::MLABufferSlice(buffers[i].get())},
          std::vector<llima::MLABufferSlice>{llima::MLABufferSlice(buffers[i + 1].get())}));
      evict_page_cache(elf);   // also before: a previous run may have left it cached
      models.back()->load();
      evict_page_cache(elf);
    }
  }
  ~Chain() { for (auto& model : models) model->free(); }
};

Runtime::Runtime(const fs::path& model_dir) : _cfg(load_config(model_dir / "clm_config.json")) {
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
  _chain = std::make_unique<Chain>(model_dir, _cfg);
}

Runtime::~Runtime() {
  _chain.reset();
  if (--g_runtimes == 0) llima::disconnect_mla_rt();
  if (_embeddings) munmap(const_cast<uint16_t*>(_embeddings), _embeddings_bytes);
}

std::vector<uint32_t> Runtime::tokenize(const std::string& text) const {
  return _tokenizer->encode(text, false);
}

std::vector<float> Runtime::hidden(const std::vector<uint32_t>& ids, double* pre_ms, double* mla_ms) {
  const uint32_t s = _cfg.seq_len, width = _cfg.hidden_size;
  if (ids.empty() || ids.size() > s)
    throw std::invalid_argument(fmt::format("{} tokens do not fit the {}-token graphs", ids.size(), s));
  auto stage = Clock::now();
  // Attention is causal, so what lies to the right of the text cannot reach it: the rest of the
  // buffer is left as it is.
  llima::MLABuffer& first = *_chain->buffers.front();
  auto* rows = static_cast<uint16_t*>(first.get_virtual_addr());
  for (size_t pos = 0; pos < ids.size(); ++pos) {
    if (ids[pos] >= _cfg.vocab_size) throw std::invalid_argument(fmt::format("token id {} out of range", ids[pos]));
    std::memcpy(rows + pos * width, _embeddings + size_t(ids[pos]) * width, width * sizeof(uint16_t));
  }
  first.flush_cache();
  if (pre_ms) *pre_ms += elapsed_ms(stage);

  stage = Clock::now();
  for (auto& model : _chain->models) model->run();
  if (mla_ms) *mla_ms += elapsed_ms(stage);

  const llima::MLABuffer& last = *_chain->buffers.back();
  last.invalidate_cache();
  const auto* out = static_cast<const uint16_t*>(last.get_virtual_addr()) + (ids.size() - 1) * width;
  std::vector<float> result(width);
  std::transform(out, out + width, result.begin(), from_bf16);
  return result;
}

Embedded Runtime::embed(const std::string& text) {
  Embedded embedded;
  auto stage = Clock::now();
  std::vector<uint32_t> ids = tokenize(text);
  embedded.tokenize_ms = elapsed_ms(stage);
  if (ids.empty()) throw std::invalid_argument("a text to embed is empty");
  if (ids.size() > _cfg.seq_len) {     // the question is at the end: keep the end
    embedded.dropped = ids.size() - _cfg.seq_len;
    ids.erase(ids.begin(), ids.begin() + embedded.dropped);
  }
  embedded.tokens = ids.size();
  embedded.hidden = hidden(ids, &embedded.pre_ms, &embedded.mla_ms);
  return embedded;
}

void Runtime::warm_up() {
  embed("Is this a warm-up?");
}

const std::vector<float>& Runtime::projected(bool is_state, const std::string& text, json& totals) {
  const std::string key = (is_state ? "s:" : "a:") + text;
  if (const auto found = _cache.find(key); found != _cache.end()) {
    _recent.splice(_recent.end(), _recent, found->second);
    totals["cached"] = totals["cached"].get<size_t>() + 1;
    return found->second->vector;
  }
  Embedded embedded = embed(text);
  const auto stage = Clock::now();
  normalise(embedded.hidden);
  std::vector<float> vector = (is_state ? _state_head : _action_head).project(embedded.hidden);
  totals["post_ms"] = totals["post_ms"].get<double>() + elapsed_ms(stage);
  totals["tokenize_ms"] = totals["tokenize_ms"].get<double>() + embedded.tokenize_ms;
  totals["pre_ms"] = totals["pre_ms"].get<double>() + embedded.pre_ms;
  totals["mla_ms"] = totals["mla_ms"].get<double>() + embedded.mla_ms;
  totals["encoder_passes"] = totals["encoder_passes"].get<size_t>() + 1;
  if (is_state) {
    totals["tokens"] = std::max(totals["tokens"].get<size_t>(), embedded.tokens);
    totals["state_tokens_dropped"] = std::max(totals["state_tokens_dropped"].get<size_t>(), embedded.dropped);
  }
  if (_cache.size() >= _cache_capacity) {
    _cache.erase(_recent.front().key);
    _recent.pop_front();
  }
  _recent.push_back({key, std::move(vector)});
  _cache[key] = std::prev(_recent.end());
  return _recent.back().vector;
}

json Runtime::predict(const json& state, const json& questions, double temperature) {
  if (!questions.is_object() || questions.empty())
    throw std::invalid_argument("questions must be a non-empty object of id -> question");
  if (!(temperature > 0 && temperature <= 100)) throw std::invalid_argument("temperature must be in (0, 100]");

  json totals = {{"tokens", size_t(0)}, {"state_tokens_dropped", size_t(0)}, {"encoder_passes", size_t(0)},
                 {"cached", size_t(0)}, {"tokenize_ms", 0.0}, {"pre_ms", 0.0}, {"mla_ms", 0.0}, {"post_ms", 0.0}};
  json answers = json::object();
  for (const auto& [qid, qdef] : questions.items()) {
    Pair pair;
    try {
      pair = build_pair(state, qdef);
    } catch (const std::exception& error) {
      throw std::invalid_argument(fmt::format("question \"{}\": {}", qid, error.what()));
    }
    const std::vector<float> zs = projected(true, pair.state_text, totals);   // a copy: the cache may move on
    const size_t k = pair.candidates.size();
    std::vector<double> p(k);
    for (size_t i = 0; i < k; ++i) {
      const std::vector<float>& za = projected(false, pair.candidates[i], totals);
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
  auto ms = [](double value) { return std::round(value * 1e3) / 1e3; };
  const double latency = totals["tokenize_ms"].get<double>() + totals["pre_ms"].get<double>() +
                         totals["mla_ms"].get<double>() + totals["post_ms"].get<double>();
  const size_t dropped = totals["state_tokens_dropped"];
  return {
      {"answers", answers},
      {"usage",
       {{"tokens", totals["tokens"]},
        {"max_len", _cfg.seq_len},
        {"seq_len", _cfg.seq_len},
        {"state_tokens_dropped", dropped},
        {"truncated", dropped > 0},
        {"decisions", questions.size()},
        {"encoder_passes", totals["encoder_passes"]},
        {"cached", totals["cached"]},
        {"tokenize_ms", ms(totals["tokenize_ms"])},
        {"pre_ms", ms(totals["pre_ms"])},
        {"mla_ms", ms(totals["mla_ms"])},
        {"post_ms", ms(totals["post_ms"])},
        {"latency_ms", ms(latency)}}},
  };
}

}  // namespace clm
