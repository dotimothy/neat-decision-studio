// CLM (Contrastive-LM/CLM-v0.1-8B) on the Modalix MLA.
//
// CLM answers the same three kinds of question as Laya, differently: a frozen Qwen3-8B reads
// a text and its last token's hidden state is the text's embedding; a small state head
// projects the embedding of "state + question", a small action head the embedding of each
// option, and the answer is the softmax over the scaled cosines.
//
// The encoder is compiled as a chain of graphs, each a run of decoder layers (`clm_sima`),
// that pass one hidden-state buffer along. Everything else is here on the CPU: tokenizing, the
// embedding lookup, the two heads, and the layout of a question, which is upstream's
// (`clm/schema.py`, Apache-2.0). Embedding a text is the expensive part, so projections are
// kept: an option that has been seen costs nothing the second time.
#ifndef LAYA_CLM_HPP_
#define LAYA_CLM_HPP_

#include <cstdint>
#include <filesystem>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace simaai::llima {
class Tokenizer;
}

namespace clm {

// Ordered: the key order of a `choice` question's criteria is the option order.
using json = nlohmann::ordered_json;

struct Config {
  uint32_t hidden_size = 0, vocab_size = 0, seq_len = 0;
  std::string model, precision, token_embeddings, tokenizer, heads;
  std::vector<std::string> elfs;  // in the order they run
};

// What the two heads see for one question (`build_pairs` upstream).
struct Pair {
  std::string type, state_text;
  std::vector<std::string> keys, candidates;
  json levels;  // score: the criteria as given, for the legend
};

// One projection head: hidden -> width -> ... -> projection, upstream's `make_head`.
struct Head {
  struct Dense { std::vector<float> weight, bias; size_t in = 0, out = 0; };
  Dense inp, out;
  std::vector<Dense> hidden;
  std::vector<std::pair<std::vector<float>, std::vector<float>>> norms;  // weight, bias; empty: none
  std::string activation;
  bool residual = false;
  std::vector<float> project(const std::vector<float>& embedding) const;  // L2-normalised
};

struct Embedded {
  std::vector<float> hidden;  // the last token's hidden state, as the encoder gave it
  size_t tokens = 0, dropped = 0;
  double tokenize_ms = 0, pre_ms = 0, mla_ms = 0;
};

class Runtime {
 public:
  explicit Runtime(const std::filesystem::path& model_dir);
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  const Config& config() const { return _cfg; }
  std::vector<uint32_t> tokenize(const std::string& text) const;

  // The encoder: the last token's hidden state for a sequence of at most seq_len token ids.
  std::vector<float> hidden(const std::vector<uint32_t>& ids, double* pre_ms = nullptr, double* mla_ms = nullptr);
  // A text, tokenized and cut to its last seq_len tokens if it is longer.
  Embedded embed(const std::string& text);

  void warm_up();

  // {"answers": {...}, "usage": {...}}, in the shape Laya's runtime answers in.
  json predict(const json& state, const json& questions, double temperature = 1.0);

 private:
  struct Chain;
  // A projection, from the cache or through the encoder and `head`.
  const std::vector<float>& projected(bool is_state, const std::string& text, json& usage_totals);

  Config _cfg;
  std::unique_ptr<simaai::llima::Tokenizer> _tokenizer;
  const uint16_t* _embeddings = nullptr;  // (vocab, hidden) bfloat16, mmapped
  size_t _embeddings_bytes = 0;
  Head _state_head, _action_head;
  float _scale = 1.0f;
  std::unique_ptr<Chain> _chain;

  // Least recently used first. Keys carry which head made the projection.
  struct Entry { std::string key; std::vector<float> vector; };
  std::list<Entry> _recent;
  std::unordered_map<std::string, std::list<Entry>::iterator> _cache;
  size_t _cache_capacity = 4096;
};

// Upstream's rendering of a state or description that may be a string, object or array.
std::string to_text(const json& value, int indent = 0);
Pair build_pair(const json& state, const json& question);

}  // namespace clm

#endif
