// CLM (Contrastive-LM/CLM-v0.1-8B) on the Modalix MLA.
//
// CLM answers the same three kinds of question as Laya, differently: a frozen Qwen3-8B reads
// a text and its last token's hidden state is the text's embedding; a small state head
// projects the embedding of "state + question", a small action head the embedding of each
// option, and the answer is the softmax over the scaled cosines.
//
// The encoder is compiled as a chain of graphs, each a run of decoder layers (`clm_sima`),
// that pass one hidden-state buffer along; there is a chain for each sequence length that was
// compiled and loaded. Everything else is here on the CPU: tokenizing, the embedding lookup,
// the attention mask, the two heads, and the layout of a question, which is upstream's
// (`clm/schema.py`, Apache-2.0).
//
// A pass through a chain costs the same however little of it is used, so it is used fully:
// the texts a request needs (a question's state and each of its options) are laid end to end
// in one pass, each seeing only itself through the mask, and come out as they would alone.
// What has been embedded is kept, so a text that has been seen costs nothing the second time.
#ifndef LAYA_CLM_HPP_
#define LAYA_CLM_HPP_

#include <cstdint>
#include <filesystem>
#include <list>
#include <map>
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
  uint32_t hidden_size = 0, vocab_size = 0;
  std::string model, precision, token_embeddings, tokenizer, heads;
  std::map<uint32_t, std::vector<std::string>> elfs;  // sequence length -> its graphs, in the order they run
};

// What a request cost the encoder.
struct Cost {
  double tokenize_ms = 0, pre_ms = 0, mla_ms = 0;
  size_t passes = 0, texts = 0, tokens = 0, dropped = 0;  // tokens: the longest text; dropped: cut from one
  uint32_t seq_len = 0;                                   // the longest chain a pass ran on
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

class Runtime {
 public:
  // `seq_lens` restricts which compiled chains are loaded; empty loads all of them.
  explicit Runtime(const std::filesystem::path& model_dir, const std::vector<uint32_t>& seq_lens = {});
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  const Config& config() const { return _cfg; }
  std::vector<uint32_t> seq_lens() const;
  uint32_t max_tokens() const;  // the longest loaded chain
  std::vector<uint32_t> tokenize(const std::string& text) const;

  // One pass: token sequences laid end to end on one chain (`seq_len`, or 0 for the shortest
  // they fit), and the last token's hidden state of each, as the encoder gave it.
  std::vector<std::vector<float>> pass(const std::vector<std::vector<uint32_t>>& texts, Cost& cost, uint32_t seq_len = 0);
  // Any number of token sequences, in as few and as short passes as their lengths allow
  // (`packed`), or one pass each. A sequence longer than the longest chain keeps its end.
  std::vector<std::vector<float>> encode(std::vector<std::vector<uint32_t>> texts, Cost& cost, bool packed = true);

  // Runs every loaded chain once, so the first request does not pay for cold caches, and notes
  // what a pass on each costs: that is what the packing is planned by.
  void warm_up();

  // {"answers": {...}, "usage": {...}}, in the shape Laya's runtime answers in.
  json predict(const json& state, const json& questions, double temperature = 1.0);

 private:
  struct Chain;

  Config _cfg;
  std::unique_ptr<simaai::llima::Tokenizer> _tokenizer;
  const uint16_t* _embeddings = nullptr;  // (vocab, hidden) bfloat16, mmapped
  size_t _embeddings_bytes = 0;
  Head _state_head, _action_head;
  float _scale = 1.0f;
  std::map<uint32_t, std::unique_ptr<Chain>> _chains;
  std::map<uint32_t, double> _pass_ms;  // what one pass on each chain takes, measured at warm-up

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
