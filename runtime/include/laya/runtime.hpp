// Laya decision model on the Modalix MLA.
//
// One request is one forward pass of one compiled graph. The graph takes token embeddings and
// returns a score for every position; this runtime does what is left over on the CPU:
// tokenizing, laying out the sequence, the embedding lookup and the attention masks before the
// run, and reading the option positions, the softmax and the tail of the act head after it.
// `laya_sima/hostio.py` is the reference for that contract.
#ifndef LAYA_RUNTIME_HPP_
#define LAYA_RUNTIME_HPP_

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace simaai::llima {
class Tokenizer;
}

namespace laya {

// Ordered: the key order of a `choice` question's criteria is the option order.
using json = nlohmann::ordered_json;

enum class QType : int { Choice = 0, Score = 1, Noul = 2 };

struct Config {
  uint32_t hidden_size = 0, vocab_size = 0, act_hidden_size = 0, num_qtypes = 0, qtype_channels = 0;
  uint32_t sliding_window = 0, max_len = 0, head_max_len = 0;
  uint32_t pad_token_id = 0, cls_token_id = 0, sep_token_id = 0, mask_token_id = 0;
  std::string mask_token, precision;
  std::array<float, 3> temperature{1.f, 1.f, 1.f};
  std::map<std::string, float> temperature_by_options;
  std::map<uint32_t, std::string> elfs;  // seq_len -> file name
  std::string token_embeddings, act_tail, tokenizer;
};

// A question, normalized: what `Agent._to_internal` + `render_options` produce upstream.
struct Question {
  QType type = QType::Choice;
  std::string instructions;
  std::vector<std::string> option_texts;  // text shown to the model after each [MASK]
  std::vector<std::string> option_keys;   // choice labels; empty for score and noul
};

struct Sequence {
  std::vector<uint32_t> ids;
  std::vector<uint32_t> markers;  // position of each option's [MASK]
  size_t state_tokens = 0, state_tokens_used = 0;
};

struct Timing {
  double tokenize_ms = 0, pre_ms = 0, mla_ms = 0, post_ms = 0;
  double total_ms() const { return tokenize_ms + pre_ms + mla_ms + post_ms; }
};

struct RawOutput {
  std::vector<float> logits;        // one per marker, before temperature
  std::array<float, 2> act_logits;  // [act, escalate]
  uint32_t seq_len = 0;             // the compiled graph that served the request
  Timing timing;
};

class Runtime {
 public:
  // `seq_lens` restricts which compiled graphs are loaded; empty loads all of them.
  explicit Runtime(const std::filesystem::path& model_dir, const std::vector<uint32_t>& seq_lens = {});
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  const Config& config() const { return _cfg; }
  uint32_t max_tokens() const;  // the largest loaded graph
  std::vector<uint32_t> seq_lens() const;

  std::vector<uint32_t> tokenize(const std::string& text) const;
  // `limit` caps the whole sequence in tokens and `head_limit` the question and its options;
  // 0 means the checkpoint's own budget, which is also never exceeded, nor is the largest
  // loaded graph.
  Sequence build_sequence(const std::vector<uint32_t>& state_ids, const Question& question,
                          bool truncate_left = false, uint32_t limit = 0, uint32_t head_limit = 0) const;

  // Runs every loaded graph and the tokenizer once, so the first real request is not the one
  // that pays for cold caches.
  void warm_up();

  // One forward pass. `force_seq_len` pins the graph instead of taking the smallest that fits.
  RawOutput infer(const std::vector<uint32_t>& ids, const std::vector<uint32_t>& markers,
                  QType qtype, uint32_t force_seq_len = 0);

  // Diagnostic: the encoder's hidden state, [seq_len][hidden], from an encoder-only ELF
  // (`laya-compile --encoder_only`). It shows what the MLA's arithmetic does to the features a
  // decision head is trained on.
  std::vector<float> encoder_hidden(const std::filesystem::path& elf, uint32_t seq_len,
                                    const std::vector<uint32_t>& ids);

  // The upstream `Agent.system_one` call: {"answers": {...}, "usage": {...}}.
  // `seq_len` pins one compiled graph; 0 takes the smallest loaded graph the request fits in.
  // `max_len` is the token budget for a sequence (question, options and state together) and
  // `head_max_len` the part of it the question and options may take; 0 means the checkpoint's.
  // A state that does not fit is cut, and `usage` says by how much.
  json predict(const json& state, const json& questions, uint32_t seq_len = 0, uint32_t max_len = 0,
               uint32_t head_max_len = 0);

 private:
  struct Graph;
  struct EncoderGraph;
  Graph& graph_for(size_t num_tokens, uint32_t force_seq_len);
  void upload_embeddings(void* rows, uint32_t seq_len, const std::vector<uint32_t>& ids) const;

  Config _cfg;
  std::unique_ptr<simaai::llima::Tokenizer> _tokenizer;
  const uint16_t* _embeddings = nullptr;  // (vocab, hidden) bfloat16, mmapped
  size_t _embeddings_bytes = 0;
  std::vector<float> _act_w_feats, _act_w_out, _act_b_out;
  std::map<uint32_t, std::unique_ptr<Graph>> _graphs;
  std::unique_ptr<EncoderGraph> _encoder_graph;
};

// JSON text the way Python's `json.dumps(value, ensure_ascii=False)` writes it (", " and ": "
// separators). The model was trained on that spelling, so structured states, instructions and
// criteria must be serialized this way and not compactly.
std::string python_json(const json& value);

Question parse_question(const json& qdef);
json decode_answer(const Config& cfg, const Question& question, const RawOutput& out);

}  // namespace laya

#endif
