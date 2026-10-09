// LiquidAI's d1 decision models on the Modalix MLA: d1-omni-600M and d1-3B.
//
// d1-omni
//
// d1 answers the same three kinds of question as Laya, about text or about a picture. A
// question is one row of tokens,
//
//   <bos> <state> state <q> instructions <opt> <mask> option_0 </opt> <opt> <mask> option_1 </opt> ... <decide>
//
// read once, in both directions, by an LFM2 trunk and a small decision head; the answer is the
// softmax over the scores at the row's markers. A picture is read by a SigLIP2 tower and goes
// in front of its row as a prefix of embeddings.
//
// The trunk and the head are compiled as a chain of graphs for each sequence length, the tower
// and its projector as a graph each (`d1_sima`). Everything else is here on the CPU:
// tokenizing, the layout of a row, the embedding lookup, the masks, and for a picture its
// resizing, its patches and the position embeddings resized to its grid.
//
// A pass costs the same however little of it is used, so it is used fully: a request's rows
// are laid end to end in one pass, each attending only to itself and with the convolution's
// taps cut at its ends, and come out as they would alone.
//
// d1-3B
//
// d1-3B is a causal language model, LFM2.5-VL-3B, and is asked as one: a question is a chat
// turn that stops where the answer would begin,
//
//   <bos><|im_start|>user\n[picture]state\n\n\nQUESTION:\nquestion and options<|im_end|>\n<|im_start|>assistant\n
//
// and the answer is read off the logits of the token that would come next, a softmax over the
// tokens that spell the options (`yes` and `no`, a digit, a letter) and over nothing else. The
// logits are the final hidden state against rows of the embedding table, so only the options'
// rows are computed, here on the CPU. A picture's embeddings take the places of the `<image>`
// tokens in its row. Its config says `"family": "lm"`; everything else is shared with d1-omni:
// the chains of graphs, the masks that lay rows end to end (causal ones here), the vision
// tower.
#ifndef LAYA_D1_HPP_
#define LAYA_D1_HPP_

#include <cstdint>
#include <filesystem>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace simaai::llima {
class Tokenizer;
}

namespace d1 {

// Ordered: the key order of a `choice` question's criteria is the option order.
using json = nlohmann::ordered_json;

struct Vision {
  uint32_t patches = 0, patch_size = 16, merge = 2, hidden_size = 0, position_grid = 16;
  uint64_t min_pixels = 0, max_pixels = 0;
  bool bicubic = false;  // how a picture is resized: d1-3B's processor uses the bicubic filter
  std::string elf, projector, position_embedding;
  std::vector<std::string> inputs;  // what the tower takes, in the MLA's order
  // What the projector takes: "merged", a block of merge x merge patches side by side, or the
  // block's patches apart as "merged0", "merged1", ..., where one input would be too wide.
  std::vector<std::string> projector_inputs;
};

struct Config {
  uint32_t hidden_size = 0, vocab_size = 0, text_limit = 0, image_text_limit = 0;
  std::string family = "omni";  // "omni": the bidirectional trunk and its head; "lm": the causal language model
  std::string model, precision, token_embeddings, type_embeddings, tokenizer;
  // lm: the tokens an answer is read from. Each option scores the best of its forms.
  std::vector<uint32_t> yes, no;
  std::vector<std::vector<uint32_t>> digits;
  std::map<std::string, std::vector<uint32_t>> letters;
  std::map<uint32_t, std::vector<std::string>> elfs;  // sequence length -> its graphs, in the order they run
  std::vector<std::vector<std::string>> inputs;       // what each graph of a chain takes, in order
  std::map<std::string, uint32_t> tokens;             // bos and the delimiters
  std::map<std::string, uint32_t> question_types;
  std::map<std::string, double> temperatures;
  std::optional<Vision> vision;
};

// A picture as pixels: rows of RGB, top to bottom.
struct Picture {
  uint32_t width = 0, height = 0;
  std::vector<uint8_t> rgb;
};

// One question over one state, as the trunk reads it.
struct Row {
  std::string type;
  std::vector<uint32_t> ids;
  std::vector<uint32_t> markers;  // omni: where each option's marker is, within `ids`
  std::vector<std::vector<uint32_t>> readout;  // lm: the tokens each option's logit is read from
  size_t dropped = 0;             // tokens of the state that did not fit
};

// What a request cost.
struct Cost {
  double tokenize_ms = 0, pre_ms = 0, mla_ms = 0, vision_ms = 0;
  size_t passes = 0, tokens = 0, dropped = 0, prefix = 0, pictures = 0, pictures_cached = 0;
  uint32_t seq_len = 0;  // the longest chain a pass ran on
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
  bool sees() const { return _tower != nullptr; }

  // A question over a state as token ids, within `max_len` tokens. `picture`: the row follows
  // a picture's prefix, where a yes-or-no question's options are worded as they were trained.
  // `picture_tokens` (lm): the positions each picture takes inside the row.
  Row encode(const json& state, const json& question, size_t max_len, bool picture,
             const std::vector<size_t>& picture_tokens = {}) const;

  // A picture's prefix embeddings, [position][hidden] in bfloat16: resized, cut into patches,
  // through the tower and the projector.
  std::vector<uint16_t> see(const Picture& picture, Cost& cost);

  // One pass: rows laid end to end on one chain (`seq_len`, or 0 for the shortest they fit),
  // and each row's scores, one an option. `prefix`: the pictures' embeddings, which go in
  // front of every row (omni) or in the places of its `<image>` tokens (lm).
  std::vector<std::vector<float>> pass(const std::vector<const Row*>& rows, const std::vector<uint16_t>& prefix, Cost& cost,
                                       uint32_t seq_len = 0);

  // Runs every loaded graph once, so the first request does not pay for cold caches, and notes
  // what a pass on each chain costs: that is what the packing is planned by.
  void warm_up();

  // {"answers": {...}, "usage": {...}}, in the shape Laya's runtime answers in. `pictures` are
  // read in the order given, in front of every question.
  json predict(const json& state, const json& questions, const std::vector<Picture>& pictures = {});

 private:
  struct Chain;
  struct Graph;
  Row encode_lm(const json& state, const json& question, size_t max_len, const std::vector<size_t>& picture_tokens) const;

  Config _cfg;
  std::unique_ptr<simaai::llima::Tokenizer> _tokenizer;
  const uint16_t* _embeddings = nullptr;  // (vocab, hidden) bfloat16, mmapped
  size_t _embeddings_bytes = 0;
  std::vector<std::vector<uint16_t>> _type_embeddings;  // a row a question type, bfloat16
  std::vector<float> _positions;                        // the tower's (grid * grid, hidden) position embeddings
  std::map<uint32_t, std::unique_ptr<Chain>> _chains;
  std::map<uint32_t, double> _pass_ms;  // what one pass on each chain takes, measured at warm-up
  std::unique_ptr<Graph> _tower, _projector;

  // The pictures seen last, by their bytes' hash: a picture asked about again costs nothing.
  struct Seen { uint64_t key; std::vector<uint16_t> prefix; };
  std::list<Seen> _seen;
  size_t _seen_capacity = 8;
};

// A picture from the bytes of a JPEG, PNG or other image file. Needs the runtime to have been
// built with OpenCV, which the DevKit has.
Picture decode_picture(const std::vector<uint8_t>& file);
std::vector<uint8_t> decode_base64(const std::string& text);

// A request's "images": each a base64 string of an image file, {"path": ...} of one on the
// board, or raw pixels as {"width", "height", "rgb": base64}.
std::vector<Picture> pictures_of(const json& images);

// The size (height, width) a picture is resized to: multiples of 32 pixels, holding between
// 64 and 256 merged patches.
std::pair<uint32_t, uint32_t> crop_size(uint32_t width, uint32_t height, const Vision& vision);

// Python's `json.dumps(value, ensure_ascii=False)`, which is how a state that is not text is
// written into the prompt.
std::string to_text(const json& value);

}  // namespace d1

#endif
