// d1-omni on the MLA: the trunk's chain of graphs, the vision tower, and the CPU work around
// them. `d1_sima/hostio.py` is this file's reference, in numpy.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>

#include <fmt/format.h>
#include <sima_lmm/mla_buffer.hpp>
#include <sima_lmm/mla_model.hpp>
#include <sima_lmm/tokenizer.hpp>

#ifdef LAYA_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#endif

#include "laya/d1.hpp"

namespace fs = std::filesystem;
namespace llima = simaai::llima;

namespace d1 {
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

float from_bf16(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// Large enough that exp() underflows to exactly 0, small enough to stay finite in bfloat16
// once an attention logit is added to it; see MASK_NEG in d1_sima/hostio.py.
constexpr float kMaskNeg = -30000.0f;

void evict_page_cache(const fs::path& path) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return;
  posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  close(fd);
}

std::vector<float> read_floats(const fs::path& path, size_t count) {
  std::ifstream stream(path, std::ios::binary);
  std::vector<float> values(count);
  if (!stream || !stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(float)))
    throw std::runtime_error(fmt::format("{}: expected {} floats", path.string(), count));
  return values;
}

Config load_config(const fs::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot read " + path.string());
  const json j = json::parse(stream);
  Config cfg;
  cfg.hidden_size = j.at("hidden_size");
  cfg.vocab_size = j.at("vocab_size");
  cfg.model = j.value("model", "d1");
  cfg.precision = j.value("precision", "");
  cfg.family = j.value("family", "omni");
  if (cfg.family != "omni" && cfg.family != "lm") throw std::runtime_error("d1_config.json: unknown family " + cfg.family);
  const bool lm = cfg.family == "lm";
  cfg.token_embeddings = j.at("token_embeddings");
  cfg.tokenizer = j.at("tokenizer");
  if (!lm) {
    cfg.type_embeddings = j.at("type_embeddings");
    cfg.text_limit = j.at("text_limit");
    cfg.image_text_limit = j.at("image_text_limit");
  }
  for (const auto& [seq_len, files] : j.at("elfs").items())
    cfg.elfs[static_cast<uint32_t>(std::stoul(seq_len))] = files.get<std::vector<std::string>>();
  if (cfg.elfs.empty()) throw std::runtime_error("d1_config.json lists no compiled graphs");
  cfg.inputs = j.at("inputs").get<std::vector<std::vector<std::string>>>();
  for (const auto& [name, id] : j.at("tokens").items()) cfg.tokens[name] = id;
  const std::vector<const char*> needed = lm ? std::vector<const char*>{"bos", "im_start", "im_end", "image", "image_start", "image_end"}
                                             : std::vector<const char*>{"bos", "state", "q", "opt", "opt_end", "decide", "marker"};
  for (const char* name : needed)
    if (!cfg.tokens.contains(name)) throw std::runtime_error(fmt::format("d1_config.json names no {} token", name));
  if (lm) {
    const json& readout = j.at("readout");
    cfg.yes = readout.at("yes").get<std::vector<uint32_t>>();
    cfg.no = readout.at("no").get<std::vector<uint32_t>>();
    cfg.digits = readout.at("digits").get<std::vector<std::vector<uint32_t>>>();
    for (const auto& [letter, ids] : readout.at("letters").items()) cfg.letters[letter] = ids.get<std::vector<uint32_t>>();
  } else {
    for (const auto& [name, index] : j.at("question_types").items()) cfg.question_types[name] = index;
    for (const auto& [name, value] : j.at("temperatures").items()) cfg.temperatures[name] = value;
  }
  if (j.contains("vision") && !j.at("vision").is_null()) {
    const json& v = j.at("vision");
    Vision vision;
    vision.patches = v.at("patches");
    vision.patch_size = v.at("patch_size");
    vision.merge = v.at("merge");
    vision.hidden_size = v.at("hidden_size");
    vision.position_grid = v.at("position_grid");
    vision.min_pixels = v.at("min_pixels");
    vision.max_pixels = v.at("max_pixels");
    vision.elf = v.at("elf");
    vision.projector = v.at("projector");
    vision.position_embedding = v.at("position_embedding");
    vision.bicubic = v.value("resample", "bilinear") == "bicubic";
    vision.projector_inputs = v.value("projector_inputs", std::vector<std::string>{"merged"});
    const size_t parts = vision.projector_inputs.size();
    if (parts != 1 && parts != size_t(vision.merge) * vision.merge)
      throw std::runtime_error("d1_config.json: the projector takes a merged block whole or its patches apart");
    for (size_t i = 0; parts > 1 && i < parts; ++i)
      if (std::count(vision.projector_inputs.begin(), vision.projector_inputs.end(), fmt::format("merged{}", i)) != 1)
        throw std::runtime_error("d1_config.json: the projector's inputs are not merged0, merged1, ...");
    vision.inputs = v.value("inputs", std::vector<std::string>{"patches", "positions", "mask"});
    for (const char* name : {"patches", "positions", "mask"})
      if (std::count(vision.inputs.begin(), vision.inputs.end(), name) != 1)
        throw std::runtime_error("d1_config.json: the vision tower's inputs are not patches, positions and mask");
    cfg.vision = vision;
  }
  return cfg;
}

bool is_blank(const json& value) {
  return value.is_null() || (value.is_string() && value.get<std::string>().empty());
}

double round4(double value) { return std::round(value * 1e4) / 1e4; }

// `<|name|>` -> `<¦name¦>`: text from a caller can never spell a delimiter or a marker.
std::string escape(const std::string& text) {
  static const std::regex special(R"(<\|([A-Za-z0-9_]+)\|>)");
  return std::regex_replace(text, special, "<\xC2\xA6$1\xC2\xA6>");
}

std::string plain(const json& value) { return value.is_string() ? value.get<std::string>() : to_text(value); }

// The options in the model's order. A noul is two options, false then true; after a picture
// they are worded `no` and `yes`, as the picture questions were trained.
std::vector<std::string> option_texts(const json& question, bool picture) {
  const std::string type = question.at("type").get<std::string>();
  const json none;
  const json& criteria = question.contains("criteria") ? question.at("criteria") : none;
  std::vector<std::string> options;
  if (type == "choice") {
    if (criteria.is_object()) {
      for (const auto& [key, description] : criteria.items())
        options.push_back(is_blank(description) ? key : key + ": " + plain(description));
    } else if (criteria.is_array()) {
      for (const auto& label : criteria) options.push_back(plain(label));
    }
    if (options.size() < 2) throw std::invalid_argument("a choice question needs \"criteria\" with at least two options");
  } else if (type == "score") {
    if (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10)
      throw std::invalid_argument("a score question needs \"criteria\": a list of 2 to 10 levels, lowest first");
    for (size_t level = 0; level < criteria.size(); ++level)
      options.push_back(fmt::format("level {}: {}", level, plain(criteria[level])));
  } else if (type == "noul") {
    const auto given = [&](const char* key, const char* other) -> json {
      if (!criteria.is_object()) return json();
      return criteria.contains(key) ? criteria.at(key) : criteria.contains(other) ? criteria.at(other) : json();
    };
    json no = given("false", "no"), yes = given("true", "yes");
    if (picture && (!criteria.is_object() || criteria.empty())) no = "no", yes = "yes";
    options.push_back("false: " + (is_blank(no) ? std::string("no, the statement does not hold") : plain(no)));
    options.push_back("true: " + (is_blank(yes) ? std::string("yes, the statement holds") : plain(yes)));
  } else {
    throw std::invalid_argument("unknown question type: " + type);
  }
  return options;
}

// Bilinear resampling with antialiasing along one axis: for each output sample, the first
// input sample it reads and the weights. A triangle as wide as the scale when shrinking, as
// PIL and torch's interpolate(antialias=True) use.
struct Taps { size_t low; std::vector<float> weights; };

// The bicubic kernel PIL and torchvision resample with (a = -0.5).
double cubic(double x) {
  x = std::abs(x);
  return x < 1 ? (1.5 * x - 2.5) * x * x + 1 : x < 2 ? ((-0.5 * x + 2.5) * x - 4) * x + 2 : 0.0;
}

std::vector<Taps> filter(size_t size_in, size_t size_out, bool bicubic = false) {
  const double scale = double(size_in) / double(size_out), stretch = std::max(scale, 1.0), support = (bicubic ? 2.0 : 1.0) * stretch;
  std::vector<Taps> taps(size_out);
  for (size_t i = 0; i < size_out; ++i) {
    const double centre = (i + 0.5) * scale;
    const size_t low = static_cast<size_t>(std::max(0.0, std::floor(centre - support + 0.5)));
    const size_t high = std::min(size_in, static_cast<size_t>(std::max(0.0, std::floor(centre + support + 0.5))));
    std::vector<double> weights;
    double sum = 0;
    for (size_t j = low; j < high; ++j) {
      const double x = (double(j) - centre + 0.5) / stretch;
      weights.push_back(bicubic ? cubic(x) : std::max(0.0, 1.0 - std::abs(x)));
      sum += weights.back();
    }
    taps[i].low = low;
    for (double w : weights) taps[i].weights.push_back(static_cast<float>(w / sum));
  }
  return taps;
}

// (H, W, C) -> (h, w, C), across then down, in floating point.
template <typename T>
std::vector<float> resize(const T* in, size_t H, size_t W, size_t C, size_t h, size_t w, bool bicubic = false) {
  const auto across = filter(W, w, bicubic), down = filter(H, h, bicubic);
  std::vector<float> wide(H * w * C, 0.0f);
  for (size_t y = 0; y < H; ++y) {
    const T* row = in + y * W * C;
    float* out = wide.data() + y * w * C;
    for (size_t x = 0; x < w; ++x) {
      const Taps& t = across[x];
      for (size_t k = 0; k < t.weights.size(); ++k) {
        const T* pixel = row + (t.low + k) * C;
        const float weight = t.weights[k];
        for (size_t c = 0; c < C; ++c) out[x * C + c] += weight * float(pixel[c]);
      }
    }
  }
  std::vector<float> result(h * w * C, 0.0f);
  for (size_t y = 0; y < h; ++y) {
    const Taps& t = down[y];
    float* out = result.data() + y * w * C;
    for (size_t k = 0; k < t.weights.size(); ++k) {
      const float* row = wide.data() + (t.low + k) * w * C;
      const float weight = t.weights[k];
      for (size_t i = 0; i < w * C; ++i) out[i] += weight * row[i];
    }
  }
  return result;
}

uint64_t hash_bytes(const std::vector<uint8_t>& bytes, uint64_t seed) {
  uint64_t value = 1469598103934665603ull ^ seed;
  for (uint8_t byte : bytes) value = (value ^ byte) * 1099511628211ull;
  return value;
}

// The MLA dispatcher is process-wide; hold it for as long as any Runtime exists.
int g_runtimes = 0;

}  // namespace

std::string to_text(const json& value) {
  if (value.is_object()) {
    std::string out = "{";
    for (auto it = value.begin(); it != value.end(); ++it)
      out += (it == value.begin() ? "" : ", ") + json(it.key()).dump() + ": " + to_text(it.value());
    return out + "}";
  }
  if (value.is_array()) {
    std::string out = "[";
    for (auto it = value.begin(); it != value.end(); ++it) out += (it == value.begin() ? "" : ", ") + to_text(*it);
    return out + "]";
  }
  return value.dump();
}

std::pair<uint32_t, uint32_t> crop_size(uint32_t width, uint32_t height, const Vision& vision) {
  if (!width || !height) throw std::invalid_argument("a picture is empty");
  const double step = vision.patch_size * vision.merge, W = width, H = height;
  // nearbyint rounds halves to even, as Python's round does.
  double h = std::max(step, std::nearbyint(H / step) * step), w = std::max(step, std::nearbyint(W / step) * step);
  if (h * w > double(vision.max_pixels)) {
    const double beta = std::sqrt(H * W / double(vision.max_pixels));
    h = std::max(step, std::floor(H / beta / step) * step);
    w = std::max(step, std::floor(W / beta / step) * step);
  } else if (h * w < double(vision.min_pixels)) {
    const double beta = std::sqrt(double(vision.min_pixels) / (H * W));
    h = std::ceil(H * beta / step) * step;
    w = std::ceil(W * beta / step) * step;
  }
  return {static_cast<uint32_t>(h), static_cast<uint32_t>(w)};
}

std::vector<uint8_t> decode_base64(const std::string& text) {
  static const auto table = [] {
    std::array<int8_t, 256> t;
    t.fill(-1);
    const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) t[static_cast<uint8_t>(alphabet[i])] = static_cast<int8_t>(i);
    t[static_cast<uint8_t>('-')] = 62;  // the URL-safe alphabet too
    t[static_cast<uint8_t>('_')] = 63;
    return t;
  }();
  // A data URL's header, if there is one, ends at the comma.
  size_t start = 0;
  if (text.rfind("data:", 0) == 0) {
    start = text.find(',');
    if (start == std::string::npos) throw std::invalid_argument("a data URL with no data");
    ++start;
  }
  std::vector<uint8_t> out;
  out.reserve((text.size() - start) * 3 / 4);
  uint32_t bits = 0;
  int have = 0;
  for (size_t i = start; i < text.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
    if (table[c] < 0) throw std::invalid_argument("a picture is not base64");
    bits = (bits << 6) | static_cast<uint32_t>(table[c]);
    if ((have += 6) >= 8) out.push_back(static_cast<uint8_t>(bits >> (have -= 8)));
  }
  return out;
}

Picture decode_picture(const std::vector<uint8_t>& file) {
#ifdef LAYA_HAVE_OPENCV
  if (file.empty()) throw std::invalid_argument("a picture is empty");
  const cv::Mat encoded(1, static_cast<int>(file.size()), CV_8UC1, const_cast<uint8_t*>(file.data()));
  const cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_COLOR);   // BGR, turned as its EXIF says
  if (decoded.empty()) throw std::invalid_argument("a picture could not be decoded: send a JPEG or a PNG");
  Picture picture;
  picture.width = static_cast<uint32_t>(decoded.cols);
  picture.height = static_cast<uint32_t>(decoded.rows);
  picture.rgb.resize(size_t(picture.width) * picture.height * 3);
  for (int y = 0; y < decoded.rows; ++y) {
    const uint8_t* row = decoded.ptr<uint8_t>(y);
    uint8_t* out = picture.rgb.data() + size_t(y) * picture.width * 3;
    for (int x = 0; x < decoded.cols; ++x) {
      out[x * 3] = row[x * 3 + 2];
      out[x * 3 + 1] = row[x * 3 + 1];
      out[x * 3 + 2] = row[x * 3];
    }
  }
  return picture;
#else
  (void)file;
  throw std::runtime_error("this runtime was built without OpenCV and cannot decode a picture; send raw pixels");
#endif
}

std::vector<Picture> pictures_of(const json& images) {
  std::vector<Picture> pictures;
  if (images.is_null()) return pictures;
  for (const auto& image : images.is_array() ? images : json::array({images})) {
    if (image.is_string()) {
      pictures.push_back(decode_picture(decode_base64(image.get<std::string>())));
    } else if (image.is_object() && image.contains("rgb")) {
      Picture picture;
      picture.width = image.at("width");
      picture.height = image.at("height");
      picture.rgb = decode_base64(image.at("rgb").get<std::string>());
      if (picture.rgb.size() != size_t(picture.width) * picture.height * 3)
        throw std::invalid_argument("a picture's pixels are not width x height x 3 bytes");
      pictures.push_back(std::move(picture));
    } else if (image.is_object() && image.contains("path")) {
      std::ifstream stream(image.at("path").get<std::string>(), std::ios::binary);
      if (!stream) throw std::invalid_argument("cannot read the picture " + image.at("path").get<std::string>());
      const std::vector<uint8_t> file((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
      pictures.push_back(decode_picture(file));
    } else {
      throw std::invalid_argument("a picture is a base64 string, {\"path\": ...} or {\"width\", \"height\", \"rgb\"}");
    }
  }
  return pictures;
}

// One compiled graph with buffers of its own: the tower, the projector.
struct Runtime::Graph {
  std::vector<std::unique_ptr<llima::MLABuffer>> inputs;
  std::unique_ptr<llima::MLABuffer> output;
  std::unique_ptr<llima::MLAModelWithBuffer> model;

  Graph(const fs::path& elf, const std::string& name, const std::vector<std::vector<size_t>>& in_shapes,
        const std::vector<size_t>& out_shape) {
    std::vector<llima::MLABufferSlice> slices;
    for (size_t i = 0; i < in_shapes.size(); ++i) {
      inputs.push_back(std::make_unique<llima::MLABuffer>(fmt::format("d1_{}_in{}", name, i), in_shapes[i], "bfloat16", true));
      inputs.back()->allocate();
      inputs.back()->clear();
      slices.emplace_back(inputs.back().get());
    }
    output = std::make_unique<llima::MLABuffer>(fmt::format("d1_{}_out", name), out_shape, "bfloat16", true);
    output->allocate();
    output->clear();
    model = std::make_unique<llima::MLAModelWithBuffer>(elf, slices, std::vector<llima::MLABufferSlice>{llima::MLABufferSlice(output.get())});
    evict_page_cache(elf);   // also before: a previous run may have left it cached
    model->load();
    evict_page_cache(elf);
  }
  ~Graph() { model->free(); }
};

// One sequence length: its graphs and the buffers between them. Graph i reads buffer i and
// writes buffer i + 1, so nothing is copied on the way; the last buffer is the scorer's output,
// one value a position. Buffers are token-major, [position][channel]; a mask is therefore
// [query][key], the transpose of the graph's NCHW (1, key, 1, query).
struct Runtime::Chain {
  uint32_t seq_len;
  size_t width;
  bool causal;
  std::vector<std::unique_ptr<llima::MLABuffer>> buffers;
  llima::MLABuffer mask, head_mask, conv_left, conv_right, type_add;
  std::vector<std::unique_ptr<llima::MLAModelWithBuffer>> models;
  std::vector<std::pair<size_t, size_t>> masked_for;  // the rows (prefix, text) the masks currently hold

  Chain(const fs::path& dir, const Config& cfg, uint32_t s, const std::vector<std::string>& elfs)
      : seq_len(s),
        width(cfg.hidden_size),
        causal(cfg.family == "lm"),
        mask(fmt::format("d1_mask_{}", s), {s, s}, "bfloat16", true),
        head_mask(fmt::format("d1_head_mask_{}", s), {s, s}, "bfloat16", true),
        // The taps' masks are a value a position, repeated over the channels: the MLA compiler
        // takes no input of one channel.
        conv_left(fmt::format("d1_conv_left_{}", s), {s, cfg.hidden_size}, "bfloat16", true),
        conv_right(fmt::format("d1_conv_right_{}", s), {s, cfg.hidden_size}, "bfloat16", true),
        type_add(fmt::format("d1_type_add_{}", s), {s, cfg.hidden_size}, "bfloat16", true) {
    if (cfg.inputs.size() != elfs.size()) throw std::runtime_error("d1_config.json: \"inputs\" does not match the graphs");
    for (auto* buffer : {&mask, &head_mask, &conv_left, &conv_right, &type_add}) {
      buffer->allocate();
      buffer->clear();
    }
    for (size_t i = 0; i <= elfs.size(); ++i) {
      // The last buffer is the scorer's output, one value a position, or for the language
      // model the final hidden state.
      const size_t width = i == elfs.size() && !causal ? 1 : cfg.hidden_size;
      buffers.push_back(std::make_unique<llima::MLABuffer>(
          fmt::format("d1_hidden_{}_{}", s, i), std::vector<size_t>{s, width}, "bfloat16", true));
      buffers.back()->allocate();
      buffers.back()->clear();
    }
    for (size_t i = 0; i < elfs.size(); ++i) {
      std::vector<llima::MLABufferSlice> inputs;
      for (const std::string& name : cfg.inputs[i]) {
        llima::MLABuffer* buffer = name == "hidden_in" ? buffers[i].get() : name == "mask" ? &mask
            : name == "head_mask" ? &head_mask : name == "conv_left" || name == "conv_back1" ? &conv_left
            : name == "conv_right" || name == "conv_back2" ? &conv_right : name == "type_add" ? &type_add : nullptr;
        if (!buffer) throw std::runtime_error("d1_config.json names a graph input this runtime does not know: " + name);
        inputs.emplace_back(buffer);
      }
      const fs::path elf = dir / elfs[i];
      models.push_back(std::make_unique<llima::MLAModelWithBuffer>(
          elf, inputs, std::vector<llima::MLABufferSlice>{llima::MLABufferSlice(buffers[i + 1].get())}));
      evict_page_cache(elf);
      models.back()->load();
      evict_page_cache(elf);
    }
  }
  ~Chain() { for (auto& model : models) model->free(); }

  // Rows (prefix, text) laid end to end. A text position reads its whole row and a prefix
  // position the prefix only; the head runs over a row's text alone; the convolution's taps
  // stop at a row's two ends, and the prefix's last position never reads the text after it. A
  // position past the last row sees itself only; nothing reads its output.
  void set_masks(const std::vector<std::pair<size_t, size_t>>& rows) {
    if (rows == masked_for) return;
    const uint16_t open = to_bf16(0.0f), closed = to_bf16(kMaskNeg), one = to_bf16(1.0f), zero = to_bf16(0.0f);
    const size_t s = seq_len;
    std::vector<uint16_t> trunk(s * s, closed), head(s * s, closed), left(s * width, zero), right(s * width, zero);
    for (size_t pos = 0; pos < s; ++pos) trunk[pos * s + pos] = head[pos * s + pos] = open;
    size_t start = 0;
    if (causal) {
      // The language model: a query sees the keys of its own row at or before it, and the two
      // taps that look back (one position, `left` here, and two, `right`) stop at a row's start.
      for (const auto& [prefix, text] : rows) {
        const size_t end = start + prefix + text;
        for (size_t query = start; query < end; ++query)
          for (size_t key = start; key <= query; ++key) trunk[query * s + key] = open;
        for (size_t pos = start; pos + 1 < end; ++pos) std::fill_n(left.begin() + pos * width, width, one);
        for (size_t pos = start; pos + 2 < end; ++pos) std::fill_n(right.begin() + pos * width, width, one);
        start = end;
      }
      mask.upload(trunk.data());
      conv_left.upload(left.data());
      conv_right.upload(right.data());
      masked_for = rows;
      return;
    }
    for (const auto& [prefix, text] : rows) {
      const size_t words = start + prefix, end = words + text;
      for (size_t query = start; query < words; ++query)
        for (size_t key = start; key < words; ++key) trunk[query * s + key] = open;
      for (size_t query = words; query < end; ++query) {
        for (size_t key = start; key < end; ++key) trunk[query * s + key] = open;
        for (size_t key = words; key < end; ++key) head[query * s + key] = open;
      }
      for (size_t pos = start; pos + 1 < end; ++pos) std::fill_n(left.begin() + pos * width, width, one);
      for (size_t pos = start + 1; pos < end; ++pos)
        if (!prefix || pos != words) std::fill_n(right.begin() + pos * width, width, one);
      start = end;
    }
    mask.upload(trunk.data());
    head_mask.upload(head.data());
    conv_left.upload(left.data());
    conv_right.upload(right.data());
    masked_for = rows;
  }
};

Runtime::Runtime(const fs::path& model_dir, const std::vector<uint32_t>& seq_lens)
    : _cfg(load_config(model_dir / "d1_config.json")) {
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
  void* mapped = mmap(nullptr, _embeddings_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (mapped == MAP_FAILED) throw std::runtime_error("cannot map " + table.string());
  _embeddings = static_cast<const uint16_t*>(mapped);

  const size_t types = _cfg.question_types.size();
  const std::vector<float> type_table = types ? read_floats(model_dir / _cfg.type_embeddings, types * _cfg.hidden_size) : std::vector<float>();
  for (size_t type = 0; type < types; ++type) {
    _type_embeddings.emplace_back(_cfg.hidden_size);
    std::transform(type_table.begin() + type * _cfg.hidden_size, type_table.begin() + (type + 1) * _cfg.hidden_size,
                   _type_embeddings.back().begin(), to_bf16);
  }

  if (g_runtimes++ == 0) llima::connect_mla_rt({});
  for (const auto& [seq_len, elfs] : _cfg.elfs) {
    if (!seq_lens.empty() && std::find(seq_lens.begin(), seq_lens.end(), seq_len) == seq_lens.end()) continue;
    _chains[seq_len] = std::make_unique<Chain>(model_dir, _cfg, seq_len, elfs);
  }
  if (_chains.empty()) throw std::runtime_error("no compiled chain was loaded from " + model_dir.string());
  if (_cfg.vision) {
    const Vision& v = *_cfg.vision;
    const size_t n = v.patches, m = n / (v.merge * v.merge), features = size_t(3) * v.patch_size * v.patch_size;
    _positions = read_floats(model_dir / v.position_embedding, size_t(v.position_grid) * v.position_grid * v.hidden_size);
    std::vector<std::vector<size_t>> shapes;   // in the order the tower takes them
    for (const std::string& name : v.inputs)
      shapes.push_back(name == "patches" ? std::vector<size_t>{n, features} : name == "positions" ? std::vector<size_t>{n, v.hidden_size}
                                                                                                    : std::vector<size_t>{n, n});
    _tower = std::make_unique<Graph>(model_dir / v.elf, "tower", shapes, std::vector<size_t>{n, v.hidden_size});
    const size_t parts = v.projector_inputs.size();
    _projector = std::make_unique<Graph>(model_dir / v.projector, "projector",
                                         std::vector<std::vector<size_t>>(parts, {m, size_t(v.hidden_size) * v.merge * v.merge / parts}),
                                         std::vector<size_t>{m, _cfg.hidden_size});
  }
}

Runtime::~Runtime() {
  _chains.clear();
  _tower.reset();
  _projector.reset();
  if (--g_runtimes == 0) llima::disconnect_mla_rt();
  if (_embeddings) munmap(const_cast<uint16_t*>(_embeddings), _embeddings_bytes);
}

std::vector<uint32_t> Runtime::seq_lens() const {
  std::vector<uint32_t> result;
  for (const auto& entry : _chains) result.push_back(entry.first);
  return result;
}

uint32_t Runtime::max_tokens() const { return _chains.rbegin()->first; }

// d1-3B's prompt, in the checkpoint's own wording (its prompt.py). The text between two special
// tokens is tokenized on its own, as a tokenizer does that meets special tokens in a text.
Row Runtime::encode_lm(const json& state, const json& question, size_t max_len, const std::vector<size_t>& picture_tokens) const {
  const auto ids_of = [&](const std::string& text) { return _tokenizer->encode(text, false); };
  const auto token = [&](const char* name) { return _cfg.tokens.at(name); };
  Row row;
  row.type = question.at("type").get<std::string>();
  const json none;
  const json& criteria = question.contains("criteria") ? question.at("criteria") : none;
  const std::string instructions = question.contains("instructions") && !question.at("instructions").is_null() ? plain(question.at("instructions")) : "";
  std::string asked;
  if (row.type == "choice") {
    std::vector<std::pair<std::string, std::string>> options;   // label, what is written for it
    if (criteria.is_object()) {
      for (const auto& [label, description] : criteria.items()) {
        std::string text = is_blank(description) ? label : plain(description);
        if (is_blank(description)) std::replace(text.begin(), text.end(), '_', ' ');
        options.emplace_back(label, text);
      }
    } else if (criteria.is_array()) {
      for (const auto& label : criteria) options.emplace_back(plain(label), plain(label));
    }
    if (options.size() < 2) throw std::invalid_argument("a choice question needs \"criteria\" with at least two options");
    if (options.size() > 26) throw std::invalid_argument("a choice question has at most 26 options on this model");
    // An option is answered by a code: its label when the labels already are single letters, else A, B, C...
    bool lettered = true;
    for (const auto& option : options) lettered = lettered && option.first.size() == 1 && std::isalpha(static_cast<unsigned char>(option.first[0]));
    asked = instructions + "\n\nOptions:";
    for (size_t i = 0; i < options.size(); ++i) {
      const std::string code = lettered ? options[i].first : std::string(1, char('A' + i));
      asked += "\n" + code + " " + options[i].second;
      row.readout.push_back(_cfg.letters.at(code));
    }
    asked += "\n\nReply with the option code only.";
  } else if (row.type == "score") {
    if (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10)
      throw std::invalid_argument("a score question needs \"criteria\": a list of 2 to 10 levels, lowest first");
    asked = instructions + "\n\n";
    for (size_t level = 0; level < criteria.size(); ++level) {
      asked += fmt::format("{} {}\n", level, plain(criteria[level]));
      row.readout.push_back(_cfg.digits.at(level));
    }
    asked += fmt::format("\nReply with a single digit 0-{} only.", criteria.size() - 1);
  } else if (row.type == "noul") {
    asked = instructions;
    if (criteria.is_object() && !criteria.empty()) {
      const auto given = [&](const char* key, const char* other) {
        return criteria.contains(key) ? plain(criteria.at(key)) : criteria.contains(other) ? plain(criteria.at(other)) : std::string("None");
      };
      asked += "\nYes: " + given("true", "yes") + "\nNo: " + given("false", "no");
    }
    asked += "\n\nReply with yes or no only.";
    row.readout = {_cfg.yes, _cfg.no};
  } else {
    throw std::invalid_argument("unknown question type: " + row.type);
  }

  // A state is written as it is when it is text, and as indented JSON when it is not.
  const bool stated = !state.is_null();
  const std::string said = !stated ? "" : state.is_string() ? state.get<std::string>() : state.dump(2);
  const std::string between = stated ? "\n\n\nQUESTION:\n" : "";
  const auto build = [&](const std::vector<uint32_t>* cut) {
    std::vector<uint32_t> ids{token("bos"), token("im_start")};
    const auto add = [&](const std::vector<uint32_t>& more) { ids.insert(ids.end(), more.begin(), more.end()); };
    std::string open = "user\n";
    if (!picture_tokens.empty()) {
      add(ids_of(open));
      open.clear();
      for (size_t count : picture_tokens) {
        ids.push_back(token("image_start"));
        ids.insert(ids.end(), count, token("image"));
        ids.push_back(token("image_end"));
      }
    }
    if (cut) {                 // a state too long to fit: what is kept of it, then the rest
      if (!open.empty()) add(ids_of(open));
      add(*cut);
      add(ids_of(between + asked));
    } else {
      add(ids_of(open + said + between + asked));
    }
    ids.push_back(token("im_end"));
    add(ids_of("\n"));
    ids.push_back(token("im_start"));
    add(ids_of("assistant\n"));
    return ids;
  };
  row.ids = build(nullptr);
  if (row.ids.size() > max_len) {
    std::vector<uint32_t> words = ids_of(said);
    const size_t over = row.ids.size() - max_len + 2;      // two to spare: the pieces may tokenize a little longer apart
    if (words.size() <= over) throw std::invalid_argument("the question and its options do not fit in the context");
    row.dropped = over;
    words.resize(words.size() - over);
    row.ids = build(&words);
    if (row.ids.size() > max_len) throw std::invalid_argument("the question and its options do not fit in the context");
  }
  return row;
}

Row Runtime::encode(const json& state, const json& question, size_t max_len, bool picture, const std::vector<size_t>& picture_tokens) const {
  if (_cfg.family == "lm") return encode_lm(state, question, max_len, picture_tokens);
  const auto ids_of = [&](const std::string& text) { return _tokenizer->encode(escape(text), false); };
  const auto token = [&](const char* name) { return _cfg.tokens.at(name); };
  Row row;
  row.type = question.at("type").get<std::string>();
  const std::vector<std::string> options = option_texts(question, picture);
  // The options get max(96, min(24 an option + 32, half the room)) tokens, shared out evenly.
  const size_t budget = std::max<size_t>(96, std::min(options.size() * 24 + 32, max_len / 2));
  const size_t each = std::max<size_t>(2, (budget - std::min(budget, 3 * options.size())) / options.size());
  std::vector<uint32_t> tail{token("q")};
  const json none;
  const json& instructions = question.contains("instructions") ? question.at("instructions") : none;
  for (uint32_t id : ids_of(instructions.is_null() ? std::string() : plain(instructions))) tail.push_back(id);
  if (tail.size() > std::max<size_t>(16, budget)) tail.resize(std::max<size_t>(16, budget));
  std::vector<uint32_t> markers;
  for (const std::string& text : options) {
    markers.push_back(static_cast<uint32_t>(tail.size() + 1));
    tail.push_back(token("opt"));
    tail.push_back(token("marker"));
    std::vector<uint32_t> words = ids_of(" " + text);
    if (words.size() > each) words.resize(each);
    tail.insert(tail.end(), words.begin(), words.end());
    tail.push_back(token("opt_end"));
  }
  tail.push_back(token("decide"));
  // The state is cut on the right to the room that is left.
  const size_t room = max_len > tail.size() + 2 ? max_len - tail.size() - 2 : 0;
  std::vector<uint32_t> words = ids_of(state.is_null() ? std::string() : plain(state));
  if (words.size() > room) {
    row.dropped = words.size() - room;
    words.resize(room);
  }
  row.ids.push_back(token("bos"));
  row.ids.push_back(token("state"));
  row.ids.insert(row.ids.end(), words.begin(), words.end());
  const size_t lead = row.ids.size();
  row.ids.insert(row.ids.end(), tail.begin(), tail.end());
  if (row.ids.size() > max_len) row.ids.resize(max_len);
  for (uint32_t marker : markers) row.markers.push_back(static_cast<uint32_t>(marker + lead));
  if (row.markers.back() >= max_len) throw std::invalid_argument("the question and its options do not fit in the context");
  return row;
}

std::vector<uint16_t> Runtime::see(const Picture& picture, Cost& cost) {
  if (!_tower) throw std::invalid_argument("this model was compiled without its vision tower: it reads text only");
  if (picture.rgb.size() != size_t(picture.width) * picture.height * 3) throw std::invalid_argument("a picture's pixels are missing");
  cost.pictures++;
  const uint64_t key = hash_bytes(picture.rgb, (uint64_t(picture.width) << 32) | picture.height);
  for (auto it = _seen.begin(); it != _seen.end(); ++it) {
    if (it->key != key) continue;
    _seen.splice(_seen.end(), _seen, it);
    cost.pictures_cached++;
    return _seen.back().prefix;
  }
  const Vision& v = *_cfg.vision;
  const auto input = [&](const char* name) -> llima::MLABuffer& {
    return *_tower->inputs[std::find(v.inputs.begin(), v.inputs.end(), name) - v.inputs.begin()];
  };
  auto stage = Clock::now();
  const auto [h, w] = crop_size(picture.width, picture.height, v);
  const size_t ph = h / v.patch_size, pw = w / v.patch_size, n = ph * pw, hidden = v.hidden_size;
  const size_t features = size_t(3) * v.patch_size * v.patch_size, merge = v.merge;
  if (n > v.patches) throw std::runtime_error(fmt::format("a picture's {} patches do not fit the tower's {}", n, v.patches));
  const std::vector<float> resized = resize(picture.rgb.data(), picture.height, picture.width, 3, h, w, v.bicubic);

  // A patch is flattened row, column, colour, with its pixels in [-1, 1].
  auto* patches = static_cast<uint16_t*>(input("patches").get_virtual_addr());
  std::memset(patches, 0, size_t(v.patches) * features * sizeof(uint16_t));
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      const size_t patch = (y / v.patch_size) * pw + x / v.patch_size;
      const size_t inside = ((y % v.patch_size) * v.patch_size + x % v.patch_size) * 3;
      for (size_t c = 0; c < 3; ++c) {
        const float value = std::clamp(std::floor(resized[(y * w + x) * 3 + c] + 0.5f), 0.0f, 255.0f);
        patches[patch * features + inside + c] = to_bf16((value - 127.5f) / 127.5f);
      }
    }
  }
  input("patches").flush_cache();
  // The position embeddings, resized to this picture's grid.
  const std::vector<float> placed = resize(_positions.data(), v.position_grid, v.position_grid, hidden, ph, pw);
  std::vector<uint16_t> positions(size_t(v.patches) * hidden, 0);
  std::transform(placed.begin(), placed.end(), positions.begin(), to_bf16);
  input("positions").upload(positions.data());
  const uint16_t open = to_bf16(0.0f), closed = to_bf16(kMaskNeg);
  std::vector<uint16_t> mask(size_t(v.patches) * v.patches, closed);
  for (size_t query = 0; query < v.patches; ++query) {
    mask[query * v.patches + query] = open;
    if (query < n) std::fill_n(mask.begin() + query * v.patches, n, open);
  }
  input("mask").upload(mask.data());
  cost.pre_ms += elapsed_ms(stage);

  stage = Clock::now();
  _tower->model->run();
  // 2x2 blocks of patches side by side: top left, top right, bottom left, bottom right.
  _tower->output->invalidate_cache();
  const auto* seen = static_cast<const uint16_t*>(_tower->output->get_virtual_addr());
  const size_t rows = ph / merge, columns = pw / merge, m = rows * columns, slots = v.patches / (merge * merge);
  // In one input, or a patch of the block an input: then input i is the one named merged<i>.
  const size_t parts = v.projector_inputs.size(), wide = hidden * merge * merge / parts;
  std::vector<uint16_t*> merged(merge * merge);
  for (size_t part = 0; part < merge * merge; ++part) {
    const size_t at = parts == 1 ? 0 : std::find(v.projector_inputs.begin(), v.projector_inputs.end(), fmt::format("merged{}", part)) - v.projector_inputs.begin();
    merged[part] = static_cast<uint16_t*>(_projector->inputs[at]->get_virtual_addr()) + (parts == 1 ? part * hidden : 0);
  }
  for (auto& input : _projector->inputs) std::memset(input->get_virtual_addr(), 0, slots * wide * sizeof(uint16_t));
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < columns; ++c)
      for (size_t dy = 0; dy < merge; ++dy)
        for (size_t dx = 0; dx < merge; ++dx)
          std::memcpy(merged[dy * merge + dx] + (r * columns + c) * wide,
                      seen + ((r * merge + dy) * pw + c * merge + dx) * hidden, hidden * sizeof(uint16_t));
  for (auto& input : _projector->inputs) input->flush_cache();
  _projector->model->run();
  _projector->output->invalidate_cache();
  const auto* out = static_cast<const uint16_t*>(_projector->output->get_virtual_addr());
  std::vector<uint16_t> prefix(out, out + m * _cfg.hidden_size);
  cost.vision_ms += elapsed_ms(stage);

  while (_seen.size() >= _seen_capacity) _seen.pop_front();
  _seen.push_back({key, prefix});
  return prefix;
}

std::vector<std::vector<float>> Runtime::pass(const std::vector<const Row*>& rows, const std::vector<uint16_t>& prefix, Cost& cost,
                                              uint32_t seq_len) {
  const uint32_t width = _cfg.hidden_size;
  const bool lm = _cfg.family == "lm";
  // The language model's pictures are inside its rows, in the places of their `<image>` tokens.
  const size_t lead = lm ? 0 : prefix.size() / width;
  const uint32_t image = lm ? _cfg.tokens.at("image") : 0;
  size_t total = 0;
  std::vector<std::pair<size_t, size_t>> sizes;
  for (const Row* row : rows) {
    if (row->ids.empty()) throw std::invalid_argument("a row to read is empty");
    sizes.emplace_back(lead, row->ids.size());
    total += lead + row->ids.size();
  }
  const auto found = seq_len ? _chains.find(seq_len) : _chains.lower_bound(static_cast<uint32_t>(total));
  if (found == _chains.end() || total > found->first)
    throw std::invalid_argument(seq_len ? fmt::format("{} positions do not fit a loaded {}-token chain", total, seq_len)
                                        : fmt::format("{} positions exceed the longest loaded chain ({})", total, max_tokens()));
  Chain& chain = *found->second;

  auto stage = Clock::now();
  llima::MLABuffer& first = *chain.buffers.front();
  auto* hidden = static_cast<uint16_t*>(first.get_virtual_addr());
  auto* types = static_cast<uint16_t*>(chain.type_add.get_virtual_addr());
  std::memset(types, 0, size_t(chain.seq_len) * width * sizeof(uint16_t));
  size_t pos = 0;
  for (const Row* row : rows) {
    const auto type = _cfg.question_types.find(row->type);
    if (!lm && type == _cfg.question_types.end()) throw std::invalid_argument("unknown question type: " + row->type);
    if (lead) std::memcpy(hidden + pos * width, prefix.data(), prefix.size() * sizeof(uint16_t));
    pos += lead;
    size_t seen = 0;           // lm: how many of the pictures' embeddings this row has taken
    for (uint32_t id : row->ids) {
      if (id >= _cfg.vocab_size) throw std::invalid_argument(fmt::format("token id {} out of range", id));
      if (lm && id == image && !prefix.empty()) {
        if ((seen + 1) * width > prefix.size()) throw std::invalid_argument("a row has more picture positions than there are picture embeddings");
        std::memcpy(hidden + pos * width, prefix.data() + seen++ * width, width * sizeof(uint16_t));
      } else {
        std::memcpy(hidden + pos * width, _embeddings + size_t(id) * width, width * sizeof(uint16_t));
      }
      if (!lm) std::memcpy(types + pos * width, _type_embeddings[type->second].data(), width * sizeof(uint16_t));
      ++pos;
    }
  }
  // What is left of the last pass must not reach this one through a cut tap as a NaN times 0.
  std::memset(hidden + pos * width, 0, (chain.seq_len - pos) * width * sizeof(uint16_t));
  first.flush_cache();
  chain.type_add.flush_cache();
  chain.set_masks(sizes);
  cost.pre_ms += elapsed_ms(stage);

  stage = Clock::now();
  for (auto& model : chain.models) model->run();
  cost.mla_ms += elapsed_ms(stage);
  cost.passes++;
  cost.seq_len = std::max(cost.seq_len, chain.seq_len);

  std::vector<std::vector<float>> result;
  if (lm) {
    // An option's logit: the final hidden state at the row's last token against the embedding
    // of each token that spells the option, the best of them. (The embedding table is the
    // language model's output layer too.)
    const llima::MLABuffer& last = *chain.buffers.back();
    last.invalidate_cache();
    const auto* out = static_cast<const uint16_t*>(last.get_virtual_addr());
    std::vector<float> state(width);
    size_t end = 0;
    for (const Row* row : rows) {
      end += row->ids.size();
      std::transform(out + (end - 1) * width, out + end * width, state.begin(), from_bf16);
      result.emplace_back();
      for (const auto& group : row->readout) {
        double best = -1e30;
        for (uint32_t id : group) {
          const uint16_t* weights = _embeddings + size_t(id) * width;
          double logit = 0;
          for (size_t c = 0; c < width; ++c) logit += double(from_bf16(weights[c])) * state[c];
          best = std::max(best, logit);
        }
        result.back().push_back(static_cast<float>(best));
      }
    }
    return result;
  }
  // Downloaded, not read in place: a buffer of one channel is laid out on the MLA with its
  // channels padded out, and `download` is what undoes that.
  std::vector<uint16_t> scores(chain.seq_len);
  chain.buffers.back()->download(scores.data());
  size_t start = 0;
  for (const Row* row : rows) {
    result.emplace_back();
    for (uint32_t marker : row->markers) result.back().push_back(from_bf16(scores[start + lead + marker]));
    start += lead + row->ids.size();
  }
  return result;
}

void Runtime::warm_up() {
  Row row;
  row.type = "noul";
  row.ids.assign(8, _cfg.tokens.at("bos"));
  row.markers = {1, 2};
  row.readout = {{_cfg.tokens.at("bos")}};
  for (const auto& [seq_len, chain] : _chains) {
    Cost cost;
    for (int i = 0; i < 2; ++i) {
      cost = Cost();
      pass({&row}, {}, cost, seq_len);
    }
    _pass_ms[seq_len] = cost.mla_ms;
  }
  if (_tower) {
    const Vision& v = *_cfg.vision;
    Picture grey;
    grey.width = grey.height = v.patch_size * v.merge * 8;
    grey.rgb.assign(size_t(grey.width) * grey.height * 3, 128);
    Cost cost;
    see(grey, cost);
    _seen.clear();
  }
}

json Runtime::predict(const json& state, const json& questions, const std::vector<Picture>& pictures) {
  if (!questions.is_object() || questions.empty()) throw std::invalid_argument("\"questions\" must be a non-empty object");
  Cost cost;
  // The pictures, in order, in front of every question.
  const bool lm = _cfg.family == "lm";
  std::vector<uint16_t> prefix;
  std::vector<size_t> picture_tokens;
  for (const Picture& picture : pictures) {
    const std::vector<uint16_t> one = see(picture, cost);
    prefix.insert(prefix.end(), one.begin(), one.end());
    picture_tokens.push_back(one.size() / _cfg.hidden_size);
  }
  cost.prefix = prefix.size() / _cfg.hidden_size;
  // In front of every row (omni), or inside it and counted with its tokens (lm).
  const size_t lead = lm ? 0 : cost.prefix;
  const bool picture = !pictures.empty();
  if (cost.prefix + 64 > max_tokens())
    throw std::invalid_argument(fmt::format("the pictures take {} of the {} positions; send fewer or smaller pictures", cost.prefix, max_tokens()));
  const size_t limit = lm ? max_tokens() : std::min<size_t>(picture ? _cfg.image_text_limit : _cfg.text_limit, max_tokens() - lead);

  auto stage = Clock::now();
  std::vector<std::string> names;
  std::vector<Row> rows;
  for (const auto& [name, question] : questions.items()) {
    if (!question.is_object() || !question.contains("type")) throw std::invalid_argument("a question is an object with a \"type\"");
    names.push_back(name);
    rows.push_back(encode(state, question, limit, picture, picture_tokens));
    cost.tokens = std::max(cost.tokens, lead + rows.back().ids.size());
    cost.dropped = std::max(cost.dropped, rows.back().dropped);
  }
  cost.tokenize_ms = elapsed_ms(stage);

  // Rows laid end to end, in as few passes as they fit, unless a pass each on shorter chains
  // is measured to be quicker.
  const auto price = [&](size_t positions) {
    const auto chain = _chains.lower_bound(static_cast<uint32_t>(positions));
    const auto measured = _pass_ms.find(chain->first);
    return measured == _pass_ms.end() ? double(chain->first) : measured->second;
  };
  std::vector<std::vector<size_t>> bins;
  std::vector<size_t> used;
  double packed_price = 0, single_price = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    const size_t need = lead + rows[i].ids.size();
    single_price += price(need);
    size_t bin = 0;
    while (bin < bins.size() && used[bin] + need > max_tokens()) ++bin;
    if (bin == bins.size()) { bins.emplace_back(); used.push_back(0); }
    bins[bin].push_back(i);
    used[bin] += need;
  }
  for (size_t positions : used) packed_price += price(positions);
  if (single_price < packed_price) {
    bins.clear();
    for (size_t i = 0; i < rows.size(); ++i) bins.push_back({i});
  }
  std::vector<std::vector<float>> scores(rows.size());
  for (const auto& bin : bins) {
    std::vector<const Row*> together;
    for (size_t i : bin) together.push_back(&rows[i]);
    const auto out = pass(together, prefix, cost);
    for (size_t k = 0; k < bin.size(); ++k) scores[bin[k]] = out[k];
  }

  stage = Clock::now();
  json answers = json::object();
  size_t index = 0;
  for (const auto& [name, question] : questions.items()) {
    const Row& row = rows[index];
    const std::vector<float>& z = scores[index++];
    const size_t k = z.size();
    // Text questions are calibrated with a temperature for their type and number of options;
    // picture questions are not.
    double temperature = 1.0;
    if (!picture && !lm) {
      const std::string key = row.type + ":" + (k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+");
      const auto found = _cfg.temperatures.find(key);
      const auto fallback = _cfg.temperatures.find(row.type);
      temperature = found != _cfg.temperatures.end() ? found->second : fallback != _cfg.temperatures.end() ? fallback->second : 1.0;
    }
    std::vector<double> p(k);
    double top = -1e30, sum = 0;
    for (size_t i = 0; i < k; ++i) top = std::max(top, p[i] = double(z[i]) / temperature);
    for (auto& value : p) sum += value = std::exp(value - top);
    for (auto& value : p) value /= sum;
    const size_t best = std::max_element(p.begin(), p.end()) - p.begin();
    // The app's confidence, as for Laya and CLM: the top probability minus the mean of the rest.
    const double confidence = k < 2 ? 1.0 : std::clamp(p[best] - (1.0 - p[best]) / double(k - 1), 0.0, 1.0);
    const json& criteria = question.contains("criteria") ? question.at("criteria") : json();
    json answer;
    if (row.type == "choice") {
      std::vector<std::string> keys;
      if (criteria.is_object()) for (const auto& [key, unused] : criteria.items()) keys.push_back(key);
      else for (const auto& label : criteria) keys.push_back(plain(label));
      json probabilities = json::object();
      for (size_t i = 0; i < k; ++i) probabilities[keys[i]] = round4(p[i]);
      answer = {{"type", "choice"}, {"choice", keys[best]}, {"probabilities", probabilities}, {"confidence", round4(confidence)}};
    } else if (row.type == "score") {
      double expected = 0;
      json legend = json::object(), probabilities = json::object();
      for (size_t i = 0; i < k; ++i) {
        expected += i * p[i];
        legend[std::to_string(i)] = plain(criteria[i]);
        probabilities[std::to_string(i)] = round4(p[i]);
      }
      answer = {{"type", "score"}, {"score", round4(expected)}, {"legend", legend},
                {"probabilities", probabilities}, {"confidence", round4(confidence)}};
    } else {
      // d1-omni reads a yes-or-no question as [false, true]; the language model's groups are [yes, no].
      const double yes = lm ? p[0] : p[1];
      answer = {{"type", "noul"}, {"noul", round4(yes)}, {"confidence", round4(std::max(yes, 1.0 - yes))}};
    }
    answer["answer_confidence"] = round4(std::clamp(p[best], 0.0, 1.0));
    answers[name] = answer;
  }
  const double post_ms = elapsed_ms(stage);

  auto ms = [](double value) { return std::round(value * 1e3) / 1e3; };
  return {
      {"answers", answers},
      {"usage",
       {{"tokens", cost.tokens},
        {"max_len", max_tokens()},
        {"seq_len", cost.seq_len},
        {"state_tokens_dropped", cost.dropped},
        {"truncated", cost.dropped > 0},
        {"decisions", questions.size()},
        {"encoder_passes", cost.passes},
        {"images", cost.pictures},
        {"images_cached", cost.pictures_cached},
        {"image_tokens", cost.prefix},
        {"tokenize_ms", ms(cost.tokenize_ms)},
        {"pre_ms", ms(cost.pre_ms)},
        {"vision_ms", ms(cost.vision_ms)},
        {"mla_ms", ms(cost.mla_ms + cost.vision_ms)},
        {"post_ms", ms(post_ms)},
        {"latency_ms", ms(cost.tokenize_ms + cost.pre_ms + cost.vision_ms + cost.mla_ms + post_ms)}}},
  };
}

}  // namespace d1
