// A stand-in tokenizer: one token a byte. See mla_buffer.hpp.
#ifndef CLM_MOCK_TOKENIZER_HPP_
#define CLM_MOCK_TOKENIZER_HPP_
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace simaai::llima {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;
  virtual std::vector<uint32_t> encode(const std::string& text, bool = true) {
    std::vector<uint32_t> ids;
    for (unsigned char byte : text) ids.push_back(byte);
    return ids;
  }
  static std::unique_ptr<Tokenizer> from_hf_json(const std::filesystem::path&) { return std::make_unique<Tokenizer>(); }
};

}  // namespace simaai::llima
#endif
