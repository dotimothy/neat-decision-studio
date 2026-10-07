// A stand-in for a compiled graph: see mla_buffer.hpp. It is not Qwen3. What it keeps of the
// real graphs is what the runtime's packing depends on: a position's output is made only of
// the positions its mask row opens, in order, so texts laid end to end come out as they would
// alone exactly when the mask and the buffers are handled right.
#ifndef CLM_MOCK_MLA_MODEL_HPP_
#define CLM_MOCK_MLA_MODEL_HPP_
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "mla_buffer.hpp"

namespace simaai::llima {

inline void connect_mla_rt(const std::vector<std::string>&) {}
inline void disconnect_mla_rt() {}

class MLAModelWithBuffer {
 public:
  MLAModelWithBuffer(std::filesystem::path path, std::vector<MLABufferSlice> ifms, std::vector<MLABufferSlice> ofms)
      : _path(std::move(path)), _ifms(std::move(ifms)), _ofms(std::move(ofms)) {}
  void load() { std::cout << "Loading model " << _path.string() << "\nDone loading " << _path.string() << std::endl; }
  void free() {}
  void run(std::map<uint8_t, MLABufferSlice>* = nullptr, std::map<uint8_t, MLABufferSlice>* = nullptr) {
    const MLABuffer& in = *_ifms.at(0).buffer;
    const MLABuffer& mask = *_ifms.at(1).buffer;
    MLABuffer& out = *_ofms.at(0).buffer;
    const size_t tokens = in.get_shape()[0], width = in.get_shape()[1];
    const auto* x = static_cast<const uint16_t*>(in.get_virtual_addr());
    const auto* open = static_cast<const uint16_t*>(mask.get_virtual_addr());   // [query][key], 0 where open
    std::vector<uint16_t> y(tokens * width);
    for (size_t query = 0; query < tokens; ++query) {
      // A running mix of the open keys, oldest first, weighted by how far back each is: it
      // depends on which keys are open, on their order and on their distance, as attention does.
      std::vector<float> acc(width, 0.0f);
      size_t seen = 0;
      for (size_t key = 0; key < tokens; ++key) {
        if (open[query * tokens + key] != 0) continue;
        ++seen;
        const float weight = 1.0f / float(1 + query - std::min(query, key));
        for (size_t c = 0; c < width; ++c) acc[c] = 0.7f * acc[c] + weight * std::tanh(from(x[key * width + c]) + 0.01f * float(c % 7));
      }
      for (size_t c = 0; c < width; ++c) y[query * width + c] = to(acc[c] / float(seen ? seen : 1) + from(x[query * width + c]) * 0.5f);
    }
    out.upload(y.data());
  }

 private:
  static float from(uint16_t v) { uint32_t bits = uint32_t(v) << 16; float f; std::memcpy(&f, &bits, 4); return f; }
  static uint16_t to(float f) { uint32_t bits; std::memcpy(&bits, &f, 4); return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16); }
  std::filesystem::path _path;
  std::vector<MLABufferSlice> _ifms, _ofms;
};

}  // namespace simaai::llima
#endif
