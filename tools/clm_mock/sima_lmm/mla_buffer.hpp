// A stand-in for the LLiMa runtime's MLA buffers, for building the runtime off the board
// (tools/clm_mock/README). Memory is ordinary memory; nothing here is the real interface's
// behaviour beyond what runtime/src/clm.cpp uses.
#ifndef CLM_MOCK_MLA_BUFFER_HPP_
#define CLM_MOCK_MLA_BUFFER_HPP_
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace simaai::llima {

class MLABuffer {
 public:
  MLABuffer(std::string name, std::vector<size_t> shape, std::string dtype, bool) : _name(std::move(name)), _shape(std::move(shape)) {
    (void)dtype;
    size_t count = 1;
    for (size_t dim : _shape) count *= dim;
    _bytes.resize(count * 2);      // bfloat16
  }
  void allocate() {}
  void clear(bool = true) { std::fill(_bytes.begin(), _bytes.end(), 0); }
  void upload(const void* data, size_t = 0, size_t = 0, bool = true) { std::memcpy(_bytes.data(), data, _bytes.size()); }
  void download(void* data) const { std::memcpy(data, _bytes.data(), _bytes.size()); }
  void flush_cache() const {}
  void invalidate_cache() const {}
  void* get_virtual_addr() const { return const_cast<uint8_t*>(_bytes.data()); }
  const std::vector<size_t>& get_shape() const { return _shape; }

 private:
  std::string _name;
  std::vector<size_t> _shape;
  std::vector<uint8_t> _bytes;
};

class MLABufferSlice {
 public:
  MLABufferSlice(MLABuffer* buffer = nullptr) : buffer(buffer) {}
  MLABuffer* buffer;
};

}  // namespace simaai::llima
#endif
