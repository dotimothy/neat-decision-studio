// Question rendering, sequence layout and answer decoding: the text side of Laya.
// Ported from upstream `laya/common.py` (`render_options`, `build_sequence`) and
// `laya/agent.py` (`_to_internal`, `_decode_answers`); see there for the reasoning.
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <fmt/format.h>

#include "laya/runtime.hpp"

namespace laya {
namespace {

constexpr size_t kMaxOptionTokens = 48;
constexpr float kTempMin = 0.5f, kTempMax = 5.0f;

std::string replace_all(std::string text, const std::string& from, const std::string& to) {
  if (from.empty()) return text;
  for (size_t pos = 0; (pos = text.find(from, pos)) != std::string::npos; pos += to.size())
    text.replace(pos, from.size(), to);
  return text;
}

// Strings pass through; anything structured becomes JSON text.
std::string render_criterion(const json& value) {
  return value.is_string() ? value.get<std::string>() : python_json(value);
}

bool is_blank(const json& value) {
  return value.is_null() || (value.is_string() && value.get<std::string>().empty());
}

std::string strip(const std::string& text) {
  const auto begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return "";
  return text.substr(begin, text.find_last_not_of(" \t\r\n") - begin + 1);
}

float clamp_temperature(float t) {
  if (!std::isfinite(t)) return 1.0f;
  return std::min(kTempMax, std::max(kTempMin, t));
}

std::string temp_bucket(QType type, size_t k) {
  static const char* names[] = {"choice", "score", "noul"};
  const char* size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return fmt::format("{}:{}", names[static_cast<int>(type)], size);
}

double round4(double value) { return std::round(value * 1e4) / 1e4; }

}  // namespace

std::string python_json(const json& value) {
  if (value.is_object()) {
    std::string out = "{";
    for (auto it = value.begin(); it != value.end(); ++it) {
      if (it != value.begin()) out += ", ";
      out += json(it.key()).dump() + ": " + python_json(it.value());
    }
    return out + "}";
  }
  if (value.is_array()) {
    std::string out = "[";
    for (size_t i = 0; i < value.size(); ++i) out += (i ? ", " : "") + python_json(value[i]);
    return out + "]";
  }
  return value.dump();  // scalars: same spelling, non-ASCII left unescaped
}

Question parse_question(const json& qdef) {
  if (!qdef.is_object() || !qdef.contains("type") || !qdef.contains("instructions"))
    throw std::invalid_argument("a question needs \"type\" and \"instructions\"");
  Question q;
  const std::string type = qdef.at("type").get<std::string>();
  const json& ins = qdef.at("instructions");
  q.instructions = ins.is_string() ? ins.get<std::string>() : python_json(ins);
  const json crit = qdef.value("criteria", json());
  if (type != "noul" && qdef.contains("labels"))
    throw std::invalid_argument("labels is only supported for noul questions");

  if (type == "choice") {
    q.type = QType::Choice;
    if (crit.is_array()) {
      for (const auto& label : crit) {
        q.option_keys.push_back(label.is_string() ? label.get<std::string>() : python_json(label));
        q.option_texts.push_back(q.option_keys.back());
      }
    } else if (crit.is_object()) {
      for (const auto& [label, description] : crit.items()) {
        q.option_keys.push_back(label);
        q.option_texts.push_back(
            is_blank(description) ? label : label + ": " + render_criterion(description));
      }
    }
    if (q.option_keys.empty())
      throw std::invalid_argument("a choice question needs a non-empty \"criteria\"");
  } else if (type == "score") {
    q.type = QType::Score;
    if (!crit.is_array() || crit.empty())
      throw std::invalid_argument("a score question needs a non-empty \"criteria\" list");
    for (size_t level = 0; level < crit.size(); ++level)
      q.option_texts.push_back(fmt::format("level {}: {}", level, render_criterion(crit[level])));
  } else if (type == "noul") {
    q.type = QType::Noul;
    std::string false_label = "false", true_label = "true";
    if (qdef.contains("labels")) {
      const json& labels = qdef.at("labels");
      if (!labels.is_object() || labels.size() != 2 || !labels.contains("false") ||
          !labels.contains("true") || !labels.at("false").is_string() || !labels.at("true").is_string())
        throw std::invalid_argument(
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
      false_label = strip(labels.at("false").get<std::string>());
      true_label = strip(labels.at("true").get<std::string>());
      if (false_label.empty() || true_label.empty() || false_label == true_label)
        throw std::invalid_argument(
            "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
    }
    const json none;
    const json& false_crit = crit.is_object() && crit.contains("false") ? crit.at("false") : none;
    const json& true_crit = crit.is_object() && crit.contains("true") ? crit.at("true") : none;
    q.option_texts = {
        false_label + ": " +
            (is_blank(false_crit) ? "no, the statement does not hold" : render_criterion(false_crit)),
        true_label + ": " +
            (is_blank(true_crit) ? "yes, the statement holds" : render_criterion(true_crit)),
    };
  } else {
    throw std::invalid_argument("unknown question type: " + type);
  }
  return q;
}

// [CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
Sequence Runtime::build_sequence(const std::vector<uint32_t>& state_ids, const Question& question,
                                 bool truncate_left, uint32_t limit) const {
  static const char* names[] = {"choice", "score", "noul"};
  const size_t max_len = std::min<size_t>(_cfg.max_len, limit ? limit : max_tokens());
  const size_t head_max_len = _cfg.head_max_len;

  auto head_ids = tokenize(fmt::format("{} question: {}", names[static_cast<int>(question.type)],
                                       replace_all(question.instructions, _cfg.mask_token, " ")));
  std::vector<std::vector<uint32_t>> options;
  for (const auto& text : question.option_texts) {
    auto tokens = tokenize(" " + replace_all(text, _cfg.mask_token, " "));
    if (tokens.size() > kMaxOptionTokens) tokens.resize(kMaxOptionTokens);
    tokens.insert(tokens.begin(), _cfg.mask_token_id);
    options.push_back(std::move(tokens));
  }
  auto option_tokens = [&] {
    return std::accumulate(options.begin(), options.end(), size_t{0},
                           [](size_t sum, const auto& o) { return sum + o.size(); });
  };
  // Signed: the options alone may exceed the head budget.
  auto budget = static_cast<long>(head_max_len) - static_cast<long>(option_tokens());
  if (budget < 16) {
    const size_t per = std::max<size_t>(4, (head_max_len - 16) / std::max<size_t>(1, options.size()));
    for (auto& option : options)
      if (option.size() > per) option.resize(per);
    budget = static_cast<long>(head_max_len) - static_cast<long>(option_tokens());
  }
  const size_t head_keep = static_cast<size_t>(std::max<long>(8, budget));
  if (head_ids.size() > head_keep) head_ids.resize(head_keep);

  Sequence seq;
  seq.ids.push_back(_cfg.cls_token_id);
  seq.ids.insert(seq.ids.end(), head_ids.begin(), head_ids.end());
  seq.ids.push_back(_cfg.sep_token_id);
  for (const auto& option : options) {
    seq.markers.push_back(static_cast<uint32_t>(seq.ids.size()));
    seq.ids.insert(seq.ids.end(), option.begin(), option.end());
  }
  seq.ids.push_back(_cfg.sep_token_id);

  const size_t room = max_len > seq.ids.size() + 1 ? max_len - seq.ids.size() - 1 : 0;
  const size_t used = std::min(room, state_ids.size());
  const auto first = truncate_left ? state_ids.end() - used : state_ids.begin();
  seq.ids.insert(seq.ids.end(), first, first + used);
  seq.ids.push_back(_cfg.sep_token_id);
  if (seq.ids.size() > max_len) seq.ids.resize(max_len);
  std::erase_if(seq.markers, [&](uint32_t m) { return m >= max_len; });
  seq.state_tokens = state_ids.size();
  seq.state_tokens_used = used;

  if (seq.markers.size() != question.option_texts.size())
    throw std::invalid_argument(fmt::format(
        "only {} of the question's {} option markers fit in max_len={} with head_max_len={}; "
        "use fewer options or compile a longer sequence length",
        seq.markers.size(), question.option_texts.size(), max_len, head_max_len));
  return seq;
}

json decode_answer(const Config& cfg, const Question& question, const RawOutput& out) {
  const size_t k = out.logits.size();
  const int qt = static_cast<int>(question.type);
  const auto bucket = cfg.temperature_by_options.find(temp_bucket(question.type, k));
  const float scale = clamp_temperature(
      bucket != cfg.temperature_by_options.end() ? bucket->second : cfg.temperature[qt]);

  std::vector<double> p(k);
  const float top_logit = *std::max_element(out.logits.begin(), out.logits.end());
  double sum = 0;
  for (size_t i = 0; i < k; ++i) sum += p[i] = std::exp((out.logits[i] - top_logit) / scale);
  for (auto& value : p) value /= sum;

  const size_t best = std::max_element(p.begin(), p.end()) - p.begin();
  double entropy = 0;
  for (double value : p) entropy -= value * std::log(std::clamp(value, 1e-12, 1.0));
  const double confidence = k < 2 ? 1.0 : std::clamp(1.0 - entropy / std::log(double(k)), 0.0, 1.0);
  const double answer_confidence = round4(std::clamp(p[best], 0.0, 1.0));
  const double act_top = std::max(out.act_logits[0], out.act_logits[1]);
  const double act0 = std::exp(out.act_logits[0] - act_top), act1 = std::exp(out.act_logits[1] - act_top);
  const json action = {{"act_probability", round4(act0 / (act0 + act1))}};

  json answer;
  if (question.type == QType::Choice) {
    json probabilities = json::object();
    for (size_t i = 0; i < k; ++i) probabilities[question.option_keys[i]] = round4(p[i]);
    answer = {{"type", "choice"}, {"choice", question.option_keys[best]},
              {"probabilities", probabilities}, {"confidence", round4(confidence)}};
  } else if (question.type == QType::Score) {
    double expected = 0;
    json legend = json::object(), probabilities = json::object();
    for (size_t i = 0; i < k; ++i) {
      expected += i * p[i];
      // "level N: text" -> "text"
      legend[std::to_string(i)] = question.option_texts[i].substr(question.option_texts[i].find(": ") + 2);
      probabilities[std::to_string(i)] = round4(p[i]);
    }
    answer = {{"type", "score"}, {"score", round4(expected)}, {"legend", legend},
              {"probabilities", probabilities}, {"confidence", round4(confidence)}};
  } else {
    answer = {{"type", "noul"}, {"noul", round4(p[1])}, {"confidence", round4(std::max(p[1], 1.0 - p[1]))}};
  }
  answer["answer_confidence"] = answer_confidence;
  answer["action"] = action;
  return answer;
}

}  // namespace laya
