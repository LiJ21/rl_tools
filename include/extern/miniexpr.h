#ifndef MINI_EXPR
#define MINI_EXPR

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mini_expr {

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class Parser {
public:
  using Vars = std::unordered_map<std::string, double>;

  Parser(std::string_view input, const Vars& vars)
      : s_(input), vars_(vars) {}

  double eval() {
    pos_ = 0;
    values_.clear();
    ops_.clear();

    // At the beginning, we expect an operand (number/ident/'(' / function / unary +/-)
    Expect expect = Expect::Operand;

    while (true) {
      skip_ws();
      if (pos_ >= s_.size()) break;

      char c = s_[pos_];

      // Number
      if (is_number_start(c)) {
        if (expect != Expect::Operand) {
          throw Error("Unexpected number at pos " + std::to_string(pos_));
        }
        values_.push_back(parse_number());
        expect = Expect::Operator;
        continue;
      }

      // Identifier: variable or function name
      if (is_ident_start(c)) {
        if (expect != Expect::Operand) {
          throw Error("Unexpected identifier at pos " + std::to_string(pos_));
        }
        std::string name = parse_ident();
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == '(') {
          // function call: push function marker, then '('
          push_op(Op::Func, name);
          push_op(Op::LParen, {});
          ++pos_; // consume '('
          expect = Expect::Operand; // expect first arg or ')'
        } else {
          // variable
          auto it = vars_.find(name);
          if (it == vars_.end()) throw Error("Unknown variable '" + name + "'");
          values_.push_back(it->second);
          expect = Expect::Operator;
        }
        continue;
      }

      // Parentheses
      if (c == '(') {
        if (expect != Expect::Operand) {
          throw Error("Unexpected '(' at pos " + std::to_string(pos_));
        }
        push_op(Op::LParen, {});
        ++pos_;
        expect = Expect::Operand;
        continue;
      }

      if (c == ')') {
        if (expect == Expect::Operand) {
          // allow empty arg list only for function calls like f()
          // but we don't have zero-arg functions; still parse structure safely
        }
        ++pos_;
        // Reduce until '('
        reduce_until_lparen();
        // Pop '('
        if (ops_.empty() || ops_.back().kind != Op::LParen)
          throw Error("Mismatched ')' at pos " + std::to_string(pos_));
        ops_.pop_back();

        // If there is a function marker before this '(', apply it now
        if (!ops_.empty() && ops_.back().kind == Op::Func) {
          apply_function(ops_.back().name);
          ops_.pop_back();
        }

        expect = Expect::Operator;
        continue;
      }

      // Comma: function argument separator
      if (c == ',') {
        if (!inside_function_call()) {
          throw Error("Unexpected ',' outside function call at pos " + std::to_string(pos_));
        }
        ++pos_;
        // Reduce up to nearest '(' (but do not pop it)
        reduce_until_lparen();
        // Mark that we finished one argument for the nearest function
        bump_func_arity();
        expect = Expect::Operand;
        continue;
      }

      // Operators
      if (c == '+' || c == '-' || c == '*' || c == '/') {
        ++pos_;
        OpKind opk = Op::Add;
        bool unary = false;

        if (expect == Expect::Operand) {
          // unary +/-
          if (c == '+') { opk = Op::UPlus; unary = true; }
          else if (c == '-') { opk = Op::UMinus; unary = true; }
          else {
            throw Error(std::string("Unexpected operator '") + c +
                        "' at pos " + std::to_string(pos_ - 1));
          }
        } else {
          // binary
          if (c == '+') opk = Op::Add;
          else if (c == '-') opk = Op::Sub;
          else if (c == '*') opk = Op::Mul;
          else opk = Op::Div;
        }

        // Reduce operators with higher or equal precedence (left-associative),
        // except unary operators are right-associative-ish in our handling:
        // We'll treat unary as higher precedence and reduce only strictly higher.
        reduce_for_incoming(opk);

        push_op(opk, {});
        expect = Expect::Operand;
        continue;
      }

      throw Error(std::string("Unexpected character '") + c +
                  "' at pos " + std::to_string(pos_));
    }

    skip_ws();
    if (pos_ != s_.size())
      throw Error("Unexpected trailing characters at pos " + std::to_string(pos_));

    // Final reductions
    while (!ops_.empty()) {
      if (ops_.back().kind == Op::LParen)
        throw Error("Mismatched '('");
      if (ops_.back().kind == Op::Func)
        throw Error("Dangling function call");
      apply_op(ops_.back().kind);
      ops_.pop_back();
    }

    if (values_.size() != 1)
      throw Error("Invalid expression");
    return values_.back();
  }

private:
  // ---- operator stack items ----
  struct Op {
    enum Kind : int { Add, Sub, Mul, Div, UPlus, UMinus, LParen, Func } kind;
    std::string name; // only for Func
    int arity = 0;    // only for Func: number of commas seen so far
    static constexpr Kind AddK = Add;
  };
  using OpKind = Op::Kind;

  enum class Expect { Operand, Operator };

  // stacks
  std::vector<double> values_;
  std::vector<Op> ops_;

  // helpers: ctype safe
  static bool is_space(char c) { return std::isspace(static_cast<unsigned char>(c)); }
  static bool is_digit(char c) { return std::isdigit(static_cast<unsigned char>(c)); }
  static bool is_alpha(char c) { return std::isalpha(static_cast<unsigned char>(c)); }
  static bool is_alnum(char c) { return std::isalnum(static_cast<unsigned char>(c)); }

  static bool is_ident_start(char c) { return is_alpha(c) || c == '_'; }
  static bool is_ident_cont(char c) { return is_alnum(c) || c == '_'; }
  static bool is_number_start(char c) { return is_digit(c) || c == '.'; }

  void skip_ws() {
    while (pos_ < s_.size() && is_space(s_[pos_])) ++pos_;
  }

  std::string parse_ident() {
    size_t start = pos_;
    ++pos_;
    while (pos_ < s_.size() && is_ident_cont(s_[pos_])) ++pos_;
    return std::string(s_.substr(start, pos_ - start));
  }

  double parse_number() {
    size_t start = pos_;
    bool saw_digit = false;

    auto eat_digits = [&] {
      while (pos_ < s_.size() && is_digit(s_[pos_])) {
        ++pos_;
        saw_digit = true;
      }
    };

    eat_digits();
    if (pos_ < s_.size() && s_[pos_] == '.') {
      ++pos_;
      eat_digits();
    }
    if (!saw_digit) throw Error("Invalid number at pos " + std::to_string(start));

    if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
      bool exp_digit = false;
      while (pos_ < s_.size() && is_digit(s_[pos_])) {
        ++pos_;
        exp_digit = true;
      }
      if (!exp_digit) throw Error("Invalid exponent at pos " + std::to_string(start));
    }

    std::string tmp(s_.substr(start, pos_ - start));
    char* endp = nullptr;
    double v = std::strtod(tmp.c_str(), &endp);
    if (!endp || *endp != '\0') throw Error("Invalid number '" + tmp + "'");
    return v;
  }

  static int precedence(OpKind k) {
    switch (k) {
      case Op::UPlus:
      case Op::UMinus: return 3;
      case Op::Mul:
      case Op::Div:    return 2;
      case Op::Add:
      case Op::Sub:    return 1;
      default:         return 0;
    }
  }

  static bool is_unary(OpKind k) {
    return k == Op::UPlus || k == Op::UMinus;
  }

  void push_op(OpKind k, std::string name) {
    Op o;
    o.kind = k;
    o.name = std::move(name);
    o.arity = 0;
    ops_.push_back(std::move(o));
  }

  void reduce_for_incoming(OpKind incoming) {
    // Reduce while top operator has:
    // - higher precedence; or
    // - equal precedence and incoming is left-associative (binary ops here)
    // For unary incoming, reduce only strictly higher (there is none), so it stays tight.
    while (!ops_.empty()) {
      OpKind top = ops_.back().kind;
      if (top == Op::LParen || top == Op::Func) break;
      int ptop = precedence(top);
      int pinc = precedence(incoming);

      if (ptop > pinc || (ptop == pinc && !is_unary(incoming))) {
        apply_op(top);
        ops_.pop_back();
      } else break;
    }
  }

  void reduce_until_lparen() {
    while (!ops_.empty() && ops_.back().kind != Op::LParen) {
      if (ops_.back().kind == Op::Func)
        throw Error("Internal error: function marker before '('");
      apply_op(ops_.back().kind);
      ops_.pop_back();
    }
  }

  bool inside_function_call() const {
    // We're inside a function call if there exists a '(' that has a Func before it
    // on the operator stack; simplest check: find nearest '(' and see if preceding is Func.
    for (int i = (int)ops_.size() - 1; i >= 0; --i) {
      if (ops_[i].kind == Op::LParen) {
        return (i > 0 && ops_[i - 1].kind == Op::Func);
      }
    }
    return false;
  }

  void bump_func_arity() {
    // Find nearest '(' and bump its function's comma count
    for (int i = (int)ops_.size() - 1; i >= 0; --i) {
      if (ops_[i].kind == Op::LParen) {
        if (i == 0 || ops_[i - 1].kind != Op::Func)
          throw Error("',' found but not in function call");
        ops_[i - 1].arity += 1; // count commas
        return;
      }
    }
    throw Error("',' found but missing '('");
  }

  void apply_op(OpKind k) {
    if (is_unary(k)) {
      if (values_.empty()) throw Error("Missing operand for unary operator");
      double a = values_.back();
      values_.pop_back();
      if (k == Op::UPlus) values_.push_back(+a);
      else values_.push_back(-a);
      return;
    }

    // binary
    if (values_.size() < 2) throw Error("Missing operand for binary operator");
    double b = values_.back(); values_.pop_back();
    double a = values_.back(); values_.pop_back();

    switch (k) {
      case Op::Add: values_.push_back(a + b); break;
      case Op::Sub: values_.push_back(a - b); break;
      case Op::Mul: values_.push_back(a * b); break;
      case Op::Div: values_.push_back(a / b); break;
      default: throw Error("Internal error: unknown operator");
    }
  }

  void apply_function(const std::string& name) {
    // arity = commas + 1, unless it was empty "f()"
    // But we don't support 0-arg functions; treat as error.
    int commas = 0;
    if (!ops_.empty() && ops_.back().kind == Op::Func && ops_.back().name == name) {
      commas = ops_.back().arity;
    } else {
      // This can happen if apply_function is called after popping ')'
      // and Func is not the very top (we call with ops_.back().name before pop),
      // so it should match. Keep this defensive:
      for (int i = (int)ops_.size() - 1; i >= 0; --i) {
        if (ops_[i].kind == Op::Func && ops_[i].name == name) {
          commas = ops_[i].arity;
          break;
        }
      }
    }
    int argc = commas + 1;

    auto need = [&](int n) {
      if (argc != n) {
        throw Error("Function '" + name + "' expects " + std::to_string(n) +
                    " arguments");
      }
      if ((int)values_.size() < n) throw Error("Not enough arguments for '" + name + "'");
    };

    if (name == "sin") { need(1); double a = popv(); values_.push_back(std::sin(a)); return; }
    if (name == "cos") { need(1); double a = popv(); values_.push_back(std::cos(a)); return; }
    if (name == "exp") { need(1); double a = popv(); values_.push_back(std::exp(a)); return; }
    if (name == "log") { need(1); double a = popv(); values_.push_back(std::log(a)); return; }
    if (name == "max") { need(2); double b = popv(); double a = popv(); values_.push_back((a > b) ? a : b); return; }
    if (name == "min") { need(2); double b = popv(); double a = popv(); values_.push_back((a < b) ? a : b); return; }

    throw Error("Unknown function '" + name + "'");
  }

  double popv() {
    if (values_.empty()) throw Error("Value stack underflow");
    double v = values_.back();
    values_.pop_back();
    return v;
  }

  std::string_view s_;
  const Vars& vars_;
  size_t pos_ = 0;
};

// convenience
inline double eval(std::string_view e, const Parser::Vars& v = {}) {
  return Parser(e, v).eval();
}

} // namespace mini_expr

#endif
