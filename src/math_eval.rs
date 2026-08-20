//! Deterministic arithmetic evaluation for pure math prompts.
//!
//! Math questions are the worst case for the tiny LLM: they burn hundreds of
//! `/think` tokens on CPU and still hallucinate digits. When the user's prompt
//! is nothing but an arithmetic expression (optionally wrapped in phrasing
//! like "calculate" or "what is 17% of 3482"), we evaluate it exactly here and
//! skip the model entirely.
//!
//! The evaluator is a plain shunting-yard parser over a whitelisted grammar —
//! no `eval`, no subprocess, no unbounded memory.

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
    Op(char),
    LParen,
    RParen,
}

fn tokenize(expr: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = expr.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' | ',' => {
                chars.next();
            }
            '0'..='9' | '.' => {
                let mut num = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() || c == '.' {
                        num.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let value: f64 = num
                    .parse()
                    .map_err(|_| format!("invalid number: {}", num))?;
                tokens.push(Token::Num(value));
            }
            '+' | '-' | '*' | '/' | '%' | '^' => {
                tokens.push(Token::Op(c));
                chars.next();
            }
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            'x' | '×' => {
                tokens.push(Token::Op('*'));
                chars.next();
            }
            '÷' => {
                tokens.push(Token::Op('/'));
                chars.next();
            }
            _ => return Err(format!("unexpected character: {:?}", c)),
        }
    }
    if tokens.is_empty() {
        return Err("empty expression".to_string());
    }
    Ok(tokens)
}

fn precedence(op: char) -> u8 {
    match op {
        '+' | '-' => 1,
        '*' | '/' | '%' => 2,
        '^' => 3,
        _ => 0,
    }
}

fn apply_op(a: f64, b: f64, op: char) -> Result<f64, String> {
    match op {
        '+' => Ok(a + b),
        '-' => Ok(a - b),
        '*' => Ok(a * b),
        '/' => {
            if b == 0.0 {
                Err("division by zero".to_string())
            } else {
                Ok(a / b)
            }
        }
        '%' => Ok(a % b),
        '^' => Ok(a.powf(b)),
        _ => Err(format!("unknown operator: {}", op)),
    }
}

/// Shunting-yard evaluation of an arithmetic expression.
pub fn eval_expression(expr: &str) -> Result<f64, String> {
    let tokens = tokenize(expr)?;
    let mut output: Vec<f64> = Vec::new();
    let mut ops: Vec<char> = Vec::new();
    let mut expect_operand = true;

    for token in tokens {
        match token {
            Token::Num(n) => {
                if !expect_operand {
                    return Err("missing operator between numbers".to_string());
                }
                output.push(n);
                expect_operand = false;
            }
            Token::Op(op) => {
                if expect_operand {
                    if op == '-' {
                        // Unary minus: fold into the next number via 0 - x.
                        output.push(0.0);
                        ops.push('-');
                        expect_operand = true;
                        continue;
                    }
                    return Err(format!("unexpected operator: {}", op));
                }
                while let Some(&top) = ops.last() {
                    if top != '('
                        && (precedence(top) > precedence(op)
                            || (precedence(top) == precedence(op) && op != '^'))
                    {
                        let b = output.pop().ok_or("malformed expression")?;
                        let a = output.pop().ok_or("malformed expression")?;
                        output.push(apply_op(a, b, top)?);
                        ops.pop();
                    } else {
                        break;
                    }
                }
                ops.push(op);
                expect_operand = true;
            }
            Token::LParen => {
                ops.push('(');
                expect_operand = true;
            }
            Token::RParen => {
                if expect_operand {
                    return Err("empty parentheses".to_string());
                }
                while let Some(&top) = ops.last() {
                    if top == '(' {
                        break;
                    }
                    let b = output.pop().ok_or("malformed expression")?;
                    let a = output.pop().ok_or("malformed expression")?;
                    output.push(apply_op(a, b, top)?);
                    ops.pop();
                }
                if ops.pop() != Some('(') {
                    return Err("unbalanced parentheses".to_string());
                }
                expect_operand = false;
            }
        }
    }

    while let Some(top) = ops.pop() {
        if top == '(' {
            return Err("unbalanced parentheses".to_string());
        }
        let b = output.pop().ok_or("malformed expression")?;
        let a = output.pop().ok_or("malformed expression")?;
        output.push(apply_op(a, b, top)?);
    }

    if output.len() != 1 {
        return Err("malformed expression".to_string());
    }
    Ok(output[0])
}

fn format_result(value: f64) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    // Round to 6 decimal places to hide binary float noise (0.1 + 0.2),
    // then collapse near-integers to plain integers.
    let rounded = (value * 1e6).round() / 1e6;
    if (rounded - rounded.round()).abs() < 1e-9 {
        format!("{}", rounded.round() as i64)
    } else {
        rounded.to_string()
    }
}

/// Strip leading math-question phrasing. Returns None when the prompt
/// contains anything that is not pure arithmetic phrasing.
fn extract_math_query(prompt: &str) -> Option<String> {
    let mut text = prompt.trim();
    // Percentage phrasing: "17% of 3482" -> "(17 / 100) * 3482"
    let lower = text.to_lowercase();
    if let Some(pos) = lower.find('%') {
        let after = &text[pos + 1..];
        let after_trim = after.trim_start();
        if let Some(number_part) = after_trim.strip_prefix("of ") {
            let before = text[..pos].trim();
            let number: f64 = before.rsplit(char::is_whitespace).next()?.parse().ok()?;
            let rest = number_part.trim();
            // Validate the remainder parses; the multiplier is the number before '%'.
            let percent = number;
            let expanded = format!("({} / 100) * {}", percent, rest);
            return tokenize(&expanded).is_ok().then_some(expanded);
        }
        return None;
    }

    for prefix in [
        "what is the result of ",
        "what's the result of ",
        "whats the result of ",
        "calculate ",
        "compute ",
        "evaluate ",
        "what is ",
        "what's ",
        "whats ",
        "how much is ",
        "how much is the ",
        "solve ",
        "= ",
    ] {
        if lower.starts_with(prefix) {
            text = &text[prefix.len()..];
            break;
        }
    }
    // Reject prompts containing any non-arithmetic words ("prove", "explain", ...).
    let cleaned = text.trim_end_matches(['?', '.', '!', ' ']).trim();
    if cleaned.is_empty() {
        return None;
    }
    tokenize(cleaned).is_ok().then(|| cleaned.to_string())
}

/// Evaluate a prompt as pure arithmetic. Returns the formatted answer when
/// the whole prompt is a math expression, None otherwise (fall through to the
/// model). Word problems, proofs, and unit conversions are intentionally NOT
/// handled here.
pub fn try_answer(prompt: &str) -> Option<String> {
    // Long prompts are conversations, not calculator queries.
    if prompt.len() > 200 {
        return None;
    }
    let expr = extract_math_query(prompt)?;
    // A trailing "= " means the user wants the result of the left-hand side.
    let value = eval_expression(expr.trim_end_matches('=')).ok()?;
    if !value.is_finite() {
        return None;
    }
    Some(format_result(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_basic_arithmetic() {
        assert_eq!(try_answer("2+2"), Some("4".to_string()));
        assert_eq!(try_answer("calculate (12+8)*3"), Some("60".to_string()));
        assert_eq!(
            try_answer("what is 17% of 3482"),
            Some("591.94".to_string())
        );
        assert_eq!(try_answer("10 / 4"), Some("2.5".to_string()));
        assert_eq!(try_answer("2^10"), Some("1024".to_string()));
        assert_eq!(try_answer("7-10"), Some("-3".to_string()));
        // Float noise from binary arithmetic must not leak into answers.
        assert_eq!(try_answer("0.1 + 0.2"), Some("0.3".to_string()));
        assert_eq!(try_answer("whats 4*12"), Some("48".to_string()));
        assert_eq!(try_answer("whats 4*22."), Some("88".to_string()));
    }

    #[test]
    fn rejects_non_math_prompts() {
        assert_eq!(try_answer("prove that pi is irrational"), None);
        assert_eq!(try_answer("explain step by step how TCP works"), None);
        assert_eq!(try_answer("hello"), None);
        assert_eq!(try_answer("what is the weather"), None);
        assert_eq!(try_answer(""), None);
        assert_eq!(try_answer("10 / 0"), None);
    }

    #[test]
    fn handles_whitespace_and_words_in_question() {
        assert_eq!(try_answer("What is 5 * 5?"), Some("25".to_string()));
        assert_eq!(try_answer("  compute   100-42  "), Some("58".to_string()));
    }

    #[test]
    fn rejects_malformed_expressions() {
        assert!(eval_expression("1 +").is_err());
        assert!(eval_expression("(1 + 2").is_err());
        assert!(eval_expression("1 2 3").is_err());
        assert!(eval_expression("()").is_err());
    }
}
