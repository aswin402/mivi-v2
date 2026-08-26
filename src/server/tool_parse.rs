//! Tool-call parsing and validation: extract tool calls from model text,
//! validate arguments against tool JSON schemas, repair malformed JSON.
//!
//! Extracted from `helpers.rs` (server decomposition).

use crate::server::types::*;

pub fn parse_tool_calls(text: &str) -> Vec<ToolCallOut> {
    let mut calls = Vec::new();

    // First try: find <tool_call> blocks.
    let mut remaining = text;
    loop {
        if let Some(start) = remaining.find("<tool_call>") {
            let after_open = &remaining[start + "<tool_call>".len()..];
            if let Some(end) = after_open.find("</tool_call>") {
                let json_str = after_open[..end].trim();
                if let Some(call) = parse_single_tool_call(json_str) {
                    calls.push(call);
                }
                remaining = &after_open[end + "</tool_call>".len()..];
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // Fallback: find first top-level JSON object.
    if calls.is_empty() {
        if let Some(start) = text.find('{') {
            let candidate = &text[start..];
            // Track brace depth to find the matching top-level closing }.
            let mut depth: i32 = 0;
            for (i, ch) in candidate.char_indices() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            let json_str = &candidate[..=i];
                            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                                if let Some(obj) = val.as_object() {
                                    if let Some(tool_calls_arr) =
                                        obj.get("tool_calls").and_then(|v| v.as_array())
                                    {
                                        for item in tool_calls_arr {
                                            if let Some(call) = parse_single_tool_call_value(item) {
                                                calls.push(call);
                                            }
                                        }
                                    } else if let Some(call) = parse_single_tool_call_value(&val) {
                                        calls.push(call);
                                    }
                                }
                            }
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    calls
}

static TOOL_CALL_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1000);

pub fn parse_single_tool_call_value(val: &serde_json::Value) -> Option<ToolCallOut> {
    let obj = val.as_object()?;

    let function_obj = obj.get("function").and_then(|value| value.as_object());
    let name = obj
        .get("name")
        .and_then(|value| value.as_str())
        .or_else(|| obj.get("tool").and_then(|value| value.as_str()))
        .or_else(|| obj.get("call").and_then(|value| value.as_str()))
        .or_else(|| obj.get("action").and_then(|value| value.as_str()))
        .or_else(|| obj.get("function").and_then(|value| value.as_str()))
        .or_else(|| {
            function_obj
                .and_then(|function| function.get("name"))
                .and_then(|value| value.as_str())
        })?;

    let arguments_value = obj
        .get("arguments")
        .or_else(|| function_obj.and_then(|function| function.get("arguments")));
    let arguments = normalize_tool_arguments(arguments_value)?;

    let count = TOOL_CALL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(ToolCallOut {
        id: format!("call_{}", count),
        r#type: "function".to_string(),
        function: FunctionCallOut {
            name: name.to_string(),
            arguments,
        },
    })
}

pub fn parse_single_tool_call(json_str: &str) -> Option<ToolCallOut> {
    let mut fixed = json_str.trim().to_string();
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&fixed) {
        return parse_single_tool_call_value(&val);
    }

    // Try basic JSON healing
    fixed = fixed.replace(",}", "}");
    fixed = fixed.replace(",]", "]");

    let open_braces = fixed.chars().filter(|&c| c == '{').count();
    let close_braces = fixed.chars().filter(|&c| c == '}').count();
    if open_braces > close_braces {
        for _ in 0..(open_braces - close_braces) {
            fixed.push('}');
        }
    }

    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&fixed) {
        return parse_single_tool_call_value(&val);
    }

    None
}

#[cfg(test)]
pub fn parse_tool_calls_for_tools(text: &str, selected_tools: &[ToolDef]) -> Vec<ToolCallOut> {
    let parsed = parse_tool_calls(text);
    validate_tool_calls_for_tools(parsed, selected_tools).0
}

pub fn required_tool_args(tool: &ToolDef) -> Vec<String> {
    tool.function
        .parameters
        .as_ref()
        .and_then(|params| params.get("required"))
        .and_then(|required| required.as_array())
        .map(|required| {
            required
                .iter()
                .filter_map(|value| value.as_str())
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn check_value_type(value: &serde_json::Value, expected_type: &str) -> bool {
    match expected_type {
        "null" => value.is_null(),
        "string" => value.is_string() || value.is_number(),
        "number" => value.is_number(),
        "integer" => {
            value.is_number() && (!value.is_f64() || value.as_f64().unwrap().fract() == 0.0)
        }
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => true,
    }
}

fn validate_value_against_schema(
    value: &serde_json::Value,
    schema: &serde_json::Value,
) -> Result<(), String> {
    let Some(schema_obj) = schema.as_object() else {
        return Ok(());
    };

    if let Some(type_val) = schema_obj.get("type") {
        let is_valid_type = match type_val {
            serde_json::Value::String(expected_type) => check_value_type(value, expected_type),
            serde_json::Value::Array(types_arr) => types_arr.iter().any(|t| {
                if let Some(t_str) = t.as_str() {
                    check_value_type(value, t_str)
                } else {
                    false
                }
            }),
            _ => true,
        };
        if !is_valid_type {
            return Err(format!(
                "Value {} does not match type {:?}",
                value, type_val
            ));
        }
    }

    if value.is_array() {
        if let Some(items_schema) = schema_obj.get("items") {
            if let Some(arr) = value.as_array() {
                for (idx, item) in arr.iter().enumerate() {
                    validate_value_against_schema(item, items_schema)
                        .map_err(|e| format!("At index {}: {}", idx, e))?;
                }
            }
        }
    }

    if value.is_object() {
        if let Some(obj) = value.as_object() {
            if let Some(serde_json::Value::Array(required_fields)) = schema_obj.get("required") {
                for req_field in required_fields {
                    if let Some(req_str) = req_field.as_str() {
                        if !obj.contains_key(req_str) {
                            return Err(format!("Missing required property '{}'", req_str));
                        }
                    }
                }
            }
            if let Some(properties) = schema_obj.get("properties").and_then(|p| p.as_object()) {
                for (prop_name, prop_val) in obj {
                    if let Some(prop_schema) = properties.get(prop_name) {
                        validate_value_against_schema(prop_val, prop_schema)
                            .map_err(|e| format!("In property '{}': {}", prop_name, e))?;
                    } else if schema_obj
                        .get("additionalProperties")
                        .and_then(|value| value.as_bool())
                        == Some(false)
                    {
                        return Err(format!("Unknown property '{}'", prop_name));
                    }
                }
            }
        }
    }

    if let Some(format) = schema_obj.get("format").and_then(|value| value.as_str()) {
        if let Some(text) = value.as_str() {
            let valid = match format {
                "uri" | "url" => {
                    let lower = text.to_ascii_lowercase();
                    (lower.starts_with("http://") || lower.starts_with("https://"))
                        && !text.trim().is_empty()
                }
                _ => true,
            };
            if !valid {
                return Err(format!("String does not match format '{}'", format));
            }
        }
    }

    if let Some(serde_json::Value::Array(enum_values)) = schema_obj.get("enum") {
        if !enum_values.contains(value) {
            return Err(format!(
                "Value {} is not one of the allowed enums {:?}",
                value, enum_values
            ));
        }
    }

    Ok(())
}

pub fn validate_tool_call_arguments(call: &ToolCallOut, tool: &ToolDef) -> Result<(), String> {
    let required = required_tool_args(tool);
    let args_val = if call.function.arguments.trim().is_empty() {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        serde_json::from_str::<serde_json::Value>(&call.function.arguments)
            .map_err(|e| format!("Invalid JSON arguments: {}", e))?
    };

    let Some(args_obj) = args_val.as_object() else {
        return Err("Arguments must be a JSON object".to_string());
    };

    for req_field in required {
        if !args_obj.contains_key(&req_field) {
            return Err(format!("Missing required property '{}'", req_field));
        }
        if let Some(val) = args_obj.get(&req_field) {
            if let Some(s) = val.as_str() {
                if s.trim().is_empty() {
                    return Err(format!("Required property '{}' cannot be empty", req_field));
                }
            }
        }
    }

    if let Some(ref schema) = tool.function.parameters {
        if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
            for (name, property_schema) in properties {
                let is_string =
                    property_schema.get("type").and_then(|value| value.as_str()) == Some("string");
                if is_string {
                    if let Some(value) = args_obj.get(name).and_then(|value| value.as_str()) {
                        if value.trim().is_empty() {
                            return Err(format!("String property '{}' cannot be empty", name));
                        }
                    }
                }
            }
        }
        validate_value_against_schema(&args_val, schema)?;
    }

    Ok(())
}

pub fn call_has_required_args(call: &ToolCallOut, tool: &ToolDef) -> bool {
    validate_tool_call_arguments(call, tool).is_ok()
}

pub fn validate_tool_calls_for_tools(
    calls: Vec<ToolCallOut>,
    selected_tools: &[ToolDef],
) -> (Vec<ToolCallOut>, usize) {
    let original_len = calls.len();
    let accepted: Vec<ToolCallOut> = calls
        .into_iter()
        .filter(|call| {
            selected_tools
                .iter()
                .find(|tool| tool.function.name == call.function.name)
                .map(|tool| call_has_required_args(call, tool))
                .unwrap_or(false)
        })
        .collect();
    let rejected = original_len.saturating_sub(accepted.len());
    (accepted, rejected)
}

pub fn tool_names(tools: &[ToolDef]) -> Vec<String> {
    tools
        .iter()
        .map(|tool| tool.function.name.clone())
        .collect()
}

pub fn call_names(calls: &[ToolCallOut]) -> Vec<String> {
    calls
        .iter()
        .map(|call| call.function.name.clone())
        .collect()
}

pub fn normalize_tool_arguments(value: Option<&serde_json::Value>) -> Option<String> {
    match value {
        None => Some("{}".to_string()),
        Some(serde_json::Value::Object(obj)) => {
            let mut new_obj = obj.clone();
            if let Some(serde_json::Value::String(url)) = obj.get("url") {
                if (url.starts_with("http://") || url.starts_with("https://"))
                    && url.matches('/').count() == 2
                {
                    new_obj.insert(
                        "url".to_string(),
                        serde_json::Value::String(format!("{}/", url)),
                    );
                }
            }
            Some(serde_json::Value::Object(new_obj).to_string())
        }
        Some(serde_json::Value::String(text)) => {
            let repaired = repair_tool_argument_string(text);
            if let Some(rep_str) = repaired {
                if let Ok(mut val) = serde_json::from_str::<serde_json::Value>(&rep_str) {
                    if let Some(obj) = val.as_object_mut() {
                        if let Some(serde_json::Value::String(url)) = obj.get("url") {
                            if (url.starts_with("http://") || url.starts_with("https://"))
                                && url.matches('/').count() == 2
                            {
                                obj.insert(
                                    "url".to_string(),
                                    serde_json::Value::String(format!("{}/", url)),
                                );
                            }
                        }
                    }
                    return Some(val.to_string());
                }
            }
            None
        }
        _ => None,
    }
}

pub fn repair_tool_argument_string(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Some("{}".to_string());
    }

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return value.as_object().map(|_| value.to_string());
    }

    let mut fixed = trimmed.to_string();

    // Strip markdown code fences
    if fixed.contains("```") {
        fixed = fixed
            .replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "");
        fixed = fixed.trim().to_string();
    }

    // Replace single quotes with double quotes
    if !fixed.contains('"') && fixed.contains('\'') {
        fixed = fixed.replace('\'', "\"");
    }

    // Remove trailing commas before } or ]
    static TRAILING_COMMA_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = TRAILING_COMMA_RE.get_or_init(|| {
        regex::Regex::new(r",\s*([}\]])").expect("trailing-comma regex must compile")
    });
    fixed = re.replace_all(&fixed, "$1").to_string();

    // Try parsing the fixed version
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&fixed) {
        return value.as_object().map(|_| value.to_string());
    }

    // Try wrapping in braces
    let wrapped = format!("{{{}}}", fixed.trim());
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&wrapped) {
        return value.as_object().map(|_| value.to_string());
    }

    // Last resort: try to extract first JSON object
    if let Some(start) = fixed.find('{') {
        if let Some(end) = fixed.rfind('}') {
            let substr = &fixed[start..=end];
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(substr) {
                return value.as_object().map(|_| value.to_string());
            }
        }
    }

    None
}
