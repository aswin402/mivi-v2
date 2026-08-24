//! Tool selection for a request: capability taxonomy, intent-driven
//! inventory filtering, and web-research tool scoring.
//!
//! Extracted from `helpers.rs` (server decomposition).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::LazyLock;

use crate::constants::MAX_PROMPT_TOOLS;
use crate::server::prompt::{
    latest_user_prompt_text, mentions_agent_subject, normalize_user_prompt_text, text_tokens,
    user_prompt_text_parts,
};
use crate::server::types::*;
use crate::server::usage::value_text_for_usage;
use crate::tool_filter::filter_tools;

pub fn prompt_tools_for_request(req: &ChatCompletionRequest) -> Vec<ToolDef> {
    select_tools_for_request(req).selected
}

pub fn select_tools_for_request(req: &ChatCompletionRequest) -> ToolSelection {
    let latest_user_prompt = latest_user_prompt_text(req);
    let intent = classify_agent_intent(&latest_user_prompt);
    let tools = match req.tools.as_deref() {
        Some(tools) if !tools.is_empty() => tools,
        _ => return ToolSelection::empty(intent),
    };

    if matches!(req.tool_choice, Some(serde_json::Value::String(ref choice)) if choice == "required")
    {
        return ToolSelection {
            intent: AgentIntent::ToolCall,
            selected: tools.to_vec(),
            blocked: Vec::new(),
        };
    }

    if let Some(serde_json::Value::Object(ref obj)) = req.tool_choice {
        if let Some(serde_json::Value::Object(ref func)) = obj.get("function") {
            if let Some(serde_json::Value::String(ref name)) = func.get("name") {
                let matched: Vec<ToolDef> = tools
                    .iter()
                    .filter(|t| t.function.name == *name)
                    .cloned()
                    .collect();
                if !matched.is_empty() {
                    return ToolSelection {
                        intent: AgentIntent::ToolCall,
                        selected: matched,
                        blocked: Vec::new(),
                    };
                }
            }
        }
    }

    let decision = agent_decision_from_request(req);
    if decision.needs_tool() {
        return ToolSelection {
            intent: AgentIntent::ToolCall,
            selected: select_web_research_tools(tools, MAX_PROMPT_TOOLS),
            blocked: Vec::new(),
        };
    }

    if intent.is_inventory() {
        let inv = select_inventory_tools(intent, tools, &latest_user_prompt);
        if !inv.selected.is_empty() {
            return inv;
        }
        // No inventory tool found — return empty to fall through to regular chat
        return ToolSelection::empty(intent);
    }

    // Score-based filtering for relevance.
    // If no tools match the user prompt, return empty — the caller
    // will skip tool generation and fall through to regular chat.
    ToolSelection {
        intent,
        selected: filter_tools(&latest_user_prompt, tools, MAX_PROMPT_TOOLS),
        blocked: Vec::new(),
    }
}

pub fn blocked_tool_names(blocked: &[ToolBlock]) -> Vec<String> {
    blocked
        .iter()
        .map(|blocked| format!("{}:{}", blocked.name, blocked.reason))
        .collect()
}

pub fn selected_tool_roles(tools: &[ToolDef]) -> Vec<String> {
    tools
        .iter()
        .map(|tool| format!("{}:{:?}", tool.function.name, classify_tool_role(tool)))
        .collect()
}

pub fn select_inventory_tools(
    intent: AgentIntent,
    tools: &[ToolDef],
    latest_user_prompt: &str,
) -> ToolSelection {
    let selected = tools
        .iter()
        .find(|tool| tool_is_inventory_for_intent(tool, intent, latest_user_prompt))
        .cloned()
        .into_iter()
        .collect::<Vec<_>>();
    let selected_name = selected.first().map(|tool| tool.function.name.as_str());

    let blocked = tools
        .iter()
        .filter(|tool| selected_name != Some(tool.function.name.as_str()))
        .filter_map(|tool| inventory_block_reason(tool, intent).map(|reason| (tool, reason)))
        .map(|(tool, reason)| ToolBlock {
            name: tool.function.name.clone(),
            reason,
        })
        .collect();

    ToolSelection {
        intent,
        selected,
        blocked,
    }
}

pub fn classify_agent_intent(query: &str) -> AgentIntent {
    let q = query.to_ascii_lowercase();
    if q.trim().is_empty() {
        return AgentIntent::Chat;
    }

    let mentions_agent_subject = mentions_agent_subject(&q);
    let asks_inventory = q.contains("what")
        || q.contains("which")
        || q.contains("list")
        || q.contains("show")
        || q.contains("tell me")
        || q.contains("inventory")
        || q.contains("available")
        || q.contains("loaded")
        || q.contains("can this agent")
        || q.contains("can you do")
        || q.contains("can u do")
        || q.contains("able to do");
    let asks_capability_subject = mentions_agent_subject
        && (q.contains("do")
            || q.contains("handle")
            || q.contains("support")
            || q.contains("use")
            || q.contains("available"));

    if !asks_inventory && !asks_capability_subject {
        return AgentIntent::Chat;
    }

    if q.contains("use the") || q.contains("call the") || q.contains("run ") {
        return AgentIntent::ToolCall;
    }

    if q.contains("skill") || q.contains("skills") {
        AgentIntent::SkillInventory
    } else if q.contains("mcp") || q.contains("mcps") {
        AgentIntent::McpInventory
    } else if q.contains("tool") || q.contains("tools") {
        AgentIntent::ToolInventory
    } else if q.contains("feature")
        || q.contains("features")
        || q.contains("capabilit")
        || asks_capability_subject
    {
        AgentIntent::CapabilityInventory
    } else {
        AgentIntent::Chat
    }
}
pub fn normalize_keyword_map(
    values: BTreeMap<String, Vec<String>>,
) -> BTreeMap<String, Vec<String>> {
    values
        .into_iter()
        .map(|(key, values)| {
            let key = key.trim().to_ascii_lowercase();
            let mut values: Vec<String> = values
                .into_iter()
                .flat_map(|value| text_tokens(&value))
                .filter(|value| !value.is_empty())
                .collect();
            values.sort_unstable();
            values.dedup();
            (key, values)
        })
        .filter(|(key, _)| !key.is_empty())
        .collect()
}

pub fn normalize_marker_map(
    values: BTreeMap<String, Vec<String>>,
) -> BTreeMap<String, Vec<String>> {
    values
        .into_iter()
        .map(|(key, values)| {
            let key = key.trim().to_ascii_lowercase();
            let values = normalize_marker_list(values);
            (key, values)
        })
        .filter(|(key, values)| !key.is_empty() && !values.is_empty())
        .collect()
}

pub fn normalize_marker_list(values: Vec<String>) -> Vec<String> {
    let mut values: Vec<String> = values
        .into_iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect();
    values.sort_unstable();
    values.dedup();
    values
}

pub fn normalize_priority_list(values: Vec<String>) -> Vec<String> {
    values.into_iter().fold(Vec::new(), |mut acc, value| {
        let value = value.trim().to_ascii_lowercase();
        if !value.is_empty() && !acc.iter().any(|existing| existing == &value) {
            acc.push(value);
        }
        acc
    })
}

pub fn parse_capability_config(text: &str) -> Result<CapabilityConfig, serde_json::Error> {
    let mut config = serde_json::from_str::<CapabilityConfig>(text)?;
    config.aliases = normalize_keyword_map(config.aliases);
    config.tool_taxonomy = normalize_keyword_map(config.tool_taxonomy);
    config.tool_error_markers = normalize_marker_list(config.tool_error_markers);
    config.tool_salient_markers = normalize_marker_list(config.tool_salient_markers);
    config.tool_error_categories = normalize_marker_map(config.tool_error_categories);
    config.tool_error_category_priority =
        normalize_priority_list(config.tool_error_category_priority);
    Ok(config)
}

pub fn load_capability_config(path: &Path) -> CapabilityConfig {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| parse_capability_config(&text).ok())
        .unwrap_or_default()
}

pub fn capability_config() -> &'static CapabilityConfig {
    static CONFIG: LazyLock<CapabilityConfig> =
        LazyLock::new(|| load_capability_config(Path::new("configs/capabilities.json")));
    &CONFIG
}

pub fn extract_first_url(text: &str) -> Option<String> {
    text.split_whitespace()
        .find(|part| part.starts_with("http://") || part.starts_with("https://"))
        .map(|part| {
            part.trim_end_matches(|ch: char| matches!(ch, '.' | ',' | ')' | ']' | '}' | '!' | '?'))
                .to_string()
        })
        .filter(|url| !url.is_empty())
}

pub fn looks_like_research_request(text: &str) -> bool {
    let tokens = text_tokens(text);
    tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "research" | "read" | "docs" | "documentation" | "tell" | "about" | "summarize"
        )
    })
}

pub fn previous_user_url(req: &ChatCompletionRequest) -> Option<String> {
    let mut skipped_latest_user = false;
    for msg in req.messages.iter().rev() {
        if msg.role != "user" {
            continue;
        }
        if !skipped_latest_user {
            skipped_latest_user = true;
            continue;
        }
        let text = user_prompt_text_parts(msg)
            .into_iter()
            .map(|part| normalize_user_prompt_text(&part))
            .find(|part| !part.is_empty())
            .unwrap_or_default();
        if let Some(url) = extract_first_url(&text) {
            return Some(url);
        }
    }
    None
}

pub fn agent_decision_from_request(req: &ChatCompletionRequest) -> AgentDecision {
    let latest = latest_user_prompt_text(req);
    if let Some(url) = extract_first_url(&latest) {
        if looks_like_research_request(&latest) {
            return AgentDecision {
                intent: AgentDecisionIntent::WebResearch,
                subject: latest,
                url: Some(url),
            };
        }
    }

    if looks_like_research_request(&latest) {
        if let Some(url) = previous_user_url(req) {
            return AgentDecision {
                intent: AgentDecisionIntent::WebResearch,
                subject: latest,
                url: Some(url),
            };
        }
    }

    AgentDecision::chat(latest)
}

#[allow(dead_code)]
pub fn tool_matches_taxonomy(
    name: &str,
    description: &str,
    category: &str,
    config: &CapabilityConfig,
) -> bool {
    let Some(keywords) = config.tool_taxonomy.get(category) else {
        return false;
    };
    let haystack = format!(
        "{} {}",
        name.to_ascii_lowercase(),
        description.to_ascii_lowercase()
    );
    let tokens = text_tokens(&haystack);
    keywords
        .iter()
        .any(|keyword| tokens.iter().any(|token| token == keyword) || haystack.contains(keyword))
}

pub fn web_research_tool_score(tool: &ToolDef) -> isize {
    let config = capability_config();
    let positives = config
        .tool_taxonomy
        .get("web")
        .map(|keywords| {
            let schema = tool_schema_text(tool);
            let schema_tokens = text_tokens(&schema);
            keywords
                .iter()
                .filter(|keyword| {
                    schema_tokens
                        .iter()
                        .any(|schema_token| schema_token == *keyword)
                        || schema.contains(keyword.as_str())
                })
                .count() as isize
        })
        .unwrap_or(0);
    let local_penalty = config
        .tool_taxonomy
        .get("local_exclude")
        .map(|keywords| {
            let schema = tool_schema_text(tool);
            let schema_tokens = text_tokens(&schema);
            keywords
                .iter()
                .filter(|keyword| {
                    schema_tokens
                        .iter()
                        .any(|schema_token| schema_token == *keyword)
                        || schema.contains(keyword.as_str())
                })
                .count() as isize
        })
        .unwrap_or(0);

    positives * 4 - local_penalty * 3
}

pub fn select_web_research_tools(tools: &[ToolDef], max_tools: usize) -> Vec<ToolDef> {
    let mut scored: Vec<(usize, isize, ToolDef)> = tools
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, tool)| (idx, web_research_tool_score(&tool), tool))
        .filter(|(_, score, _)| *score > 0)
        .collect();
    scored.sort_by(|(left_idx, left_score, _), (right_idx, right_score, _)| {
        right_score
            .cmp(left_score)
            .then_with(|| left_idx.cmp(right_idx))
    });
    scored
        .into_iter()
        .take(max_tools)
        .map(|(_, _, tool)| tool)
        .collect()
}

pub fn tool_schema_text(tool: &ToolDef) -> String {
    format!(
        "{} {}",
        tool.function.name.to_ascii_lowercase(),
        tool.function
            .description
            .as_deref()
            .unwrap_or("")
            .to_ascii_lowercase()
    )
}

pub fn classify_tool_role(tool: &ToolDef) -> ToolRole {
    let text = tool_schema_text(tool);
    let has_broad_inventory_signal = text.contains("capabilit") || text.contains("introspection");
    let has_inventory_signal = text.contains("inventory")
        || has_broad_inventory_signal
        || text.contains("available")
        || text.contains("registered")
        || text.contains("loaded");
    let has_diagnostic_signal = text.contains("diagnose")
        || text.contains("diagnostic")
        || text.contains("debug")
        || text.contains("troubleshoot")
        || text.contains("repair")
        || text.contains("fix");
    let has_management_signal = text.contains("manage")
        || text.contains("configure")
        || text.contains("configuration")
        || text.contains("start server")
        || text.contains("stop server")
        || text.contains("restart server");
    let has_action_signal = has_management_signal
        || text.contains("spawn")
        || text.contains("delegate")
        || text.contains("create subagent")
        || text.contains("run task")
        || text.contains("execute task");
    let has_resource_signal = text.contains("resource") || text.contains("template");
    let has_mcp_signal = text.contains("mcp");
    let has_skill_signal = text.contains("skill");
    let has_tool_signal =
        text.contains("tool") || text.contains("function") || text.contains("schema");

    if has_mcp_signal && has_resource_signal {
        ToolRole::McpResource
    } else if has_diagnostic_signal {
        ToolRole::Diagnostic
    } else if has_action_signal && !(text.contains("inventory") || text.contains("introspection")) {
        ToolRole::Action
    } else if has_broad_inventory_signal {
        ToolRole::Inventory
    } else if has_mcp_signal && has_inventory_signal {
        ToolRole::McpInventory
    } else if has_skill_signal && has_inventory_signal {
        ToolRole::SkillInventory
    } else if has_inventory_signal && has_tool_signal {
        ToolRole::Inventory
    } else if has_inventory_signal && !has_action_signal {
        ToolRole::Inventory
    } else if has_action_signal {
        ToolRole::Action
    } else {
        ToolRole::General
    }
}

pub fn is_resource_template_listing_tool(tool: &ToolDef) -> bool {
    classify_tool_role(tool) == ToolRole::McpResource
}

pub fn is_diagnostic_or_action_tool(tool: &ToolDef) -> bool {
    matches!(
        classify_tool_role(tool),
        ToolRole::Diagnostic | ToolRole::Action
    )
}

pub fn is_skill_inventory_tool(tool: &ToolDef) -> bool {
    classify_tool_role(tool) == ToolRole::SkillInventory
}

pub fn is_mcp_inventory_tool(tool: &ToolDef) -> bool {
    classify_tool_role(tool) == ToolRole::McpInventory
}

pub fn is_tool_inventory_tool(tool: &ToolDef) -> bool {
    classify_tool_role(tool) == ToolRole::Inventory
}

pub fn is_broad_inventory_tool(tool: &ToolDef) -> bool {
    classify_tool_role(tool) == ToolRole::Inventory
}

pub fn tool_is_inventory_for_intent(
    tool: &ToolDef,
    intent: AgentIntent,
    latest_user_prompt: &str,
) -> bool {
    match intent {
        AgentIntent::SkillInventory => {
            is_skill_inventory_tool(tool) || is_broad_inventory_tool(tool)
        }
        AgentIntent::McpInventory => is_mcp_inventory_tool(tool) || is_broad_inventory_tool(tool),
        AgentIntent::ToolInventory => is_tool_inventory_tool(tool) || is_broad_inventory_tool(tool),
        AgentIntent::CapabilityInventory => {
            let query = latest_user_prompt.to_ascii_lowercase();
            let wants_agent_scope = mentions_agent_subject(&query);
            is_broad_inventory_tool(tool)
                || (wants_agent_scope
                    && (is_tool_inventory_tool(tool)
                        || is_skill_inventory_tool(tool)
                        || is_mcp_inventory_tool(tool)))
        }
        AgentIntent::Chat | AgentIntent::ToolCall => false,
    }
}

pub fn inventory_block_reason(tool: &ToolDef, intent: AgentIntent) -> Option<&'static str> {
    if intent == AgentIntent::McpInventory && is_resource_template_listing_tool(tool) {
        return Some("mcp_resource_not_inventory");
    }
    if intent == AgentIntent::ToolInventory && is_skill_inventory_tool(tool) {
        return Some("skill_inventory_not_tool_inventory");
    }
    if is_diagnostic_or_action_tool(tool) {
        return Some("diagnostic_or_action_tool_not_inventory");
    }
    None
}

pub fn asks_agent_inventory(query: &str) -> bool {
    classify_agent_intent(query).is_inventory()
}

pub fn tool_is_inventory_for_query(tool: &ToolDef, query: &str) -> bool {
    let intent = classify_agent_intent(query);
    tool_is_inventory_for_intent(tool, intent, query)
}

#[allow(dead_code)]
pub fn tool_error_category_with_config(text: &str, config: &CapabilityConfig) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let mut categories = config.tool_error_category_priority.clone();
    for category in config.tool_error_categories.keys() {
        if !categories.iter().any(|existing| existing == category) {
            categories.push(category.clone());
        }
    }

    categories.into_iter().find(|category| {
        config
            .tool_error_categories
            .get(category)
            .map(|markers| markers.iter().any(|marker| lower.contains(marker)))
            .unwrap_or(false)
    })
}
