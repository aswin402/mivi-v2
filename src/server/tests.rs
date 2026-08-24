//! Integration-style tests for the server module. Moved verbatim from
//! `helpers.rs` (server decomposition); imports mirror the old `use super::*`
//! which now spans the decomposed modules.

#![cfg(test)]

use std::path::PathBuf;

use super::*;
use crate::brain::EdgeBrain;
use crate::constants::MIVI_CHAT_SYSTEM_PROMPT;
use crate::orchestrator::AgentOrchestrator;
use crate::router::NeedleRouter;
use crate::server::chat::*;
use crate::server::middleware::*;
use crate::server::startup::*;
use crate::server::streaming::*;
use crate::server::tool_generate::*;
use crate::server::tool_parse::*;
use std::sync::Arc;

use axum::extract::State;
use serde_json::json;

pub fn tool_request(
    content: &str,
    tool_choice: Option<serde_json::Value>,
) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: Some(MODEL_NAME.to_string()),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: json!(content),
            tool_call_id: None,
            tool_calls: None,
        }],
        stream: None,
        tools: Some(vec![ToolDef {
            function: FunctionDef {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: None,
            },
            r#type: "function".to_string(),
        }]),
        tool_choice,
        max_tokens: None,
        stop: None,
        seed: None,
        response_format: None,
        stream_options: None,
        parallel_tool_calls: None,
        reasoning_effort: None,
        temperature: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        user: None,
        logit_bias: None,
        logprobs: None,
        top_logprobs: None,
        n: None,
        service_tier: None,
    }
}

pub fn server_tool(name: &str, description: &str) -> ToolDef {
    ToolDef {
        function: FunctionDef {
            name: name.to_string(),
            description: Some(description.to_string()),
            parameters: None,
        },
        r#type: "function".to_string(),
    }
}

#[test]
pub fn chat_request_accepts_openai_compatibility_fields() {
    let req: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "mivi",
        "messages": [{"role":"user", "content":"hi"}],
        "max_tokens": 32,
        "stop": ["END"],
        "seed": 7,
        "parallel_tool_calls": false,
        "reasoning_effort": "low",
        "stream_options": {"include_usage": true},
        "response_format": {"type":"json_object"}
    }))
    .expect("request should parse compatibility fields");

    assert_eq!(req.max_tokens, Some(32));
    assert_eq!(req.seed, Some(7));
    assert_eq!(req.reasoning_effort.as_deref(), Some("low"));
    assert_eq!(req.parallel_tool_calls, Some(false));
    assert_eq!(req.stop, Some(json!(["END"])));
    assert_eq!(req.stream_options.as_ref().unwrap()["include_usage"], true);
    assert_eq!(response_format_type(&req), Some("json_object".to_string()));
}

#[test]
pub fn test_grammar_dynamic_compilation() {
    let req = tool_request("What is the weather in Tokyo?", None);
    let path_opt = get_grammar_path(&req);
    assert!(path_opt.is_some(), "get_grammar_path returned None");
    let path = path_opt.unwrap();
    let content = std::fs::read_to_string(&path).expect("failed to read grammar file");

    // Assert that get_weather is present in the grammar content, proving the replacement occurred
    assert!(
        content.contains("get_weather"),
        "tool name was not dynamically injected: {}",
        content
    );

    let mut cleaned_str = String::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }
        if let Some(idx) = line.find(" #") {
            cleaned_str.push_str(&line[..idx]);
        } else {
            cleaned_str.push_str(line);
        }
        cleaned_str.push('\n');
    }
    let res = schoolmarm::Grammar::new(cleaned_str.trim());
    assert!(
        res.is_ok(),
        "schoolmarm failed to parse generated grammar: {:?}",
        res.err()
    );
}

#[test]
pub fn responses_request_passes_compatibility_fields_to_chat_request() {
    let req: ResponsesRequest = serde_json::from_value(json!({
        "model": "mivi",
        "input": "hello",
        "max_output_tokens": 64,
        "parallel_tool_calls": true,
        "response_format": {"type":"json_object"},
        "stream_options": {"include_usage": true},
        "reasoning": {"effort":"medium"}
    }))
    .expect("responses request should parse compatibility fields");

    let chat = responses_request_to_chat_request(req);

    assert_eq!(chat.max_tokens, Some(64));
    assert_eq!(chat.parallel_tool_calls, Some(true));
    assert_eq!(chat.reasoning_effort.as_deref(), Some("medium"));
    assert!(include_stream_usage(&chat));
    assert_eq!(response_format_type(&chat), Some("json_object".to_string()));
}

#[test]
pub fn json_response_format_wraps_verified_answer_as_json() {
    let mut req = tool_request("what ai model are you", None);
    req.tools = None;
    req.response_format = Some(json!({"type":"json_object"}));

    let answer = apply_response_format(
        "I am MIVI, exposed to agents as the local OpenAI-compatible model `mivi`.".to_string(),
        &req,
    )
    .expect("json format should be supported");
    let parsed: serde_json::Value = serde_json::from_str(&answer).expect("valid json");

    assert!(parsed["answer"].as_str().unwrap().contains("MIVI"));
    assert!(parsed["answer"].as_str().unwrap().contains("`mivi`"));
}

#[test]
pub fn strict_json_schema_response_format_is_accepted() {
    let mut req = tool_request("what ai model are you", None);
    req.response_format = Some(json!({
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "response": { "type": "string" }
                }
            }
        }
    }));

    assert!(validate_response_format(&req).is_ok());
    let schema = extract_json_schema(&req).expect("schema extraction");
    assert!(schema.contains("properties"));
}

#[test]
pub fn llama_tokenize_output_counts_token_ids_without_prompt_metadata() {
    let output = "main: tokenizing prompt\n[1, 29871, 15043, 13]\n[ Prompt: 4 tokens ]";

    assert_eq!(count_llama_tokenize_output(output), Some(4));
}

#[test]
pub fn token_counter_config_uses_catalog_reasoner_when_model_env_missing() {
    let catalog = crate::model_catalog::ModelCatalog::from_json(
        r#"{
          "external_model": "mivi",
          "models": [
            {
              "id": "reasoner",
              "role": "reasoner",
              "backend": "llama-cli",
              "path": "models/catalog-reasoner.gguf",
              "context_tokens": 4096,
              "ram_mb_estimate": 512,
              "enabled": true
            }
          ]
        }"#,
    )
    .expect("catalog should parse");

    let config =
        TokenCounterConfig::from_sources(Some("llama-tokenize"), None, None, Some(&catalog));

    assert_eq!(
        config.backend,
        TokenCounterBackend::LlamaCpp {
            command: PathBuf::from("llama-tokenize"),
            model: PathBuf::from("models/catalog-reasoner.gguf"),
        }
    );
}

#[test]
pub fn token_counter_config_prefers_explicit_tokenizer_model_over_catalog() {
    let catalog = crate::model_catalog::ModelCatalog::from_json(
        r#"{
          "external_model": "mivi",
          "models": [
            {
              "id": "reasoner",
              "role": "reasoner",
              "backend": "llama-cli",
              "path": "models/catalog-reasoner.gguf",
              "context_tokens": 4096,
              "ram_mb_estimate": 512,
              "enabled": true
            }
          ]
        }"#,
    )
    .expect("catalog should parse");

    let config = TokenCounterConfig::from_sources(
        Some("llama-tokenize"),
        Some("models/explicit-tokenizer.gguf"),
        Some("models/env-reasoner.gguf"),
        Some(&catalog),
    );

    assert_eq!(
        config.backend,
        TokenCounterBackend::LlamaCpp {
            command: PathBuf::from("llama-tokenize"),
            model: PathBuf::from("models/explicit-tokenizer.gguf"),
        }
    );
}

#[test]
pub fn token_counter_config_uses_external_backend_only_when_command_and_model_exist() {
    let configured =
        TokenCounterConfig::from_sources(Some("llama-tokenize"), Some("models/a.gguf"), None, None);
    let fallback = TokenCounterConfig::from_sources(Some("llama-tokenize"), None, None, None);

    assert!(matches!(
        configured.backend,
        TokenCounterBackend::LlamaCpp { .. }
    ));
    assert!(matches!(fallback.backend, TokenCounterBackend::Cheap));
}

#[test]
pub fn cheap_token_counter_remains_fallback_for_plain_text() {
    let counter = TokenCounterConfig::default().counter();

    assert_eq!(counter.count_tokens("hello, world!"), 4);
}

#[test]
pub fn estimated_usage_counts_prompt_completion_and_total_tokens() {
    let req = tool_request("hello world", None);

    let usage = estimated_usage_for_text(&req, "hi there");

    assert!(usage.prompt_tokens >= 2);
    assert_eq!(usage.completion_tokens, 2);
    assert_eq!(
        usage.total_tokens,
        usage.prompt_tokens + usage.completion_tokens
    );
}

#[test]
pub fn non_stream_chat_response_serializes_openai_usage() {
    let response = ChatCompletionResponse {
        id: "chatcmpl-test".to_string(),
        object: "chat.completion".to_string(),
        created: 123,
        model: MODEL_NAME.to_string(),
        choices: vec![ChoiceOut {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: "hello".to_string(),
                refusal: None,
                reasoning_content: None,
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: Some(UsageInfo::new(3, 1)),
        system_fingerprint: Some("fp_mivi".to_string()),
    };

    let value = serde_json::to_value(response).expect("serializable response");

    assert_eq!(value["usage"]["prompt_tokens"], 3);
    assert_eq!(value["usage"]["completion_tokens"], 1);
    assert_eq!(value["usage"]["total_tokens"], 4);
}

#[test]
pub fn responses_response_carries_chat_usage() {
    let chat = ChatCompletionResponse {
        id: "chatcmpl-test".to_string(),
        object: "chat.completion".to_string(),
        created: 123,
        model: MODEL_NAME.to_string(),
        choices: vec![ChoiceOut {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: "streamed text".to_string(),
                refusal: None,
                reasoning_content: None,
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        usage: Some(UsageInfo::new(4, 2)),
        system_fingerprint: Some("fp_mivi".to_string()),
    };

    let response = responses_response_from_chat(chat);

    assert_eq!(response.usage.unwrap().total_tokens, 6);
}

#[test]
pub fn responses_string_input_maps_to_chat_request() {
    let req = ResponsesRequest {
        model: Some(MODEL_NAME.to_string()),
        input: ResponsesInput::Text("hello from responses".to_string()),
        stream: None,
        tools: None,
        tool_choice: None,
        max_output_tokens: None,
        stop: None,
        seed: None,
        response_format: None,
        stream_options: None,
        parallel_tool_calls: None,
        reasoning_effort: None,
        reasoning: None,
        temperature: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        user: None,
    };

    let chat = responses_request_to_chat_request(req);

    assert_eq!(chat.model.as_deref(), Some(MODEL_NAME));
    assert_eq!(chat.messages.len(), 1);
    assert_eq!(chat.messages[0].role, "user");
    assert_eq!(chat.messages[0].content, json!("hello from responses"));
}

#[test]
pub fn responses_message_array_input_maps_to_chat_request() {
    let req = ResponsesRequest {
        model: Some(MODEL_NAME.to_string()),
        input: ResponsesInput::Messages(vec![ResponsesInputMessage {
            role: "user".to_string(),
            content: json!([{"type":"input_text","text":"research this"}]),
        }]),
        stream: Some(false),
        tools: Some(vec![server_tool("webfetch", "Fetch a URL from the web")]),
        tool_choice: None,
        max_output_tokens: None,
        stop: None,
        seed: None,
        response_format: None,
        stream_options: None,
        parallel_tool_calls: None,
        reasoning_effort: None,
        reasoning: None,
        temperature: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        user: None,
    };

    let chat = responses_request_to_chat_request(req);

    assert_eq!(chat.messages.len(), 1);
    assert_eq!(
        chat.messages[0].content,
        json!([{"type":"text","text":"research this"}])
    );
    assert!(chat.tools.as_ref().unwrap()[0].function.name == "webfetch");
}

#[test]
pub fn chat_response_maps_to_responses_output_text() {
    let chat = ChatCompletionResponse {
        id: "chatcmpl-test".to_string(),
        object: "chat.completion".to_string(),
        created: 123,
        model: MODEL_NAME.to_string(),
        usage: None,
        choices: vec![ChoiceOut {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: "answer text".to_string(),
                refusal: None,
                reasoning_content: Some("summary".to_string()),
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: "stop".to_string(),
        }],
        system_fingerprint: Some("fp_mivi".to_string()),
    };

    let response = responses_response_from_chat(chat);

    assert_eq!(response.object, "response");
    assert_eq!(response.status, "completed");
    assert_eq!(response.output[0].r#type, "message");
    assert_eq!(response.output[0].content[0].r#type, "output_text");
    assert_eq!(response.output[0].content[0].text, "answer text");
}

#[test]
pub fn chat_prompt_injects_agent_contract_with_tool_summary() {
    let mut req = tool_request("what can this agent do", None);
    req.tools = Some(vec![
        server_tool(
            "agent_capabilities",
            "Introspection: available tools and skills",
        ),
        server_tool("read", "Read files"),
    ]);

    let prompt = build_chat_prompt(&req);

    assert!(prompt.contains("Agent contract:"));
    assert!(prompt.contains("External model identity is `mivi`"));
    assert!(prompt
        .contains("Current prompt exposes 1 selected callable tool schemas: agent_capabilities"));
    assert!(prompt.contains("<tools>"));
    assert!(prompt.contains("\"agent_capabilities\""));
    assert!(!prompt.contains("\"read\""));
}

#[test]
pub fn chat_prompt_wraps_calling_agent_system_prompt() {
    let mut req = tool_request("hello", None);
    req.tools = None;
    req.messages.insert(
        0,
        ChatMessage {
            role: "system".to_string(),
            content: json!("You are the calling agent. Use its policies."),
            tool_call_id: None,
            tool_calls: None,
        },
    );

    let prompt = build_chat_prompt(&req);

    assert!(prompt.contains("Agent contract:"));
    assert!(prompt.contains("You are the calling agent. Use its policies."));
    assert!(prompt.contains("Current prompt exposes no selected callable tool schemas."));
}

#[test]
pub fn tool_prompt_filters_irrelevant_opencode_tools() {
    let mut req = tool_request("please use apply_patch to edit src/main.rs", None);
    let mut tools = vec![
        server_tool("read", "Read a file"),
        server_tool("apply_patch", "Edit files by applying a patch"),
        server_tool("bash", "Run command"),
    ];
    for idx in 0..40 {
        tools.push(server_tool(
            &format!("irrelevant_tool_{idx}"),
            "Unrelated plugin action",
        ));
    }
    req.tools = Some(tools);

    let prompt = build_chat_prompt(&req);

    assert!(prompt.contains("<tools>"));
    assert!(prompt.contains("\"apply_patch\""));
    assert!(!prompt.contains("irrelevant_tool_17"));
}

#[test]
pub fn tools_available_does_not_force_tool_generation_for_plain_chat() {
    let req = tool_request("hi", None);
    assert!(has_tool_involvement(&req));
}

#[test]
pub fn auto_tools_do_not_force_generation_for_general_chat() {
    let mut req = tool_request("hey whats new", None);
    req.tools = Some(vec![
        server_tool("schedule_job", "Schedule a recurring job"),
        server_tool("create_subagent", "Create a subagent"),
    ]);

    assert!(!should_use_tool_path(&req, "hey whats new"));
}

#[test]
pub fn chat_intent_does_not_call_selected_auto_tools() {
    let req = tool_request("hey whats new", None);
    let selection = ToolSelection {
        intent: AgentIntent::Chat,
        selected: vec![server_tool("schedule_job", "Schedule a recurring job")],
        blocked: Vec::new(),
    };

    assert!(!should_generate_tool_calls(
        &req,
        "hey whats new",
        &selection
    ));
}

#[test]
pub fn explicit_stop_request_can_call_selected_tool() {
    let req = tool_request("stop scheduled job 1", None);
    let selection = ToolSelection {
        intent: AgentIntent::Chat,
        selected: vec![server_tool("remove_job", "Remove a scheduled job")],
        blocked: Vec::new(),
    };

    assert!(should_generate_tool_calls(
        &req,
        "stop the cron job",
        &selection
    ));
}

#[test]
pub fn code_capability_question_does_not_enter_tool_generation() {
    let mut req = tool_request("so is u can write codes", None);
    req.tools = Some(vec![server_tool("write", "Write a file to the workspace")]);

    assert!(has_tool_involvement(&req));
}

pub fn server_tool_with_params(
    name: &str,
    description: &str,
    properties: serde_json::Value,
) -> ToolDef {
    ToolDef {
        function: FunctionDef {
            name: name.to_string(),
            description: Some(description.to_string()),
            parameters: Some(json!({
                "type": "object",
                "properties": properties,
            })),
        },
        r#type: "function".to_string(),
    }
}

#[test]
pub fn tool_taxonomy_classifies_web_and_file_tools_from_config() {
    let config = parse_capability_config(
        r#"{
            "aliases": {},
            "tool_taxonomy": {
                "web": ["web", "url", "browser"],
                "file": ["file", "path", "workspace"]
            },
            "tool_error_markers": ["error", "failed"],
            "tool_salient_markers": ["error", "failed", "status"]
        }"#,
    )
    .expect("valid capability config");

    assert!(tool_matches_taxonomy(
        "webfetch",
        "Fetch a URL from the web",
        "web",
        &config
    ));
    assert!(tool_matches_taxonomy(
        "read_file",
        "Read a workspace path",
        "file",
        &config
    ));
    assert!(!tool_matches_taxonomy(
        "read_file",
        "Read a workspace path",
        "web",
        &config
    ));
}

#[test]
pub fn tool_call_missing_required_argument_is_rejected() {
    let tool = server_tool_with_params(
        "search_and_read",
        "Search and read web pages",
        json!({"url": {"type": "string"}, "query": {"type": "string"}}),
    );
    let mut tool = tool;
    tool.function.parameters = Some(json!({
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "query": {"type": "string"}
        },
        "required": ["url", "query"]
    }));
    let raw = r#"<tool_call>{"name":"search_and_read","arguments":{"url":"https://hono.dev/"}}</tool_call>"#;

    let calls = parse_tool_calls_for_tools(raw, &[tool]);

    assert!(calls.is_empty());
}

#[test]
pub fn url_research_request_enters_tool_generation_and_selects_web_tool() {
    let mut req = tool_request("so https://hono.dev/ research about this and tell me", None);
    req.tools = Some(vec![
        server_tool_with_params(
            "searchxyz_search_and_read",
            "Search and read web pages from the internet",
            json!({"query": {"type": "string"}, "url": {"type": "string"}}),
        ),
        server_tool("bash", "Run shell commands"),
        server_tool("read_file", "Read a workspace file"),
    ]);

    assert!(has_tool_involvement(&req));
    let selected = prompt_tools_for_request(&req);
    assert_eq!(tool_names(&selected), vec!["searchxyz_search_and_read"]);
}

#[test]
pub fn agent_intent_classifies_inventory_queries_without_platform_phrases() {
    assert_eq!(
        classify_agent_intent("what tools are available here"),
        AgentIntent::ToolInventory
    );
    assert_eq!(
        classify_agent_intent("which MCP servers can this agent use"),
        AgentIntent::McpInventory
    );
    assert_eq!(
        classify_agent_intent("list skills loaded for this task"),
        AgentIntent::SkillInventory
    );
    assert_eq!(classify_agent_intent("1+1 is what"), AgentIntent::Chat);
}

#[test]
pub fn tool_selection_trace_blocks_action_tools_for_tool_inventory() {
    let mut req = tool_request("what tools are available here", None);
    req.tools = Some(vec![
        server_tool("spawn_agent", "Create or delegate to a subagent"),
        server_tool("get_available_skills", "List available skills"),
        server_tool(
            "agent_capabilities",
            "Return inventory of available tools and features",
        ),
    ]);

    let selection = select_tools_for_request(&req);

    assert_eq!(selection.intent, AgentIntent::ToolInventory);
    assert_eq!(tool_names(&selection.selected), vec!["agent_capabilities"]);
    assert!(selection
        .blocked
        .iter()
        .any(|blocked| blocked.name == "spawn_agent"));
    assert!(selection
        .blocked
        .iter()
        .any(|blocked| blocked.name == "get_available_skills"));
}

#[test]
pub fn mcp_inventory_selection_blocks_resource_template_tools() {
    let mut req = tool_request("which MCP servers can this agent use", None);
    req.tools = Some(vec![
        server_tool("list_mcp_resource_templates", "List MCP resource templates"),
        server_tool("list_mcp_resources", "List MCP resources"),
        server_tool("mcp_inventory", "Inventory available MCP servers"),
    ]);

    let selection = select_tools_for_request(&req);

    assert_eq!(selection.intent, AgentIntent::McpInventory);
    assert_eq!(tool_names(&selection.selected), vec!["mcp_inventory"]);
    assert_eq!(selection.blocked.len(), 2);
}

#[test]
pub fn assistant_think_history_is_not_reprompted_to_model() {
    let mut req = tool_request("so whats new", None);
    req.tools = None;
    req.messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: json!("hii"),
            tool_call_id: None,
            tool_calls: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: json!(
                "<think>Classified request as chat; route streaming.</think>

Hello!"
            ),
            tool_call_id: None,
            tool_calls: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: json!("so whats new"),
            tool_call_id: None,
            tool_calls: None,
        },
    ];

    let prompt = build_chat_prompt(&req);

    assert!(!prompt.contains("<think>"));
    assert!(!prompt.contains("Classified request as chat"));
    assert!(prompt.contains("Hello!"));
}

#[test]
pub fn inventory_questions_do_not_call_server_management_tools() {
    let mut req = tool_request("what are the mcps u have", None);
    req.tools = Some(vec![
        server_tool(
            "manage_servers",
            "Manage configured MCP servers and available capabilities",
        ),
        server_tool("read", "Read files"),
        server_tool("bash", "Run shell commands"),
    ]);

    assert_eq!(
        classify_tool_role(&server_tool(
            "manage_servers",
            "Manage configured MCP servers and available capabilities",
        )),
        ToolRole::Action
    );
    assert!(has_tool_involvement(&req));
    // No inventory tool matches this MCP query, so prompt_tools is empty.
    // generate_tool_calls will early-return and fall through to regular chat.
    assert!(prompt_tools_for_request(&req).is_empty());
}

#[test]
pub fn tool_role_classifier_separates_inventory_from_diagnostics_and_actions() {
    assert_eq!(
        classify_tool_role(&server_tool(
            "agent_capabilities",
            "Introspection: return available tools, skills, subagents, and runtime capabilities",
        )),
        ToolRole::Inventory
    );
    assert_eq!(
        classify_tool_role(&server_tool(
            "diagnose_tool",
            "Diagnose tool selection and available capability failures",
        )),
        ToolRole::Diagnostic
    );
    assert_eq!(
        classify_tool_role(&server_tool(
            "delegate_task",
            "Delegate work to a specialized subagent",
        )),
        ToolRole::Action
    );
    assert_eq!(
        classify_tool_role(&server_tool(
            "list_mcp_resource_templates",
            "List MCP resource templates",
        )),
        ToolRole::McpResource
    );
}

#[test]
pub fn agent_reasoning_summary_is_safe_for_openz_thought_ui() {
    let mut req = tool_request("what can this agent do", None);
    req.tools = Some(vec![server_tool(
        "agent_capabilities",
        "Introspection: return available tools and skills",
    )]);

    let summary = agent_reasoning_summary(&req, "what can this agent do", "verified_tools")
        .expect("reasoning summary expected");

    assert!(summary.contains("capability_inventory"));
    assert!(summary.contains("agent-provided"));
    assert!(!summary.contains("<think>"));
    assert!(!summary.to_ascii_lowercase().contains("private"));
}

#[test]
pub fn opencode_injected_skill_context_does_not_force_tool_generation() {
    let mut req = tool_request("hi", None);
    req.messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: json!(
                "<available-skills>Use the use_skill and read_skill_file tools</available-skills>"
            ),
            tool_call_id: None,
            tool_calls: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: json!([{"type":"text","text":"<user-prompt-submit-hook>tool metadata</user-prompt-submit-hook>"},{"type":"text","text":"hi"}]),
            tool_call_id: None,
            tool_calls: None,
        },
    ];

    assert!(has_tool_involvement(&req));
}

#[test]
pub fn opencode_skill_evaluation_context_does_not_hide_latest_array_prompt() {
    let mut req = tool_request("so hey", None);
    req.messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: json!("<skill-evaluation-required>SKILL EVALUATION PROCESS use_skill tool may be relevant</skill-evaluation-required>"),
            tool_call_id: None,
            tool_calls: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: json!([
                {"type":"text","text":"<user-prompt-submit-hook>{}</user-prompt-submit-hook>"},
                {"type":"text","text":"so hey"}
            ]),
            tool_call_id: None,
            tool_calls: None,
        },
    ];

    assert_eq!(latest_user_prompt_text(&req), "so hey");
    assert!(has_tool_involvement(&req));
}

#[test]
pub fn extract_content_uses_latest_real_opencode_prompt() {
    let mut req = tool_request("hii", None);
    req.messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: json!("x".repeat(1000)),
            tool_call_id: None,
            tool_calls: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: json!("<available-skills>Use the use_skill tool</available-skills>"),
            tool_call_id: None,
            tool_calls: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: json!([
                {"type":"text","text":"<user-prompt-submit-hook>{}</user-prompt-submit-hook>"},
                {"type":"text","text":"hii"}
            ]),
            tool_call_id: None,
            tool_calls: None,
        },
    ];

    let (prompt, image_path) = extract_content(&req);
    assert_eq!(prompt, "hii");
    assert_eq!(image_path, None);
}

#[test]
pub fn extract_content_returns_image_path_from_multimodal_request() {
    let mut req = tool_request("Describe image", None);
    req.messages = vec![ChatMessage {
        role: "user".to_string(),
        content: json!([
            {"type":"text","text":"Describe image"},
            {"type":"image_url","image_url":{"url":"/tmp/screenshot.png"}}
        ]),
        tool_call_id: None,
        tool_calls: None,
    }];

    let (prompt, image_path) = extract_content(&req);
    assert_eq!(prompt, "Describe image");
    assert_eq!(image_path, Some("/tmp/screenshot.png".to_string()));
}

#[test]
pub fn extract_content_normalizes_file_image_urls() {
    let mut req = tool_request("Describe image", None);
    req.messages = vec![ChatMessage {
        role: "user".to_string(),
        content: json!([
            {"type":"text","text":"Describe image"},
            {"type":"image_url","image_url":{"url":"file:///tmp/screenshot.png"}}
        ]),
        tool_call_id: None,
        tool_calls: None,
    }];

    let (_, image_path) = extract_content(&req);
    assert_eq!(image_path, Some("/tmp/screenshot.png".to_string()));
}

#[test]
pub fn lowercase_chat_intent_uses_direct_reasoner_path() {
    assert!(is_direct_reasoner_intent("chat"));
    assert!(is_direct_reasoner_intent("reason"));
    assert!(is_direct_reasoner_intent("multi_step"));
    assert!(is_direct_reasoner_intent("VISION"));
    assert!(!is_direct_reasoner_intent("code"));
}

#[test]
pub fn mivi_identity_prompt_names_external_and_internal_models() {
    assert!(MIVI_CHAT_SYSTEM_PROMPT.contains("model name is mivi"));
    assert!(MIVI_CHAT_SYSTEM_PROMPT.contains("Never identify as"));
}

#[test]
pub fn explicit_tool_request_enters_tool_generation() {
    let req = tool_request("Use the get_weather tool for Paris", None);
    assert!(has_tool_involvement(&req));
}

#[test]
pub fn required_tool_choice_enters_tool_generation() {
    let req = tool_request("weather in Paris", Some(json!("required")));
    assert!(has_tool_involvement(&req));
}

#[test]
pub fn object_tool_choice_selects_specific_tool() {
    let tool_c = json!({
        "type": "function",
        "function": {
            "name": "bash"
        }
    });
    let mut req = tool_request("hello", Some(tool_c));
    req.tools = Some(vec![
        ToolDef {
            function: FunctionDef {
                name: "bash".to_string(),
                description: Some("Run command".to_string()),
                parameters: None,
            },
            r#type: "function".to_string(),
        },
        ToolDef {
            function: FunctionDef {
                name: "read_file".to_string(),
                description: Some("Read file".to_string()),
                parameters: None,
            },
            r#type: "function".to_string(),
        },
    ]);

    let selection = select_tools_for_request(&req);
    assert_eq!(selection.selected.len(), 1);
    assert_eq!(selection.selected[0].function.name, "bash");
}

#[test]
pub fn tool_argument_json_schema_validation() {
    let tool = ToolDef {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: "test_tool".to_string(),
            description: None,
            parameters: Some(json!({
                "type": "object",
                "required": ["cmd", "retries"],
                "properties": {
                    "cmd": {
                        "type": "string"
                    },
                    "retries": {
                        "type": "integer"
                    },
                    "env": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "slow"]
                    },
                    "label": {
                        "type": "string"
                    },
                    "url": {
                        "type": "string",
                        "format": "uri"
                    }
                }
            })),
        },
    };

    // Valid call
    let call_valid = ToolCallOut {
        id: "call_1".to_string(),
        r#type: "function".to_string(),
        function: FunctionCallOut {
            name: "test_tool".to_string(),
            arguments: r#"{"cmd": "npm run test", "retries": 3, "env": ["PATH"], "mode": "fast"}"#
                .to_string(),
        },
    };
    assert!(validate_tool_call_arguments(&call_valid, &tool).is_ok());

    // Missing required property
    let call_missing = ToolCallOut {
        id: "call_2".to_string(),
        r#type: "function".to_string(),
        function: FunctionCallOut {
            name: "test_tool".to_string(),
            arguments: r#"{"cmd": "npm run test"}"#.to_string(),
        },
    };
    let err_missing = validate_tool_call_arguments(&call_missing, &tool).unwrap_err();
    assert!(err_missing.contains("Missing required property 'retries'"));

    // Empty declared string fields are invalid even when the client omitted
    // them from the schema's required list.
    let call_empty_string = ToolCallOut {
        id: "call_empty".to_string(),
        r#type: "function".to_string(),
        function: FunctionCallOut {
            name: "test_tool".to_string(),
            arguments: r#"{"cmd": "npm run test", "retries": 3, "label": ""}"#.to_string(),
        },
    };
    let err_empty = validate_tool_call_arguments(&call_empty_string, &tool).unwrap_err();
    assert!(err_empty.contains("String property 'label' cannot be empty"));

    // Invalid type
    let call_invalid_type = ToolCallOut {
        id: "call_3".to_string(),
        r#type: "function".to_string(),
        function: FunctionCallOut {
            name: "test_tool".to_string(),
            arguments: r#"{"cmd": 12345, "retries": 3}"#.to_string(),
        },
    };
    let err_type = validate_tool_call_arguments(&call_invalid_type, &tool).unwrap_err();
    assert!(err_type.contains("does not match type"));

    // Invalid enum
    let call_invalid_enum = ToolCallOut {
        id: "call_4".to_string(),
        r#type: "function".to_string(),
        function: FunctionCallOut {
            name: "test_tool".to_string(),
            arguments: r#"{"cmd": "test", "retries": 3, "mode": "normal"}"#.to_string(),
        },
    };
    let err_enum = validate_tool_call_arguments(&call_invalid_enum, &tool).unwrap_err();
    assert!(err_enum.contains("not one of the allowed enums"));

    let call_invalid_url = ToolCallOut {
        id: "call_url".to_string(),
        r#type: "function".to_string(),
        function: FunctionCallOut {
            name: "test_tool".to_string(),
            arguments: r#"{"cmd":"test","retries":3,"url":"hono.dev"}"#.to_string(),
        },
    };
    let err_url = validate_tool_call_arguments(&call_invalid_url, &tool).unwrap_err();
    assert!(err_url.contains("format 'uri'"));

    let strict_tool = ToolDef {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: "strict_tool".to_string(),
            description: None,
            parameters: Some(json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {"value": {"type": "string"}}
            })),
        },
    };
    let call_unknown = ToolCallOut {
        id: "call_unknown".to_string(),
        r#type: "function".to_string(),
        function: FunctionCallOut {
            name: "strict_tool".to_string(),
            arguments: r#"{"value":"ok","extra":true}"#.to_string(),
        },
    };
    let err_unknown = validate_tool_call_arguments(&call_unknown, &strict_tool).unwrap_err();
    assert!(err_unknown.contains("Unknown property 'extra'"));
}

#[test]
pub fn test_rate_limiter_allows_under_limit_and_blocks_over_limit() {
    let _guard = crate::server::types::rate_limit_env_lock().lock().unwrap();
    let limiter = crate::server::types::RateLimiter::new();
    let client = "test_client_1".to_string();

    for _ in 0..60 {
        assert!(limiter.check_rate_limit(client.clone()).is_ok());
    }

    let res = limiter.check_rate_limit(client);
    assert!(res.is_err());
    assert!(res.unwrap_err().contains("Rate limit exceeded"));

    assert!(limiter
        .check_rate_limit("test_client_2".to_string())
        .is_ok());
}

#[test]
pub fn tool_prompt_uses_compact_schema_summary() {
    let mut req = tool_request("run npm test", None);
    req.tools = Some(vec![ToolDef {
        function: FunctionDef {
            name: "bash".to_string(),
            description: Some("Run a shell command".to_string()),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "command to run"},
                    "timeout": {"type": "number", "description": "timeout seconds"}
                },
                "required": ["cmd"]
            })),
        },
        r#type: "function".to_string(),
    }]);

    let prompt = build_chat_prompt(&req);

    assert!(prompt.contains("<tools>"));
    assert!(prompt.contains("\"bash\""));
    assert!(prompt.contains("\"cmd\""));
}

#[test]
pub fn terminal_prompt_with_matching_tool_enters_tool_generation() {
    let mut req = tool_request("Run npm test.", None);
    req.tools = Some(vec![server_tool(
        "bash",
        "Run a shell command in the project terminal",
    )]);

    assert!(has_tool_involvement(&req));
}

#[test]
pub fn repaired_tool_arguments_are_valid_json() {
    let raw = r#"<tool_call>{"name":"bash","arguments":"{'cmd':'npm test'}"}</tool_call>"#;
    let calls = parse_tool_calls_for_tools(raw, &[server_tool("bash", "Run shell commands")]);

    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "bash");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments)
        .expect("tool arguments must be valid JSON");
    assert_eq!(
        args.get("cmd").and_then(|value| value.as_str()),
        Some("npm test")
    );
}

#[test]
pub fn parses_openai_format_tool_calls() {
    let raw = r#"{"tool_calls":[{"id":"call_read_file","type":"function","function":{"name":"read_file","arguments":{"path":"src/main.rs"}}}]}"#;
    let calls = parse_tool_calls_for_tools(
        raw,
        &[server_tool("read_file", "Read a file from workspace")],
    );

    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "read_file");
    let args: serde_json::Value =
        serde_json::from_str(&calls[0].function.arguments).expect("arguments must be valid JSON");
    assert_eq!(
        args.get("path").and_then(|value| value.as_str()),
        Some("src/main.rs")
    );
}

#[test]
pub fn parses_custom_tool_format_tool_calls() {
    let raw =
        r#"{"tool":"inspect_browsers","arguments":{"action":"open","tool":"firefox_browser"}}"#;
    let calls =
        parse_tool_calls_for_tools(raw, &[server_tool("inspect_browsers", "Inspect browsers")]);

    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "inspect_browsers");
    let args: serde_json::Value =
        serde_json::from_str(&calls[0].function.arguments).expect("arguments must be valid JSON");
    assert_eq!(
        args.get("action").and_then(|value| value.as_str()),
        Some("open")
    );
    assert_eq!(
        args.get("tool").and_then(|value| value.as_str()),
        Some("firefox_browser")
    );
}

#[test]
pub fn rejects_tool_calls_not_present_in_selected_tools() {
    let raw = r#"<tool_call>{"name":"delete_everything","arguments":{}}</tool_call>"#;
    let calls = parse_tool_calls_for_tools(raw, &[server_tool("bash", "Run shell commands")]);

    assert!(calls.is_empty());
}

#[test]
pub fn tool_error_category_uses_config_priority() {
    let config = parse_capability_config(
        r#"{
            "aliases": {},
            "tool_taxonomy": {},
            "tool_error_markers": ["error", "timed out"],
            "tool_salient_markers": ["error", "timed out"],
            "tool_error_categories": {
                "network_error": ["network error", "connection"],
                "timeout": ["timed out", "timeout"]
            },
            "tool_error_category_priority": ["timeout", "network_error"]
        }"#,
    )
    .expect("valid capability config");

    assert_eq!(
        tool_error_category_with_config("network error: connection timed out", &config),
        Some("timeout".to_string())
    );
}

#[test]
pub fn tool_result_followup_without_tool_intent_does_not_force_tool_generation() {
    let mut req = tool_request("Run cargo test.", None);
    req.tools = Some(vec![server_tool("bash", "Run a shell command")]);
    req.messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: json!("Run cargo test."),
            tool_call_id: None,
            tool_calls: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: json!(""),
            tool_call_id: None,
            tool_calls: Some(vec![ToolCallIn {
                id: "call_bash".to_string(),
                r#type: "function".to_string(),
                function: FunctionCallIn {
                    name: "bash".to_string(),
                    arguments: json!({"cmd":"cargo test"}).to_string(),
                },
            }]),
        },
        ChatMessage {
            role: "tool".to_string(),
            content: json!("error[E0425]: cannot find value `x` in this scope"),
            tool_call_id: Some("call_bash".to_string()),
            tool_calls: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: json!("Summarize the failure in one sentence."),
            tool_call_id: None,
            tool_calls: None,
        },
    ];

    assert!(has_tool_involvement(&req));
}

#[test]
pub fn worker_stream_content_delta_reads_llama_completion_content() {
    let chunk = json!({"content":"hello"});

    assert_eq!(worker_stream_content_delta(&chunk), Some("hello"));
}

#[test]
pub fn worker_stream_content_delta_reads_openai_delta_content() {
    let chunk = json!({
        "choices": [{
            "index": 0,
            "delta": {"content": "hello"},
            "finish_reason": null
        }]
    });

    assert_eq!(worker_stream_content_delta(&chunk), Some("hello"));
}

#[test]
pub fn text_stream_chunks_start_with_assistant_role() {
    let chunks = text_stream_chunks(
        "chatcmpl-test".to_string(),
        123,
        None,
        "hello".to_string(),
        None,
    );

    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[0]["choices"][0]["delta"]["content"], "");
}

#[test]
pub fn streaming_tool_call_chunks_end_with_tool_calls_finish_reason() {
    let expected_args = json!({"cmd":"cargo test"}).to_string();
    let calls = vec![ToolCallOut {
        id: "call_bash".to_string(),
        r#type: "function".to_string(),
        function: FunctionCallOut {
            name: "bash".to_string(),
            arguments: expected_args.clone(),
        },
    }];

    let chunks = tool_call_stream_chunks(
        "chatcmpl-test".to_string(),
        123,
        Some("selected shell tool".to_string()),
        &calls,
        None,
    );

    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[0]["choices"][0]["delta"]["content"], "");
    assert_eq!(
        chunks[1]["choices"][0]["delta"]["reasoning_content"],
        "selected shell tool"
    );
    assert_eq!(
        chunks[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "bash"
    );
    assert_eq!(
        chunks[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
        ""
    );

    // Assemble the arguments from all middle chunks
    let mut assembled_args = String::new();
    for chunk in &chunks[3..chunks.len() - 1] {
        if let Some(tool_calls) = chunk["choices"][0]["delta"].get("tool_calls") {
            if let Some(args) = tool_calls[0]["function"].get("arguments") {
                assembled_args.push_str(args.as_str().unwrap());
            }
        }
    }
    assert_eq!(assembled_args, expected_args);

    assert_eq!(
        chunks[chunks.len() - 1]["choices"][0]["finish_reason"],
        "tool_calls"
    );
}

#[test]
pub fn test_health_endpoint_response() {
    let brain = EdgeBrain {
        llama_cli: PathBuf::new(),
        minicpm_cli: PathBuf::new(),
        llama_path: PathBuf::new(),
        qwen_path: PathBuf::new(),
        tool_path: PathBuf::new(),
        minicpm_path: PathBuf::new(),
        minicpm_proj: PathBuf::new(),
        ultra_low_ram: false,
        text_worker: Arc::new(crate::worker::WorkerManager::new(
            crate::worker::WorkerConfig {
                server_path: PathBuf::new(),
                model_path: PathBuf::new(),
                host: "127.0.0.1".to_string(),
                port: 18080,
                context_tokens: 1024,
                gpu_layers: "0".to_string(),
                idle_secs: 10,
                cache_reuse_tokens: 0,
                threads: 2,
            },
        )),
        native: crate::native_brain::NativeBrain::new(),
    };
    let state = Arc::new(AppState {
        brain: brain.clone(),
        orchestrator: AgentOrchestrator::new(brain),
        router: NeedleRouter::new(),
        semaphore: std::sync::Arc::new(tokio::sync::Semaphore::new(1)),
        rate_limiter: crate::server::types::RateLimiter::new(),
    });

    let resp = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { handle_health(State(state)).await });
    assert_eq!(resp.0["status"], "healthy");
}

#[test]
pub fn test_usage_details_serialization() {
    let usage = UsageInfo::new(10, 20);
    let val = serde_json::to_value(usage).unwrap();
    assert_eq!(val["prompt_tokens"], 10);
    assert_eq!(val["completion_tokens"], 20);
    assert_eq!(val["total_tokens"], 30);
    assert_eq!(val["prompt_tokens_details"]["cached_tokens"], 0);
    assert_eq!(val["completion_tokens_details"]["reasoning_tokens"], 0);
}

#[test]
fn clamp_tool_result_keeps_head_and_tail() {
    let short = "ok";
    assert_eq!(clamp_tool_result(short), "ok");

    let long: String = "h".repeat(3000);
    let clamped = clamp_tool_result(&long);
    assert!(clamped.len() < 2200);
    assert!(clamped.contains("truncated"));
    // Head preserved, truncation marker present, UTF-8 valid (no panic).
    assert!(clamped.starts_with("hhh"));
}

#[test]
fn bind_host_defaults_to_loopback_and_honors_env() {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    let _guard = LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap();

    std::env::remove_var("MIVI_HOST");
    assert_eq!(resolve_bind_host(), "127.0.0.1");

    std::env::set_var("MIVI_HOST", "0.0.0.0");
    assert_eq!(resolve_bind_host(), "0.0.0.0");

    // Whitespace-only values fall back to the safe default.
    std::env::set_var("MIVI_HOST", "   ");
    assert_eq!(resolve_bind_host(), "127.0.0.1");

    std::env::remove_var("MIVI_HOST");
}

#[test]
fn constant_time_eq_matches_only_equal_strings() {
    assert!(constant_time_eq("secret", "secret"));
    assert!(!constant_time_eq("secret", "secreT"));
    assert!(!constant_time_eq("secret", "secret "));
    assert!(!constant_time_eq("short", "shorter"));
    assert!(constant_time_eq("", ""));
    assert!(!constant_time_eq("", "x"));
}

#[test]
fn request_timeout_reads_env_with_safe_default() {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    let _guard = LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap();

    std::env::remove_var("MIVI_REQUEST_TIMEOUT_SECS");
    assert_eq!(request_timeout_secs(), 300);

    std::env::set_var("MIVI_REQUEST_TIMEOUT_SECS", "60");
    assert_eq!(request_timeout_secs(), 60);

    std::env::set_var("MIVI_REQUEST_TIMEOUT_SECS", "0");
    assert_eq!(request_timeout_secs(), 300, "zero/negative falls back");

    std::env::remove_var("MIVI_REQUEST_TIMEOUT_SECS");
}
