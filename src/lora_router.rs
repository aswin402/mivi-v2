use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpecialistPersona {
    Reasoner,
    Tools,
    Coder,
    Debugger,
    Chat,
}

impl SpecialistPersona {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Reasoner => "reasoner",
            Self::Tools => "tools",
            Self::Coder => "coder",
            Self::Debugger => "debugger",
            Self::Chat => "chat",
        }
    }

    pub fn system_prompt_directive(&self) -> &'static str {
        match self {
            Self::Reasoner => {
                "Specialist Role: MIVI Reasoner (Deep Logic & Planning). Plan and reason deeply step-by-step before formulating clean architectures and solutions."
            }
            Self::Tools => {
                "Specialist Role: MIVI Tools (Agent & Research). Formulate precise JSON tool calls matching schemas to interact with the environment."
            }
            Self::Coder => {
                "Specialist Role: MIVI Coder (Full-Stack Polyglot). Write clean, production-grade, idiomatic code in Rust, Python, TypeScript, React, Tailwind, and Bash."
            }
            Self::Debugger => {
                "Specialist Role: MIVI Debugger (Diagnosis & Fix). Analyze compiler tracebacks and errors carefully to provide minimal, surgical replacement diffs."
            }
            Self::Chat => {
                "Specialist Role: MIVI Chat (Conversational Intelligence). Understand user needs deeply, speak fluent natural English, and provide direct, clear, and engaging responses."
            }
        }
    }

    pub fn adapter_path(&self) -> Option<PathBuf> {
        let base_dir =
            std::env::var("MIVI_LORA_DIR").unwrap_or_else(|_| "models/loras".to_string());
        let path = PathBuf::from(base_dir).join(format!("mivi-{}.bin", self.as_str()));
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }
}

pub fn resolve_specialist_persona(
    requested_model: Option<&str>,
    has_tools: bool,
    intent: &str,
    has_error_context: bool,
) -> SpecialistPersona {
    if let Some(model) = requested_model {
        let lower = model.to_ascii_lowercase();
        if lower.contains("chat") || lower.ends_with(":chat") {
            return SpecialistPersona::Chat;
        }
        if lower.contains("reason") || lower.ends_with(":reasoner") {
            return SpecialistPersona::Reasoner;
        }
        if lower.contains("tool") || lower.ends_with(":tools") {
            return SpecialistPersona::Tools;
        }
        if lower.contains("debug") || lower.ends_with(":debugger") {
            return SpecialistPersona::Debugger;
        }
        if lower.contains("code") || lower.ends_with(":coder") {
            return SpecialistPersona::Coder;
        }
    }

    if has_error_context {
        return SpecialistPersona::Debugger;
    }

    if has_tools {
        return SpecialistPersona::Tools;
    }

    match intent.to_ascii_lowercase().as_str() {
        "code" => SpecialistPersona::Coder,
        "multi_step" | "reason" => SpecialistPersona::Reasoner,
        "chat" => SpecialistPersona::Chat,
        _ => SpecialistPersona::Chat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_specialist_from_model_name() {
        assert_eq!(
            resolve_specialist_persona(Some("mivi:coder"), false, "CHAT", false),
            SpecialistPersona::Coder
        );
        assert_eq!(
            resolve_specialist_persona(Some("mivi:reasoner"), false, "CHAT", false),
            SpecialistPersona::Reasoner
        );
        assert_eq!(
            resolve_specialist_persona(Some("mivi:tools"), false, "CHAT", false),
            SpecialistPersona::Tools
        );
        assert_eq!(
            resolve_specialist_persona(Some("mivi:debugger"), false, "CHAT", false),
            SpecialistPersona::Debugger
        );
        assert_eq!(
            resolve_specialist_persona(Some("mivi:chat"), false, "CHAT", false),
            SpecialistPersona::Chat
        );
    }

    #[test]
    fn test_resolve_specialist_from_context() {
        assert_eq!(
            resolve_specialist_persona(None, true, "CHAT", false),
            SpecialistPersona::Tools
        );
        assert_eq!(
            resolve_specialist_persona(None, false, "CODE", false),
            SpecialistPersona::Coder
        );
        assert_eq!(
            resolve_specialist_persona(None, false, "CHAT", true),
            SpecialistPersona::Debugger
        );
        assert_eq!(
            resolve_specialist_persona(None, false, "MULTI_STEP", false),
            SpecialistPersona::Reasoner
        );
        assert_eq!(
            resolve_specialist_persona(None, false, "CHAT", false),
            SpecialistPersona::Chat
        );
    }
}
