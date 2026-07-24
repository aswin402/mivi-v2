use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OkfMemory {
    pub id: String,
    pub title: String,
    pub kind: String,
    pub tags: Vec<String>,
    pub body: String,
}

pub fn load_memory_dir(path: &Path) -> Result<Vec<OkfMemory>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let mut files = fs::read_dir(path)
        .map_err(|err| format!("failed to read memory dir {}: {err}", path.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("md"))
        .collect::<Vec<_>>();

    files.sort();

    files
        .iter()
        .map(|file| {
            let text = fs::read_to_string(file)
                .map_err(|err| format!("failed to read memory file {}: {err}", file.display()))?;
            parse_memory_file(file, &text)
        })
        .collect()
}

pub fn write_memory(path: &Path, memory: &OkfMemory) -> Result<PathBuf, String> {
    fs::create_dir_all(path)
        .map_err(|err| format!("failed to create memory dir {}: {err}", path.display()))?;

    let file_path = path.join(format!("{}.md", safe_file_stem(&memory.id)));
    let tags = if memory.tags.is_empty() {
        "[]".to_string()
    } else {
        format!("[{}]", memory.tags.join(", "))
    };
    let text = format!(
        "---\nid: {}\ntitle: {}\ntype: {}\ntags: {}\n---\n{}\n",
        memory.id,
        memory.title,
        memory.kind,
        tags,
        memory.body.trim()
    );

    fs::write(&file_path, text)
        .map_err(|err| format!("failed to write memory file {}: {err}", file_path.display()))?;
    Ok(file_path)
}

fn parse_memory_file(path: &Path, text: &str) -> Result<OkfMemory, String> {
    let (frontmatter, body) = split_frontmatter(text)
        .ok_or_else(|| format!("{} missing OKF frontmatter", path.display()))?;
    let fields = parse_frontmatter_fields(frontmatter);

    let id = required_field(path, &fields, "id")?;
    let title = required_field(path, &fields, "title")?;
    let kind = required_field(path, &fields, "type")?;
    let tags = fields
        .get("tags")
        .map(|value| parse_tags(value))
        .unwrap_or_default();

    Ok(OkfMemory {
        id,
        title,
        kind,
        tags,
        body: body.trim().to_string(),
    })
}

fn split_frontmatter(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix("---\n")?;
    let end = rest.find("\n---")?;
    let frontmatter = &rest[..end];
    let body = rest[end + "\n---".len()..].trim_start_matches(['\r', '\n']);
    Some((frontmatter, body))
}

fn parse_frontmatter_fields(frontmatter: &str) -> HashMap<String, String> {
    frontmatter
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            Some((key.trim().to_string(), value.trim().to_string()))
        })
        .collect()
}

fn required_field(
    path: &Path,
    fields: &HashMap<String, String>,
    key: &str,
) -> Result<String, String> {
    fields
        .get(key)
        .map(|value| value.trim().trim_matches('"').to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{} missing required {key}", path.display()))
}

fn parse_tags(value: &str) -> Vec<String> {
    value
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(|tag| tag.trim().trim_matches('"').trim_matches('\'').to_string())
        .filter(|tag| !tag.is_empty())
        .collect()
}

fn safe_file_stem(id: &str) -> String {
    let stem = id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();

    if stem.is_empty() {
        "memory".to_string()
    } else {
        stem
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_memory_dir() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "mivi_okf_memory_test_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parses_markdown_with_required_frontmatter() {
        let dir = temp_memory_dir();
        fs::write(
            dir.join("project.md"),
            "---\nid: project-main\ntitle: Project Notes\ntype: project\ntags: [mivi, runtime]\n---\nMIVI exposes only the mivi model.",
        )
        .unwrap();

        let memories = load_memory_dir(&dir).unwrap();

        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].id, "project-main");
        assert_eq!(memories[0].title, "Project Notes");
        assert_eq!(memories[0].kind, "project");
        assert_eq!(memories[0].tags, vec!["mivi", "runtime"]);
        assert!(memories[0].body.contains("only the mivi model"));
    }

    #[test]
    fn rejects_memory_missing_required_type() {
        let dir = temp_memory_dir();
        fs::write(
            dir.join("bad.md"),
            "---\nid: bad\ntitle: Missing Type\n---\nBody",
        )
        .unwrap();

        let err = load_memory_dir(&dir).unwrap_err();

        assert!(err.contains("missing required type"));
    }

    #[test]
    fn writes_memory_as_okf_markdown() {
        let dir = temp_memory_dir();
        let memory = OkfMemory {
            id: "user-pref".to_string(),
            title: "User Preference".to_string(),
            kind: "preference".to_string(),
            tags: vec!["agent".to_string(), "tools".to_string()],
            body: "Prefer concise agent prompts.".to_string(),
        };

        let path = write_memory(&dir, &memory).unwrap();
        let written = fs::read_to_string(&path).unwrap();
        let loaded = load_memory_dir(&dir).unwrap();

        assert!(written.contains("type: preference"));
        assert_eq!(loaded, vec![memory]);
    }
}
