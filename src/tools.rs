use std::process::Stdio;
use std::time::Duration;

use anyhow::Context;
use serde_json::{Value, json};

use crate::api::Tool;

// Longest tool output sent back to the model, so one big file or command can't flood the context.
const MAX_OUTPUT_BYTES: usize = 30_000;
const DEFAULT_BASH_TIMEOUT_SECS: u64 = 120;
const MAX_BASH_TIMEOUT_SECS: u64 = 600;

pub fn definitions() -> Vec<Tool> {
    vec![
        Tool {
            name: "read_file".into(),
            description: "Read a UTF-8 text file and return its contents.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute, or relative to the working directory." }
                },
                "required": ["path"]
            }),
        },
        Tool {
            name: "list_dir".into(),
            description: "List a directory's entries, one per line, sorted. Directories end with `/`.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute, or relative to the working directory." }
                },
                "required": ["path"]
            }),
        },
        Tool {
            name: "write_file".into(),
            description: "Create or overwrite a file with the given contents, creating parent directories as needed. Prefer edit_file for changes to existing files.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute, or relative to the working directory." },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
        },
        Tool {
            name: "edit_file".into(),
            description: "Replace one exact occurrence of `old_string` in a file with `new_string`. `old_string` must match the file exactly (including whitespace) and appear exactly once; include surrounding lines to make it unique.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute, or relative to the working directory." },
                    "old_string": { "type": "string" },
                    "new_string": { "type": "string" }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        },
        Tool {
            name: "bash".into(),
            description: "Run a command with `bash -c` in the working directory and return its exit code, stdout, and stderr. No stdin; interactive commands will not work.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Kill the command after this many seconds. Default 120, max 600."
                    }
                },
                "required": ["command"]
            }),
        },
    ]
}

async fn call_tool(name: &str, input: &Value) -> anyhow::Result<String> {
    match name {
        "read_file" => {
            let path = input["path"].as_str().context("missing `path`")?;
            let contents = std::fs::read_to_string(path)?;
            Ok(contents)
        }
        "list_dir" => {
            let path = input["path"].as_str().context("missing `path`")?;
            let mut names = Vec::new();
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let mut name = entry.file_name().to_string_lossy().into_owned();
                if entry.file_type()?.is_dir() {
                    name.push('/');
                }
                names.push(name);
            }
            names.sort();
            Ok(names.join("\n"))
        }
        "write_file" => {
            let path = input["path"].as_str().context("missing `path`")?;
            let content = input["content"].as_str().context("missing `content`")?;
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, content)?;
            Ok(format!("Wrote {} bytes to {path}.", content.len()))
        }
        "edit_file" => {
            let path = input["path"].as_str().context("missing `path`")?;
            let old = input["old_string"]
                .as_str()
                .context("missing `old_string`")?;
            let new = input["new_string"]
                .as_str()
                .context("missing `new_string`")?;
            let contents = std::fs::read_to_string(path)?;
            let count = contents.matches(old).count();
            if count == 0 {
                anyhow::bail!("`old_string` not found in {path}");
            }
            if count > 1 {
                anyhow::bail!(
                    "`old_string` appears {count} times in {path}; include more surrounding lines to make it unique"
                );
            }
            std::fs::write(path, contents.replacen(old, new, 1))?;
            Ok(format!("Edited {path}."))
        }
        "bash" => {
            let command = input["command"].as_str().context("missing `command`")?;
            let timeout_secs = input["timeout_secs"]
                .as_u64()
                .unwrap_or(DEFAULT_BASH_TIMEOUT_SECS);
            let timeout = Duration::from_secs(timeout_secs.min(MAX_BASH_TIMEOUT_SECS));

            let child = tokio::process::Command::new("bash")
                .arg("-c")
                .arg(command)
                .stdin(Stdio::null())
                .kill_on_drop(true) // so a timeout actually kills it
                .output();
            let Ok(output) = tokio::time::timeout(timeout, child).await else {
                anyhow::bail!("timed out after {}s", timeout.as_secs());
            };
            let output = output?;

            let code = match output.status.code() {
                Some(code) => code.to_string(),
                None => "none (killed by a signal)".to_string(),
            };
            let text = format!(
                "exit code: {code}\n\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            if !output.status.success() {
                anyhow::bail!(text);
            }
            Ok(text)
        }
        _ => anyhow::bail!("unknown tool: {name}"),
    }
}

pub struct Output {
    pub content: String,
    pub is_error: bool,
}

/// Runs a tool. Failures become error output for the model rather than aborting the turn.
pub async fn run_tool(name: &str, input: &Value) -> Output {
    let (content, is_error) = match call_tool(name, input).await {
        Ok(content) => (content, false),
        Err(e) => (e.to_string(), true),
    };
    Output {
        content: truncate(content),
        is_error,
    }
}

fn truncate(mut s: String) -> String {
    if s.len() <= MAX_OUTPUT_BYTES {
        return s;
    }
    let total = s.len();
    let mut end = MAX_OUTPUT_BYTES;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    s.push_str(&format!(
        "\n\n[truncated: showing the first {end} of {total} bytes]"
    ));
    s
}
