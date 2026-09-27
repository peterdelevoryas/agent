use anyhow::Context;
use serde_json::{Value, json};

use crate::api::Tool;

pub fn definitions() -> Vec<Tool> {
    vec![Tool {
        name: "read_file".into(),
        description: "Read a UTF-8 text file and return its contents.".into(),
        input_schema: json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        }),
    }]
}

fn call_tool(name: &str, input: &Value) -> anyhow::Result<String> {
    match name {
        "read_file" => {
            let path = input["path"].as_str().context("missing `path`")?;
            let contents = std::fs::read_to_string(path)?;
            Ok(contents)
        }
        _ => anyhow::bail!("unknown tool: {name}"),
    }
}

pub struct Output {
    pub content: String,
    pub is_error: bool,
}

/// Runs a tool. Failures become error output for the model rather than aborting the turn.
pub fn run_tool(name: &str, input: &Value) -> Output {
    match call_tool(name, input) {
        Ok(content) => Output {
            content,
            is_error: false,
        },
        Err(e) => Output {
            content: e.to_string(),
            is_error: true,
        },
    }
}
