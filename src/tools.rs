use anyhow::Context;
use serde_json::{Value, json};

use crate::api::{Block, Tool};

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

pub fn run_tool(id: &str, name: &str, input: &Value) -> Block {
    let (content, is_error) = match call_tool(name, input) {
        Ok(s) => (s, false),
        Err(e) => (e.to_string(), true),
    };
    Block::ToolResult {
        tool_use_id: id.to_string(),
        content,
        is_error,
    }
}
