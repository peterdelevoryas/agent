//! MCP servers the agent can use, over streamable HTTP: their tools join the
//! built-in ones, and their instructions join the system prompt. Configured by
//! AGENT_MCP_SERVERS (`name=url` pairs, comma-separated), with each server's
//! bearer token in AGENT_MCP_TOKEN_<NAME> (e.g. AGENT_MCP_TOKEN_MEMORY).

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use rmcp::{
    RoleClient, ServiceExt,
    model::CallToolRequestParams,
    service::RunningService,
    transport::{
        StreamableHttpClientTransport, streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde_json::Value;

use crate::api::Tool;
use crate::tools;

struct Server {
    name: String,
    client: RunningService<RoleClient, ()>,
}

#[derive(Default)]
pub struct Mcp {
    servers: Vec<Server>,
    /// Tool name → index into `servers`.
    owners: HashMap<String, usize>,
    tools: Vec<Tool>,
}

impl Mcp {
    /// Connects to every server in AGENT_MCP_SERVERS; none if it's unset.
    pub async fn from_env() -> Result<Self> {
        let mut mcp = Self::default();
        let Ok(spec) = std::env::var("AGENT_MCP_SERVERS") else {
            return Ok(mcp);
        };
        for entry in spec.split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let Some((name, url)) = entry.split_once('=') else {
                bail!("AGENT_MCP_SERVERS: {entry:?} should be name=url");
            };
            mcp.connect(name.trim(), url.trim())
                .await
                .with_context(|| format!("connecting to MCP server {name}"))?;
        }
        Ok(mcp)
    }

    async fn connect(&mut self, name: &str, url: &str) -> Result<()> {
        let token_var = format!("AGENT_MCP_TOKEN_{}", name.to_uppercase());
        let token = std::env::var(&token_var).with_context(|| format!("{token_var} not set"))?;
        let config = StreamableHttpClientTransportConfig::with_uri(url).auth_header(token);
        let transport = StreamableHttpClientTransport::from_config(config);
        let client = ().serve(transport).await?;
        let index = self.servers.len();
        for tool in client.list_all_tools().await? {
            let tool_name = tool.name.to_string();
            if self.owners.insert(tool_name.clone(), index).is_some() {
                bail!("two MCP servers both have a tool named {tool_name}");
            }
            self.tools.push(Tool {
                name: tool_name,
                description: tool.description.map(|d| d.to_string()).unwrap_or_default(),
                input_schema: Value::Object((*tool.input_schema).clone()),
            });
        }
        self.servers.push(Server {
            name: name.to_string(),
            client,
        });
        Ok(())
    }

    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    pub fn has(&self, tool: &str) -> bool {
        self.owners.contains_key(tool)
    }

    /// Each server's instructions, as a system prompt section; empty if none.
    pub fn instructions(&self) -> String {
        let mut out = String::new();
        for server in &self.servers {
            let Some(info) = server.client.peer_info() else {
                continue;
            };
            let Some(instructions) = &info.instructions else {
                continue;
            };
            out.push_str(&format!(
                "\n## {}\n\n{}\n",
                server.name,
                instructions.trim()
            ));
        }
        if out.is_empty() {
            return out;
        }
        format!("\n# Connected services\n{out}")
    }

    /// Calls an MCP tool. Failures become error output, like built-in tools.
    pub async fn call(&self, tool: &str, input: &Value) -> tools::Output {
        match self.try_call(tool, input).await {
            Ok((content, is_error)) => tools::output(&content, is_error),
            Err(e) => tools::output(&format!("{e:#}"), true),
        }
    }

    async fn try_call(&self, tool: &str, input: &Value) -> Result<(String, bool)> {
        let index = *self.owners.get(tool).context("no such MCP tool")?;
        let server = &self.servers[index];
        let mut params = CallToolRequestParams::new(tool.to_string());
        if let Value::Object(arguments) = input {
            params = params.with_arguments(arguments.clone());
        }
        let result = server
            .client
            .call_tool(params)
            .await
            .with_context(|| format!("calling {tool} on {}", server.name))?;
        let mut text = String::new();
        for block in &result.content {
            if let Some(t) = block.as_text() {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&t.text);
            }
        }
        if text.is_empty()
            && let Some(structured) = &result.structured_content
        {
            text = structured.to_string();
        }
        Ok((text, result.is_error.unwrap_or(false)))
    }
}
