use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const API_URL: &str = "https://api.anthropic.com/v1/messages";

// ---- wire types ----

#[derive(Serialize)]
pub struct Request<'a> {
    pub model: &'a str,
    pub max_tokens: u32,
    pub system: &'a str,
    pub tools: &'a [Tool],
    pub messages: &'a [Message],
    // Automatic caching: the API puts the breakpoint on the last cacheable block,
    // so each turn reads the previous turn's prefix from cache.
    pub cache_control: CacheControl,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CacheControl {
    Ephemeral,
}

#[derive(Serialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Serialize)]
pub struct Message {
    pub role: Role,
    // Raw JSON so blocks we don't model (thinking, etc.) round-trip byte-for-byte.
    pub content: Vec<Value>,
}

#[derive(Deserialize)]
pub struct Response {
    pub content: Vec<Value>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

#[derive(Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    PauseTurn,
    Refusal,
    #[serde(other)]
    Unknown,
}

// Typed view of the blocks we actually care about.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

// ---- client ----

pub struct Client {
    http: reqwest::Client,
    api_key: String,
}

impl Client {
    pub fn from_env() -> anyhow::Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").context("ANTHROPIC_API_KEY not set")?;
        Ok(Self {
            http: reqwest::Client::new(),
            api_key,
        })
    }

    pub async fn send(&self, req: &Request<'_>) -> anyhow::Result<Response> {
        let resp = self
            .http
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(req)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            anyhow::bail!("API error {status}: {}", resp.text().await?);
        }
        Ok(resp.json().await?)
    }
}
