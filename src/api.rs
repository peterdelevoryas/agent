use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const API_URL: &str = "https://api.anthropic.com/v1/messages";
// Lets `thinking.display: "updates"` return the model's between-tool-call notes as text.
const BETA_HEADER: &str = "thinking-display-updates-2026-08-18";

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
    pub thinking: Thinking,
    pub stream: bool,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Thinking {
    Adaptive { display: ThinkingDisplay },
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingDisplay {
    /// Reasoning stays hidden, but the model's short progress notes come back as thinking text.
    Updates,
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

#[derive(Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    // Raw JSON so blocks we don't model (thinking, etc.) round-trip byte-for-byte.
    pub content: Vec<Value>,
}

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

// Typed view of the blocks the agent loop acts on. Text and thinking are shown as they
// stream, so they don't appear here.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
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

    /// Sends a streaming request. `on_update` sees text and thinking as they arrive; the
    /// returned `Response` is the complete message, built from the stream.
    pub async fn send(
        &self,
        req: &Request<'_>,
        mut on_update: impl FnMut(Update<'_>),
    ) -> anyhow::Result<Response> {
        let mut resp = self
            .http
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", BETA_HEADER)
            .json(req)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            anyhow::bail!("API error {status}: {}", resp.text().await?);
        }

        // Server-sent events: `event: ...` and `data: {json}` lines, with a blank line after
        // each event. The JSON carries its own `type`, so only the data lines matter.
        let mut stream = MessageStream::default();
        let mut buf = Vec::new();
        while let Some(chunk) = resp.chunk().await? {
            buf.extend_from_slice(&chunk);
            while let Some(end) = find_blank_line(&buf) {
                let event: Vec<u8> = buf.drain(..end + 2).collect();
                for line in std::str::from_utf8(&event)?.lines() {
                    if let Some(data) = line.strip_prefix("data:") {
                        let event: Value = serde_json::from_str(data.trim_start())?;
                        stream.apply(&event, &mut on_update)?;
                    }
                }
            }
        }
        stream.finish()
    }
}

/// What the stream reports while a response is being generated.
pub enum Update<'a> {
    TextDelta(&'a str),
    ThinkingDelta(&'a str),
    /// A content block is complete (the raw block, as it will be echoed back).
    BlockDone(&'a Value),
}

fn find_blank_line(buf: &[u8]) -> Option<usize> {
    for i in 1..buf.len() {
        if buf[i - 1] == b'\n' && buf[i] == b'\n' {
            return Some(i - 1);
        }
    }
    None
}

/// Builds the complete message out of stream events. Blocks are kept as raw JSON, so
/// thinking blocks (and their signatures) are echoed back exactly as generated.
#[derive(Default)]
struct MessageStream {
    content: Vec<Value>,
    // Tool inputs arrive as JSON fragments; they're parsed when their block ends.
    partial_json: Vec<String>,
    usage: Option<Usage>,
    stop_reason: Option<StopReason>,
}

impl MessageStream {
    fn apply(
        &mut self,
        event: &Value,
        on_update: &mut impl FnMut(Update<'_>),
    ) -> anyhow::Result<()> {
        match event["type"].as_str().unwrap_or("") {
            "message_start" => {
                self.usage = Some(Usage::deserialize(&event["message"]["usage"])?);
            }
            "content_block_start" => {
                let index = block_index(event)?;
                if index != self.content.len() {
                    anyhow::bail!("stream: block {index} started out of order");
                }
                self.content.push(event["content_block"].clone());
                self.partial_json.push(String::new());
            }
            "content_block_delta" => {
                let index = block_index(event)?;
                let block = self
                    .content
                    .get_mut(index)
                    .context("stream: delta for unknown block")?;
                let delta = &event["delta"];
                match delta["type"].as_str().unwrap_or("") {
                    "text_delta" => {
                        let text = delta["text"].as_str().unwrap_or("");
                        append(block, "text", text);
                        on_update(Update::TextDelta(text));
                    }
                    "thinking_delta" => {
                        let thinking = delta["thinking"].as_str().unwrap_or("");
                        append(block, "thinking", thinking);
                        on_update(Update::ThinkingDelta(thinking));
                    }
                    "signature_delta" => {
                        append(
                            block,
                            "signature",
                            delta["signature"].as_str().unwrap_or(""),
                        );
                    }
                    "input_json_delta" => {
                        self.partial_json[index]
                            .push_str(delta["partial_json"].as_str().unwrap_or(""));
                    }
                    // Other delta types (e.g. citations) aren't used by this agent yet.
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = block_index(event)?;
                let block = self
                    .content
                    .get_mut(index)
                    .context("stream: stop for unknown block")?;
                let json = &self.partial_json[index];
                if !json.is_empty() {
                    // A block cut off at max_tokens can hold broken JSON; that response is
                    // dropped anyway, so leave the input empty rather than fail the turn.
                    if let Ok(input) = serde_json::from_str(json) {
                        block["input"] = input;
                    }
                }
                on_update(Update::BlockDone(block));
            }
            "message_delta" => {
                self.stop_reason = Some(StopReason::deserialize(&event["delta"]["stop_reason"])?);
                // Output tokens are only known at the end.
                if let (Some(usage), Some(output)) =
                    (&mut self.usage, event["usage"]["output_tokens"].as_u64())
                {
                    usage.output_tokens = output;
                }
            }
            "error" => anyhow::bail!("API error during stream: {}", event["error"]),
            // message_stop, ping, and anything new.
            _ => {}
        }
        Ok(())
    }

    fn finish(self) -> anyhow::Result<Response> {
        Ok(Response {
            content: self.content,
            stop_reason: self
                .stop_reason
                .context("stream ended before the message finished")?,
            usage: self.usage.context("stream ended before message_start")?,
        })
    }
}

fn block_index(event: &Value) -> anyhow::Result<usize> {
    let index = event["index"]
        .as_u64()
        .context("stream event without an index")?;
    Ok(index as usize)
}

/// Appends to a string field of a block, creating it if needed.
fn append(block: &mut Value, field: &str, text: &str) {
    if let Some(Value::String(existing)) = block.get_mut(field) {
        existing.push_str(text);
    } else {
        block[field] = Value::String(text.to_string());
    }
}
