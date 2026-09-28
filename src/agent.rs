use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::api::{Block, CacheControl, Client, Message, Request, Role, StopReason, Tool, Usage};
use crate::tools;

const MAX_TOKENS: u32 = 16000;
const SYSTEM_PROMPT: &str = include_str!("system_prompt.md");

/// What front ends send to the agent.
pub enum Input {
    Message(String),
}

/// What the agent reports back. Front ends decide how (or whether) to show each one.
pub enum Event {
    RequestStarted,
    Text(String),
    ToolCall { name: String, input: Value },
    ToolResult { name: String, output: tools::Output },
    Usage { usage: Usage, latency: Duration },
    Stopped(StopReason),
    Error(String),
    TurnEnded,
}

pub struct Agent {
    client: Client,
    model: String,
    system: String,
    tools: Vec<Tool>,
    messages: Vec<Message>,
}

impl Agent {
    pub fn new(client: Client, model: String) -> anyhow::Result<Self> {
        // Built once per session: the cached prefix must stay byte-identical across turns.
        let system = format!(
            "{SYSTEM_PROMPT}\n# Working directory\n\n{}\n",
            std::env::current_dir()?.display()
        );
        Ok(Self {
            client,
            model,
            system,
            tools: tools::definitions(),
            messages: Vec::new(),
        })
    }

    /// Runs until every input sender is dropped.
    pub async fn run(
        mut self,
        mut inputs: mpsc::UnboundedReceiver<Input>,
        events: mpsc::UnboundedSender<Event>,
    ) {
        while let Some(Input::Message(text)) = inputs.recv().await {
            self.messages.push(Message {
                role: Role::User,
                content: vec![json!({ "type": "text", "text": text })],
            });
            if let Err(e) = self.run_turn(&events).await {
                let _ = events.send(Event::Error(format!("{e:#}")));
            }
            let _ = events.send(Event::TurnEnded);
        }
    }

    /// Keeps calling the model until it stops asking for tools.
    async fn run_turn(&mut self, events: &mpsc::UnboundedSender<Event>) -> anyhow::Result<()> {
        loop {
            let _ = events.send(Event::RequestStarted);
            let start = Instant::now();
            let resp = self
                .client
                .send(&Request {
                    model: &self.model,
                    max_tokens: MAX_TOKENS,
                    system: &self.system,
                    tools: &self.tools,
                    messages: &self.messages,
                    cache_control: CacheControl::Ephemeral,
                })
                .await?;
            let latency = start.elapsed();
            let _ = events.send(Event::Usage {
                usage: resp.usage,
                latency,
            });

            // Only complete responses enter history. Anything else (cut off at max_tokens,
            // refused, unknown) is dropped whole: nothing is appended, so there are no
            // half-finished tool calls to answer and history stays valid.
            let done = match resp.stop_reason {
                StopReason::EndTurn => true,
                StopReason::ToolUse => false,
                other => {
                    let _ = events.send(Event::Stopped(other));
                    return Ok(());
                }
            };

            // One pass over the blocks: report each one, and run each tool call as we reach it.
            // With streaming, this is the code that moves into the `content_block_stop` handler.
            let mut results = Vec::new();
            for raw in &resp.content {
                // Blocks that don't match `Block` (e.g. thinking) are skipped here but still
                // echoed back in the assistant message below.
                let Ok(block) = Block::deserialize(raw) else {
                    continue;
                };
                match block {
                    Block::Text { text } => {
                        let _ = events.send(Event::Text(text));
                    }
                    Block::ToolUse { id, name, input } => {
                        let _ = events.send(Event::ToolCall {
                            name: name.clone(),
                            input: input.clone(),
                        });
                        let output = tools::run_tool(&name, &input).await;
                        results.push(serde_json::to_value(Block::ToolResult {
                            tool_use_id: id,
                            content: output.content.clone(),
                            is_error: output.is_error,
                        })?);
                        let _ = events.send(Event::ToolResult { name, output });
                    }
                    Block::ToolResult { .. } => {}
                }
            }
            self.messages.push(Message {
                role: Role::Assistant,
                content: resp.content,
            });
            if !results.is_empty() {
                // All results in one user message.
                self.messages.push(Message {
                    role: Role::User,
                    content: results,
                });
            }

            if done {
                return Ok(());
            }
        }
    }
}
