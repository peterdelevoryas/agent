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

            // Blocks that don't match `Block` (e.g. thinking) are skipped here but still echoed below.
            let mut blocks = Vec::new();
            for b in &resp.content {
                if let Ok(block) = Block::deserialize(b) {
                    blocks.push(block);
                }
            }

            for b in &blocks {
                match b {
                    Block::Text { text } => {
                        let _ = events.send(Event::Text(text.clone()));
                    }
                    Block::ToolUse { name, input, .. } => {
                        let _ = events.send(Event::ToolCall {
                            name: name.clone(),
                            input: input.clone(),
                        });
                    }
                    _ => {}
                }
            }
            let _ = events.send(Event::Usage {
                usage: resp.usage,
                latency,
            });

            self.messages.push(Message {
                role: Role::Assistant,
                content: resp.content,
            });

            match resp.stop_reason {
                StopReason::EndTurn => return Ok(()),
                StopReason::ToolUse => {
                    let mut content = Vec::new();
                    for b in &blocks {
                        if let Block::ToolUse { id, name, input } = b {
                            let result = tools::run_tool(id, name, input);
                            content.push(serde_json::to_value(result)?);
                        }
                    }
                    // All results in one user message.
                    self.messages.push(Message {
                        role: Role::User,
                        content,
                    });
                }
                other => {
                    // Don't kill the session; hand control back to the user.
                    let _ = events.send(Event::Stopped(other));
                    return Ok(());
                }
            }
        }
    }
}
