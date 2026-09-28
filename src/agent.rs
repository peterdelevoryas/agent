use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};

use crate::api::{Block, CacheControl, Client, Message, Request, Role, StopReason, Tool, Usage};
use crate::tools;

const MAX_TOKENS: u32 = 16000;
const SYSTEM_PROMPT: &str = include_str!("system_prompt.md");
const INTERRUPTED: &str = "Interrupted by the user before this finished.";
const INTERRUPTED_BEFORE_RESPONSE: &str = "[The user interrupted this turn before you responded. Don't act on the messages above unless asked again.]";

/// What front ends send to the agent.
pub enum Input {
    Message(String),
    /// Stop the current turn. Ignored between turns.
    Interrupt,
}

/// What the agent reports back. Front ends decide how (or whether) to show each one.
pub enum Event {
    RequestStarted,
    Text(String),
    ToolCall { name: String, input: Value },
    ToolResult { name: String, output: tools::Output },
    Usage { usage: Usage, latency: Duration },
    Stopped(StopReason),
    Interrupted,
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
        while let Some(input) = inputs.recv().await {
            // An interrupt that arrives after its turn already ended lands here; drop it.
            let Input::Message(text) = input else {
                continue;
            };
            self.messages.push(Message {
                role: Role::User,
                content: vec![json!({ "type": "text", "text": text })],
            });
            if let Err(e) = self.run_turn(&mut inputs, &events).await {
                let _ = events.send(Event::Error(format!("{e:#}")));
            }
            let _ = events.send(Event::TurnEnded);
        }
    }

    /// Keeps calling the model until it stops asking for tools.
    /// An interrupt drops whatever is in flight.
    async fn run_turn(
        &mut self,
        inputs: &mut mpsc::UnboundedReceiver<Input>,
        events: &mpsc::UnboundedSender<Event>,
    ) -> anyhow::Result<()> {
        loop {
            let _ = events.send(Event::RequestStarted);
            let start = Instant::now();
            let req = Request {
                model: &self.model,
                max_tokens: MAX_TOKENS,
                system: &self.system,
                tools: &self.tools,
                messages: &self.messages,
                cache_control: CacheControl::Ephemeral,
            };
            let resp = tokio::select! {
                resp = self.client.send(&req) => resp?,
                // Nothing has been appended for this request yet, so dropping it leaves
                // history valid.
                _ = wait_for_interrupt(inputs) => {
                    // Record it, so the model doesn't treat the unanswered message as still
                    // open. Appended as a user message; the API merges consecutive ones.
                    self.messages.push(Message {
                        role: Role::User,
                        content: vec![json!({ "type": "text", "text": INTERRUPTED_BEFORE_RESPONSE })],
                    });
                    let _ = events.send(Event::Interrupted);
                    return Ok(());
                }
            };
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
            let mut interrupted = false;
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
                        // After an interrupt, the remaining calls still need a result each.
                        let output = if interrupted {
                            interrupted_output()
                        } else {
                            let (cancel, cancel_rx) = oneshot::channel();
                            let run = tools::run_tool(&name, &input, cancel_rx);
                            tokio::pin!(run);
                            tokio::select! {
                                // If both are ready, keep the tool's real result.
                                biased;
                                output = &mut run => output,
                                _ = wait_for_interrupt(inputs) => {
                                    interrupted = true;
                                    // Ask the tool to stop, then wait for it to report how
                                    // far it got (bash returns its partial output).
                                    let _ = cancel.send(());
                                    run.await
                                }
                            }
                        };
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

            if interrupted {
                let _ = events.send(Event::Interrupted);
                return Ok(());
            }
            if done {
                return Ok(());
            }
        }
    }
}

/// Resolves when the front end asks to interrupt the current turn.
async fn wait_for_interrupt(inputs: &mut mpsc::UnboundedReceiver<Input>) {
    loop {
        match inputs.recv().await {
            Some(Input::Interrupt) => return,
            // The terminal front end only sends messages between turns.
            Some(Input::Message(_)) => {}
            // Front end gone: never interrupt, let the turn finish.
            None => std::future::pending::<()>().await,
        }
    }
}

fn interrupted_output() -> tools::Output {
    tools::Output {
        content: INTERRUPTED.to_string(),
        is_error: true,
    }
}
