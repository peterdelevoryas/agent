use std::collections::VecDeque;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};

use crate::api::{
    Block, CacheControl, Client, Message, Request, Role, StopReason, Thinking, ThinkingDisplay,
    Tool, Update, Usage,
};
use crate::mcp::Mcp;
use crate::store::Store;
use crate::tools;

const MAX_TOKENS: u32 = 64000;
const SYSTEM_PROMPT: &str = include_str!("system_prompt.md");
const INTERRUPTED: &str = "Interrupted by the user before this finished.";
const SKIPPED_FOR_NEW_MESSAGE: &str =
    "Not run: a new message arrived first. Read it and decide whether this is still needed.";
const RESTARTED: &str = "The agent restarted before this finished; it may or may not have run.";
const INTERRUPTED_BEFORE_RESPONSE: &str = "[The user interrupted your response before it finished, and it was discarded. Don't act on the messages above unless asked again.]";

/// What front ends send to the agent.
pub enum Input {
    Message(String),
    /// Stop the current turn. Ignored between turns.
    Interrupt,
}

/// What the agent reports back. Front ends decide how (or whether) to show each one.
pub enum Event {
    RequestStarted,
    /// Streamed text of the model's reply.
    TextDelta(String),
    TextEnd,
    /// Streamed progress notes (reasoning itself stays hidden).
    ThinkingDelta(String),
    ThinkingEnd,
    ToolCall {
        name: String,
        input: Value,
    },
    ToolResult {
        name: String,
        output: tools::Output,
    },
    Usage {
        usage: Usage,
        first_token: Option<Duration>,
        latency: Duration,
    },
    Stopped(StopReason),
    Interrupted,
    /// New messages arrived mid-turn and were added to it.
    Steered(usize),
    Error(String),
    TurnEnded,
}

pub struct Agent {
    client: Client,
    model: String,
    system: String,
    tools: Vec<Tool>,
    mcp: Mcp,
    messages: Vec<Message>,
    /// Where the conversation is saved, if anywhere.
    store: Option<Store>,
}

impl Agent {
    /// `extra_system` is appended to the system prompt (e.g. how serve mode
    /// differs). With a store, the saved conversation is loaded and continued.
    pub async fn new(
        client: Client,
        model: String,
        extra_system: &str,
        mcp: Mcp,
        store: Option<Store>,
    ) -> anyhow::Result<Self> {
        // Built once per session: the cached prefix must stay byte-identical across turns.
        let system = format!(
            "{SYSTEM_PROMPT}{extra_system}{}\n# Working directory\n\n{}\n",
            mcp.instructions(),
            std::env::current_dir()?.display()
        );
        let mut tools = tools::definitions();
        for tool in mcp.tools() {
            if tools.iter().any(|t| t.name == tool.name) {
                anyhow::bail!(
                    "MCP tool {} has the same name as a built-in tool",
                    tool.name
                );
            }
            tools.push(Tool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema: tool.input_schema.clone(),
            });
        }
        let mut agent = Self {
            client,
            model,
            system,
            tools,
            mcp,
            messages: Vec::new(),
            store: None,
        };
        if let Some(store) = store {
            agent.messages = store.load().await?;
            agent.store = Some(store);
            agent.repair().await?;
        }
        Ok(agent)
    }

    pub fn history_len(&self) -> usize {
        self.messages.len()
    }

    /// Adds a message to the history, saving it first if there's a store.
    async fn push(&mut self, message: Message) -> anyhow::Result<()> {
        if let Some(store) = &self.store {
            store.append(&message).await?;
        }
        self.messages.push(message);
        Ok(())
    }

    /// If the agent stopped between asking for tools and recording their
    /// results, the saved history ends with unanswered tool calls, which the
    /// API rejects. Answer each one with an error.
    async fn repair(&mut self) -> anyhow::Result<()> {
        let Some(last) = self.messages.last() else {
            return Ok(());
        };
        if last.role != Role::Assistant {
            return Ok(());
        }
        let mut results = Vec::new();
        for raw in &last.content {
            if let Ok(Block::ToolUse { id, .. }) = Block::deserialize(raw) {
                results.push(serde_json::to_value(Block::ToolResult {
                    tool_use_id: id,
                    content: RESTARTED.to_string(),
                    is_error: true,
                })?);
            }
        }
        if !results.is_empty() {
            self.push(Message {
                role: Role::User,
                content: results,
            })
            .await?;
        }
        Ok(())
    }

    /// Runs until every input sender is dropped.
    pub async fn run(
        mut self,
        mut inputs: mpsc::UnboundedReceiver<Input>,
        events: mpsc::UnboundedSender<Event>,
    ) {
        // Messages that arrived during a turn, for the next one.
        let mut pending = VecDeque::new();
        loop {
            if pending.is_empty() {
                match inputs.recv().await {
                    Some(Input::Message(text)) => pending.push_back(text),
                    // An interrupt that arrives after its turn already ended lands here; drop it.
                    Some(Input::Interrupt) => continue,
                    None => break,
                }
            }
            // Everything waiting goes into this turn together.
            while let Ok(input) = inputs.try_recv() {
                if let Input::Message(text) = input {
                    pending.push_back(text);
                }
            }
            let mut content = Vec::new();
            while let Some(text) = pending.pop_front() {
                content.push(json!({ "type": "text", "text": text }));
            }
            let started = self
                .push(Message {
                    role: Role::User,
                    content,
                })
                .await;
            let result = match started {
                Ok(()) => self.run_turn(&mut inputs, &mut pending, &events).await,
                Err(e) => Err(e),
            };
            if let Err(e) = result {
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
        pending: &mut VecDeque<String>,
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
                thinking: Thinking::Adaptive {
                    display: ThinkingDisplay::Updates,
                },
                stream: true,
            };
            let mut first_token = None;
            let on_update = |update: Update<'_>| {
                if first_token.is_none() {
                    first_token = Some(start.elapsed());
                }
                let event = match update {
                    Update::TextDelta(text) => Event::TextDelta(text.to_string()),
                    Update::ThinkingDelta(text) => Event::ThinkingDelta(text.to_string()),
                    Update::BlockDone(block) => match block["type"].as_str() {
                        Some("text") => Event::TextEnd,
                        // Empty thinking blocks (hidden reasoning) show nothing.
                        Some("thinking") if block["thinking"].as_str().unwrap_or("") != "" => {
                            Event::ThinkingEnd
                        }
                        _ => return,
                    },
                };
                let _ = events.send(event);
            };
            let resp = tokio::select! {
                resp = self.client.send(&req, on_update) => resp?,
                // Nothing has been appended for this request yet, so dropping it leaves
                // history valid.
                arrival = next_arrival(inputs, pending) => match arrival {
                    Arrival::Interrupt => {
                        // Record it, so the model doesn't treat the unanswered message as
                        // still open. Appended as a user message; the API merges
                        // consecutive ones.
                        self.push(Message {
                            role: Role::User,
                            content: vec![json!({ "type": "text", "text": INTERRUPTED_BEFORE_RESPONSE })],
                        })
                        .await?;
                        let _ = events.send(Event::Interrupted);
                        return Ok(());
                    }
                    // Start over with the new message, so it gets one answer together
                    // with what came before.
                    Arrival::Message => {
                        let content = drain_as_text(pending);
                        let _ = events.send(Event::Steered(content.len()));
                        self.push(Message {
                            role: Role::User,
                            content,
                        })
                        .await?;
                        continue;
                    }
                },
            };
            let latency = start.elapsed();
            let _ = events.send(Event::Usage {
                usage: resp.usage,
                first_token,
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
            // A new message arrived: finish the running tool, skip the rest.
            let mut steered = false;
            for raw in &resp.content {
                // Blocks that don't match `Block` (text and thinking) are skipped here but still
                // echoed back in the assistant message below.
                let Ok(block) = Block::deserialize(raw) else {
                    continue;
                };
                match block {
                    Block::ToolUse { id, name, input } => {
                        let _ = events.send(Event::ToolCall {
                            name: name.clone(),
                            input: input.clone(),
                        });
                        // After an interrupt, the remaining calls still need a result each.
                        let output = if interrupted {
                            interrupted_output()
                        } else if steered {
                            tools::output(SKIPPED_FOR_NEW_MESSAGE, true)
                        } else {
                            let (cancel, cancel_rx) = oneshot::channel();
                            let mcp = &self.mcp;
                            let (tool, tool_input) = (&name, &input);
                            // Dropping an MCP call cancels it, so only bash needs `cancel`.
                            let run = async move {
                                if mcp.has(tool) {
                                    mcp.call(tool, tool_input).await
                                } else {
                                    tools::run_tool(tool, tool_input, cancel_rx).await
                                }
                            };
                            tokio::pin!(run);
                            tokio::select! {
                                // If both are ready, keep the tool's real result.
                                biased;
                                output = &mut run => output,
                                arrival = next_arrival(inputs, pending) => match arrival {
                                    Arrival::Interrupt => {
                                        interrupted = true;
                                        // Ask the tool to stop, then wait for it to report
                                        // how far it got (bash returns its partial output).
                                        let _ = cancel.send(());
                                        run.await
                                    }
                                    // A command can be stopped and rerun if still
                                    // needed, so stop bash; let anything else (a send,
                                    // a write) finish rather than leave it half done.
                                    Arrival::Message => {
                                        steered = true;
                                        if name == "bash" {
                                            let _ = cancel.send(());
                                        }
                                        run.await
                                    }
                                },
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
            // Messages that arrived during this response join the turn, after the tool
            // results in the same user message. After an interrupt they wait for the
            // next turn instead.
            let steer = !interrupted && !pending.is_empty();
            if steer {
                let texts = drain_as_text(pending);
                let _ = events.send(Event::Steered(texts.len()));
                results.extend(texts);
            }
            self.push(Message {
                role: Role::Assistant,
                content: resp.content,
            })
            .await?;
            if !results.is_empty() {
                // All results in one user message.
                self.push(Message {
                    role: Role::User,
                    content: results,
                })
                .await?;
            }

            if interrupted {
                let _ = events.send(Event::Interrupted);
                return Ok(());
            }
            if done && !steer {
                return Ok(());
            }
        }
    }
}

enum Arrival {
    /// The front end asked to stop the turn.
    Interrupt,
    /// A new message, now in `pending`.
    Message,
}

/// Resolves when an input arrives during a turn.
async fn next_arrival(
    inputs: &mut mpsc::UnboundedReceiver<Input>,
    pending: &mut VecDeque<String>,
) -> Arrival {
    match inputs.recv().await {
        Some(Input::Interrupt) => Arrival::Interrupt,
        Some(Input::Message(text)) => {
            pending.push_back(text);
            Arrival::Message
        }
        // Front end gone: nothing more will arrive; let the turn finish.
        None => std::future::pending().await,
    }
}

/// Takes every pending message as text blocks.
fn drain_as_text(pending: &mut VecDeque<String>) -> Vec<Value> {
    let mut blocks = Vec::new();
    while let Some(text) = pending.pop_front() {
        blocks.push(json!({ "type": "text", "text": text }));
    }
    blocks
}

fn interrupted_output() -> tools::Output {
    tools::Output {
        content: INTERRUPTED.to_string(),
        is_error: true,
    }
}
