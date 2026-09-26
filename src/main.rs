mod api;
mod cli;
mod tools;

use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::json;

use api::{Block, CacheControl, Client, Message, Request, Role, StopReason};

const MAX_TOKENS: u32 = 16000;
const SYSTEM_PROMPT: &str = include_str!("system_prompt.md");

// ANSI escapes for de-emphasizing status lines (tool calls, token counts).
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";
// Return to column 0 and erase the whole line.
const CLEAR_LINE: &str = "\r\x1b[2K";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = cli::parse_args()?;
    let client = Client::from_env()?;
    let tools = tools::definitions();
    // Only draw the live stopwatch on a real terminal, so logs don't fill with escape codes.
    let show_stopwatch = std::io::stderr().is_terminal();

    // Built once per session: the cached prefix must stay byte-identical across turns.
    let system = format!(
        "{SYSTEM_PROMPT}\n# Working directory\n\n{}\n",
        std::env::current_dir()?.display()
    );

    let mut messages = Vec::new();
    let stdin = std::io::stdin();

    // Outer loop: one iteration per user message.
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut text = String::new();
        if stdin.read_line(&mut text)? == 0 {
            break; // EOF (ctrl-d)
        }
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        println!();
        messages.push(Message {
            role: Role::User,
            content: vec![json!({ "type": "text", "text": text })],
        });

        // Inner loop: keep calling the model until it stops asking for tools.
        loop {
            // Scoped so `req` (which borrows `messages`) is dropped before we push to it.
            let (resp, latency) = {
                let req = Request {
                    model: &args.model,
                    max_tokens: MAX_TOKENS,
                    system: &system,
                    tools: &tools,
                    messages: &messages,
                    cache_control: CacheControl::Ephemeral,
                };

                // Redraw a stopwatch on the current line until the response arrives.
                let start = Instant::now();
                let request = client.send(&req);
                tokio::pin!(request);
                // 37ms, not 50: an irregular period makes the hundredths digit look like a running clock.
                let mut ticker = tokio::time::interval(Duration::from_millis(37));
                let resp = loop {
                    tokio::select! {
                        resp = &mut request => break resp,
                        _ = ticker.tick(), if show_stopwatch => {
                            eprint!("{CLEAR_LINE}{DIM}{:.2}s{RESET}", start.elapsed().as_secs_f64());
                        }
                    }
                };
                (resp, start.elapsed())
            };
            if show_stopwatch {
                eprint!("{CLEAR_LINE}");
            }
            let resp = resp?;

            // Blocks that don't match `Block` (e.g. thinking) are skipped here but still echoed below.
            let mut blocks = Vec::new();
            for b in &resp.content {
                if let Ok(block) = Block::deserialize(b) {
                    blocks.push(block);
                }
            }

            for b in &blocks {
                match b {
                    Block::Text { text } => println!("{text}\n"),
                    Block::ToolUse { name, input, .. } => {
                        eprintln!("{DIM}→ {name}({input}){RESET}")
                    }
                    _ => {}
                }
            }

            let u = &resp.usage;
            eprintln!(
                "{DIM}[tokens: {} in / {} cache write / {} cache read / {} out | {:.1}s]{RESET}\n",
                u.input_tokens,
                u.cache_creation_input_tokens.unwrap_or(0),
                u.cache_read_input_tokens.unwrap_or(0),
                u.output_tokens,
                latency.as_secs_f64()
            );

            messages.push(Message {
                role: Role::Assistant,
                content: resp.content,
            });

            match resp.stop_reason {
                StopReason::EndTurn => break,
                StopReason::ToolUse => {
                    let mut results = Vec::new();
                    for b in &blocks {
                        if let Block::ToolUse { id, name, input } = b {
                            let result = tools::run_tool(id, name, input);
                            results.push(serde_json::to_value(result)?);
                        }
                    }
                    // All results in one user message.
                    messages.push(Message {
                        role: Role::User,
                        content: results,
                    });
                }
                other => {
                    // Don't kill the session; hand control back to the user.
                    eprintln!("[stopped: {other:?}]");
                    break;
                }
            }
        }
    }

    Ok(())
}
