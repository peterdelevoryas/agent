mod agent;
mod api;
mod cli;
mod health;
mod mcp;
mod serve;
mod store;
mod tools;

use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;

use agent::{Agent, Event, Input};
use api::Client;
use mcp::Mcp;

// ANSI escapes for de-emphasizing status lines (tool calls, token counts).
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";
// Return to column 0 and erase the whole line.
const CLEAR_LINE: &str = "\r\x1b[2K";

// Terminal progress bar (OSC 9;4), shown by Ghostty in the tab bar.
// Terminals that don't support it ignore the sequence.
const PROGRESS_BUSY: &str = "\x1b]9;4;3\x07"; // indeterminate
const PROGRESS_CLEAR: &str = "\x1b]9;4;0\x07";

/// Shortens `s` to at most `max` characters for a one-line display.
fn one_line(s: &str, max: usize) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i == max {
            out.push('…');
            break;
        }
        out.push(if c == '\n' { ' ' } else { c });
    }
    out
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = cli::parse_args()?;
    if args.serve {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "agent=info".into()),
            )
            // Under systemd, stdout is the journal: color codes would show up raw.
            .with_ansi(std::io::stdout().is_terminal())
            .init();
        return serve::run(args.model).await;
    }
    // The terminal starts a fresh conversation each time; only serve mode saves one.
    let agent = Agent::new(
        Client::from_env()?,
        args.model,
        "",
        Mcp::from_env().await?,
        None,
    )
    .await?;

    let (input_tx, input_rx) = mpsc::unbounded_channel();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    tokio::spawn(agent.run(input_rx, event_tx));

    // Read stdin on a plain thread, not with tokio::io::stdin(): a pending tokio stdin read
    // blocks runtime shutdown, so ctrl-c couldn't exit until the user pressed enter.
    let (line_tx, mut line_rx) = mpsc::unbounded_channel();
    std::thread::spawn(move || {
        for line in std::io::stdin().lines() {
            let Ok(line) = line else { break };
            if line_tx.send(line).is_err() {
                break;
            }
        }
    });

    // Only show terminal status (stopwatch, progress bar) on a real terminal,
    // so logs don't fill with escape codes.
    let show_status = std::io::stderr().is_terminal();
    // 37ms, not 50: an irregular period makes the hundredths digit look like a running clock.
    let mut ticker = tokio::time::interval(Duration::from_millis(37));
    // The ticker keeps running while idle; without this, the first wait after an idle
    // stretch fires every missed tick at once.
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut waiting_since: Option<Instant> = None;
    // True while streamed text has left the cursor mid-line.
    let mut mid_line = false;

    // ctrl-c during a turn interrupts it; at the prompt, or a second time, it quits.
    let mut ctrl_c = signal(SignalKind::interrupt())?;

    // One turn at a time: read a line, then render the turn's events until it ends.
    'session: loop {
        print!("> ");
        std::io::stdout().flush()?;
        let line = tokio::select! {
            line = line_rx.recv() => match line {
                Some(line) => line,
                None => break 'session, // EOF (ctrl-d)
            },
            _ = ctrl_c.recv() => break 'session,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if input_tx.send(Input::Message(line.to_string())).is_err() {
            break 'session;
        }

        let mut interrupting = false;
        loop {
            tokio::select! {
                event = event_rx.recv() => {
                    let Some(event) = event else { break 'session };

                    // Any event ends the wait for a response.
                    if waiting_since.take().is_some() && show_status {
                        eprint!("{CLEAR_LINE}{PROGRESS_CLEAR}");
                    }

                    // Anything that isn't more streamed text starts on its own line.
                    let streaming = matches!(event, Event::TextDelta(_) | Event::ThinkingDelta(_));
                    if mid_line && !streaming {
                        eprintln!();
                        mid_line = false;
                    }

                    match event {
                        Event::RequestStarted => {
                            println!();
                            waiting_since = Some(Instant::now());
                            if show_status {
                                eprint!("{PROGRESS_BUSY}");
                            }
                        }
                        Event::TextDelta(text) => {
                            print!("{text}");
                            std::io::stdout().flush()?;
                            mid_line = !text.ends_with('\n');
                        }
                        // mid_line was already cleared above.
                        Event::TextEnd => println!(),
                        Event::ThinkingDelta(text) => {
                            eprint!("{DIM}{text}{RESET}");
                            mid_line = !text.ends_with('\n');
                        }
                        Event::ThinkingEnd => {}
                        Event::ToolCall { name, input } => {
                            eprintln!("{DIM}→ {name}({}){RESET}", one_line(&input.to_string(), 120));
                        }
                        Event::ToolResult { name, output } => {
                            if output.is_error {
                                eprintln!("{DIM}  ✗ {name}: {}{RESET}", one_line(&output.content, 120));
                            } else {
                                eprintln!("{DIM}  ✓ {name}: {} lines{RESET}", output.content.lines().count());
                            }
                        }
                        Event::Usage { usage: u, first_token, latency } => {
                            let first = match first_token {
                                Some(t) => format!("{:.1}s to first token, ", t.as_secs_f64()),
                                None => String::new(),
                            };
                            eprintln!(
                                "{DIM}[tokens: {} in / {} cache write / {} cache read / {} out | {first}{:.1}s total]{RESET}",
                                u.input_tokens,
                                u.cache_creation_input_tokens.unwrap_or(0),
                                u.cache_read_input_tokens.unwrap_or(0),
                                u.output_tokens,
                                latency.as_secs_f64()
                            );
                        }
                        Event::Stopped(reason) => eprintln!("[stopped: {reason:?}]"),
                        Event::Interrupted => eprintln!("{DIM}[interrupted]{RESET}"),
                        Event::Steered(n) => eprintln!("{DIM}[{n} new message(s) added to this turn]{RESET}"),
                        Event::Error(e) => eprintln!("error: {e}"),
                        Event::TurnEnded => {
                            println!();
                            break;
                        }
                    }
                }
                _ = ticker.tick(), if show_status && waiting_since.is_some() => {
                    if let Some(start) = waiting_since {
                        eprint!("{CLEAR_LINE}{DIM}{:.2}s{RESET}", start.elapsed().as_secs_f64());
                    }
                }
                _ = ctrl_c.recv() => {
                    if interrupting || input_tx.send(Input::Interrupt).is_err() {
                        break 'session;
                    }
                    interrupting = true;
                }
            }
        }
    }

    // Runs on every exit path out of the loop: ctrl-d, ctrl-c, or the agent stopping.
    // The newline keeps the shell prompt off the `> ` line.
    println!();
    if show_status {
        eprint!("{CLEAR_LINE}{PROGRESS_CLEAR}");
    }

    Ok(())
}
