use std::os::unix::process::ExitStatusExt;
use std::process::{ExitStatus, Stdio};
use std::time::{Duration, Instant};

use anyhow::Context;
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Child;
use tokio::sync::oneshot;

use crate::api::Tool;

// Longest tool output sent back to the model, so one big file or command can't flood the context:
// the first and last bytes are kept.
const MAX_OUTPUT_HEAD_BYTES: usize = 5_000;
const MAX_OUTPUT_TAIL_BYTES: usize = 25_000;
// Per pipe, for bash. Both pipes together stay under the overall limit above, so bash output
// is only ever cut once, here, and the note shows the real byte count.
const BASH_PIPE_HEAD_BYTES: usize = 2_000;
const BASH_PIPE_TAIL_BYTES: usize = 12_000;
// How long a timed-out or interrupted command gets to exit after SIGTERM before SIGKILL.
const TERM_GRACE: Duration = Duration::from_secs(2);
const DEFAULT_BASH_TIMEOUT_SECS: u64 = 120;
const MAX_BASH_TIMEOUT_SECS: u64 = 600;

pub fn definitions() -> Vec<Tool> {
    vec![
        Tool {
            name: "read_file".into(),
            description: "Read a UTF-8 text file and return its contents.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute, or relative to the working directory." }
                },
                "required": ["path"]
            }),
        },
        Tool {
            name: "list_dir".into(),
            description: "List a directory's entries, one per line, sorted. Directories end with `/`.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute, or relative to the working directory." }
                },
                "required": ["path"]
            }),
        },
        Tool {
            name: "write_file".into(),
            description: "Create or overwrite a file with the given contents, creating parent directories as needed. Prefer edit_file for changes to existing files.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute, or relative to the working directory." },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
        },
        Tool {
            name: "edit_file".into(),
            description: "Replace one exact occurrence of `old_string` in a file with `new_string`. `old_string` must match the file exactly (including whitespace) and appear exactly once; include surrounding lines to make it unique.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute, or relative to the working directory." },
                    "old_string": { "type": "string" },
                    "new_string": { "type": "string" }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        },
        Tool {
            name: "bash".into(),
            description: "Run a command with `bash -c` in the working directory and return its exit status, stdout, and stderr. No stdin; interactive commands will not work. If the command times out or the user interrupts it, it is terminated and you get the output it produced so far. Background processes it starts are killed when it finishes. Long output keeps its start and end, with the middle cut.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Kill the command after this many seconds. Default 120, max 600."
                    }
                },
                "required": ["command"]
            }),
        },
    ]
}

async fn call_tool(
    name: &str,
    input: &Value,
    cancel: oneshot::Receiver<()>,
) -> anyhow::Result<String> {
    match name {
        "read_file" => {
            let path = input["path"].as_str().context("missing `path`")?;
            let contents = std::fs::read_to_string(path)?;
            Ok(contents)
        }
        "list_dir" => {
            let path = input["path"].as_str().context("missing `path`")?;
            let mut names = Vec::new();
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let mut name = entry.file_name().to_string_lossy().into_owned();
                if entry.file_type()?.is_dir() {
                    name.push('/');
                }
                names.push(name);
            }
            names.sort();
            Ok(names.join("\n"))
        }
        "write_file" => {
            let path = input["path"].as_str().context("missing `path`")?;
            let content = input["content"].as_str().context("missing `content`")?;
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, content)?;
            Ok(format!("Wrote {} bytes to {path}.", content.len()))
        }
        "edit_file" => {
            let path = input["path"].as_str().context("missing `path`")?;
            let old = input["old_string"]
                .as_str()
                .context("missing `old_string`")?;
            let new = input["new_string"]
                .as_str()
                .context("missing `new_string`")?;
            let contents = std::fs::read_to_string(path)?;
            let count = contents.matches(old).count();
            if count == 0 {
                anyhow::bail!("`old_string` not found in {path}");
            }
            if count > 1 {
                anyhow::bail!(
                    "`old_string` appears {count} times in {path}; include more surrounding lines to make it unique"
                );
            }
            std::fs::write(path, contents.replacen(old, new, 1))?;
            Ok(format!("Edited {path}."))
        }
        "bash" => {
            let command = input["command"].as_str().context("missing `command`")?;
            let timeout_secs = input["timeout_secs"]
                .as_u64()
                .unwrap_or(DEFAULT_BASH_TIMEOUT_SECS);
            let timeout = Duration::from_secs(timeout_secs.min(MAX_BASH_TIMEOUT_SECS));
            run_bash(command, timeout, cancel).await
        }
        _ => anyhow::bail!("unknown tool: {name}"),
    }
}

pub struct Output {
    pub content: String,
    pub is_error: bool,
}

/// Runs a tool. Failures become error output for the model rather than aborting the turn.
/// Sending on `cancel` asks a long-running tool (bash) to stop and report what it has so far.
pub async fn run_tool(name: &str, input: &Value, cancel: oneshot::Receiver<()>) -> Output {
    let (content, is_error) = match call_tool(name, input, cancel).await {
        Ok(content) => (content, false),
        Err(e) => (e.to_string(), true),
    };
    Output {
        content: truncate(&content, MAX_OUTPUT_HEAD_BYTES, MAX_OUTPUT_TAIL_BYTES),
        is_error,
    }
}

enum End {
    Exited(ExitStatus),
    TimedOut,
    Interrupted,
}

async fn run_bash(
    command: &str,
    timeout: Duration,
    mut cancel: oneshot::Receiver<()>,
) -> anyhow::Result<String> {
    let start = Instant::now();
    let mut child = tokio::process::Command::new("bash")
        .arg("-c")
        .arg(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Its own process group: the terminal's ctrl-c doesn't reach it (the agent decides
        // what an interrupt does), and killpg reaches everything it started.
        .process_group(0)
        .kill_on_drop(true) // safety net only
        .spawn()?;
    let pgid = child.id().context("bash exited before we got its pid")? as i32;

    // Read both pipes as the command runs, so partial output survives however it ends.
    let stdout = tokio::spawn(capture(child.stdout.take().context("no stdout")?));
    let stderr = tokio::spawn(capture(child.stderr.take().context("no stderr")?));

    let end = tokio::select! {
        status = child.wait() => End::Exited(status?),
        _ = tokio::time::sleep(timeout) => End::TimedOut,
        Ok(()) = &mut cancel => End::Interrupted,
    };
    let (status, why) = match end {
        End::Exited(status) => (status, None),
        End::TimedOut => (
            terminate(&mut child, pgid).await?,
            Some(format!("timed out after {}s", timeout.as_secs())),
        ),
        End::Interrupted => (
            terminate(&mut child, pgid).await?,
            Some(format!(
                "interrupted by the user after {:.1}s",
                start.elapsed().as_secs_f64()
            )),
        ),
    };
    // Kill anything it left running in the background, so nothing outlives the command
    // and the pipes close. (The group can already be empty; that's fine.)
    killpg(pgid, libc::SIGKILL);

    let stdout = stdout.await?;
    let stderr = stderr.await?;
    let mut text = String::new();
    if let Some(why) = &why {
        text.push_str(&format!("[{why}; partial output below]\n"));
    }
    text.push_str(&format!(
        "{}\n\nstdout:\n{}\nstderr:\n{}",
        describe(status),
        stdout.render(),
        stderr.render()
    ));
    if why.is_some() || !status.success() {
        anyhow::bail!(text);
    }
    Ok(text)
}

/// SIGTERM the whole group, give it a moment, then SIGKILL. Returns bash's exit status.
async fn terminate(child: &mut Child, pgid: i32) -> std::io::Result<ExitStatus> {
    killpg(pgid, libc::SIGTERM);
    if let Ok(status) = tokio::time::timeout(TERM_GRACE, child.wait()).await {
        return status;
    }
    killpg(pgid, libc::SIGKILL);
    child.wait().await
}

fn killpg(pgid: i32, signal: i32) {
    // Safety: killpg only sends a signal; the worst case is ESRCH when the group is gone.
    unsafe {
        libc::killpg(pgid, signal);
    }
}

fn describe(status: ExitStatus) -> String {
    if let Some(code) = status.code() {
        return format!("exit code: {code}");
    }
    match status.signal() {
        Some(libc::SIGINT) => "killed by SIGINT".to_string(),
        Some(libc::SIGTERM) => "killed by SIGTERM".to_string(),
        Some(libc::SIGKILL) => "killed by SIGKILL".to_string(),
        Some(signal) => format!("killed by signal {signal}"),
        None => "exit status unknown".to_string(),
    }
}

/// What a command wrote to one pipe: the first and last bytes, with the middle dropped
/// so a runaway command can't use unbounded memory.
struct Capture {
    head: Vec<u8>,
    tail: Vec<u8>,
    total: usize,
}

impl Capture {
    fn render(&self) -> String {
        let dropped = self.total - self.head.len() - self.tail.len();
        let mut text = String::from_utf8_lossy(&self.head).into_owned();
        if dropped > 0 {
            text.push_str(&format!("\n[… {dropped} bytes cut …]\n"));
        }
        text.push_str(&String::from_utf8_lossy(&self.tail));
        text
    }
}

async fn capture(mut pipe: impl AsyncRead + Unpin) -> Capture {
    let mut capture = Capture {
        head: Vec::new(),
        tail: Vec::new(),
        total: 0,
    };
    let mut chunk = [0u8; 8192];
    loop {
        let n = match pipe.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        capture.total += n;
        let mut bytes = &chunk[..n];
        let room = BASH_PIPE_HEAD_BYTES - capture.head.len();
        if room > 0 {
            let take = room.min(bytes.len());
            capture.head.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
        }
        capture.tail.extend_from_slice(bytes);
        // Let the tail grow to twice its limit before trimming, so we don't shift it on every read.
        if capture.tail.len() > 2 * BASH_PIPE_TAIL_BYTES {
            let excess = capture.tail.len() - BASH_PIPE_TAIL_BYTES;
            capture.tail.drain(..excess);
        }
    }
    if capture.tail.len() > BASH_PIPE_TAIL_BYTES {
        let excess = capture.tail.len() - BASH_PIPE_TAIL_BYTES;
        capture.tail.drain(..excess);
    }
    capture
}

///
/// Keeps the first `head` and last `tail` bytes of `s`, with a note about what was cut.
/// The start often has the first error; the end has where things stopped.
fn truncate(s: &str, head: usize, tail: usize) -> String {
    if s.len() <= head + tail {
        return s.to_string();
    }
    let mut head_end = head;
    while !s.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = s.len() - tail;
    while !s.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    format!(
        "{}\n\n[… {} of {} bytes cut …]\n\n{}",
        &s[..head_end],
        tail_start - head_end,
        s.len(),
        &s[tail_start..]
    )
}
