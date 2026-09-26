use anyhow::Context;

// Keep in sync with the default shown in USAGE.
const MODEL: &str = "claude-opus-5-5";

const USAGE: &str = "\
Usage: agent [OPTIONS]

Options:
  -m, --model <MODEL>  Model to use [env: AGENT_MODEL] [default: claude-opus-5-5]
  -h, --help           Print this help
";

pub struct Args {
    pub model: String,
}

pub fn parse_args() -> anyhow::Result<Args> {
    // Precedence: --model flag, then AGENT_MODEL, then the built-in default.
    let mut model = std::env::var("AGENT_MODEL").unwrap_or_else(|_| MODEL.to_string());
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-m" | "--model" => model = args.next().context("--model needs a value")?,
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument: {other}\n\n{USAGE}"),
        }
    }
    Ok(Args { model })
}
