//! The agent's state on disk, so it survives restarts: the conversation (every
//! message is appended as it joins the history, and the history is loaded back
//! on startup), and how far it has read each relay's inbox. Only `agent serve`
//! uses it; a terminal session starts fresh.

use anyhow::{Context, Result, bail};
use turso::{Builder, Database, Value};

use crate::api::Message;

const TABLES: &str = "
CREATE TABLE IF NOT EXISTS messages (
  seq     INTEGER PRIMARY KEY,
  message TEXT NOT NULL,  -- the API message as JSON: role and content blocks
  at      TEXT NOT NULL   -- RFC 3339, UTC
);
-- Per relay: every message up to `seq` has reached the agent.
CREATE TABLE IF NOT EXISTS cursors (
  relay TEXT PRIMARY KEY,
  seq   INTEGER NOT NULL
);
-- Messages above a relay's cursor that already reached the agent (pushed
-- ahead of a catch-up), so the catch-up doesn't deliver them twice.
CREATE TABLE IF NOT EXISTS seen (
  relay TEXT NOT NULL,
  seq   INTEGER NOT NULL,
  PRIMARY KEY (relay, seq)
);
";

#[derive(Clone)]
pub struct Store {
    db: Database,
}

impl Store {
    pub async fn open(path: &str) -> Result<Self> {
        let db = Builder::new_local(path)
            .build()
            .await
            .with_context(|| format!("opening database {path}"))?;
        db.connect()?
            .execute_batch(TABLES)
            .await
            .context("creating tables")?;
        Ok(Self { db })
    }

    fn conn(&self) -> Result<turso::Connection> {
        let conn = self.db.connect()?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(conn)
    }

    /// The whole conversation, oldest first.
    pub async fn load(&self) -> Result<Vec<Message>> {
        let mut rows = self
            .conn()?
            .query("SELECT seq, message FROM messages ORDER BY seq", ())
            .await?;
        let mut messages = Vec::new();
        while let Some(row) = rows.next().await? {
            let seq = row.get_value(0)?;
            let Value::Text(json) = row.get_value(1)? else {
                bail!("message {seq:?} isn't text");
            };
            let message =
                serde_json::from_str(&json).with_context(|| format!("parsing message {seq:?}"))?;
            messages.push(message);
        }
        Ok(messages)
    }

    pub async fn append(&self, message: &Message) -> Result<()> {
        let at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        self.conn()?
            .execute(
                "INSERT INTO messages (message, at) VALUES (?1, ?2)",
                vec![
                    Value::Text(serde_json::to_string(message)?),
                    Value::Text(at),
                ],
            )
            .await?;
        Ok(())
    }

    /// How far the agent has read `relay`'s inbox; None before its first catch-up.
    pub async fn cursor(&self, relay: &str) -> Result<Option<i64>> {
        let mut rows = self
            .conn()?
            .query(
                "SELECT seq FROM cursors WHERE relay = ?1",
                vec![Value::Text(relay.to_string())],
            )
            .await?;
        match rows.next().await? {
            Some(row) => match row.get_value(0)? {
                Value::Integer(n) => Ok(Some(n)),
                other => bail!("cursor: expected an integer, got {other:?}"),
            },
            None => Ok(None),
        }
    }

    /// Whether message `seq` from `relay` already reached the agent.
    pub async fn seen(&self, relay: &str, seq: i64) -> Result<bool> {
        if let Some(cursor) = self.cursor(relay).await?
            && seq <= cursor
        {
            return Ok(true);
        }
        let mut rows = self
            .conn()?
            .query(
                "SELECT 1 FROM seen WHERE relay = ?1 AND seq = ?2",
                vec![Value::Text(relay.to_string()), Value::Integer(seq)],
            )
            .await?;
        Ok(rows.next().await?.is_some())
    }

    pub async fn mark_seen(&self, relay: &str, seq: i64) -> Result<()> {
        self.conn()?
            .execute(
                "INSERT OR IGNORE INTO seen (relay, seq) VALUES (?1, ?2)",
                vec![Value::Text(relay.to_string()), Value::Integer(seq)],
            )
            .await?;
        Ok(())
    }

    /// Moves `relay`'s cursor to `seq`: everything up to it has reached the agent.
    pub async fn advance(&self, relay: &str, seq: i64) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO cursors (relay, seq) VALUES (?1, ?2) \
             ON CONFLICT (relay) DO UPDATE SET seq = excluded.seq",
            vec![Value::Text(relay.to_string()), Value::Integer(seq)],
        )
        .await?;
        conn.execute(
            "DELETE FROM seen WHERE relay = ?1 AND seq <= ?2",
            vec![Value::Text(relay.to_string()), Value::Integer(seq)],
        )
        .await?;
        Ok(())
    }

    pub async fn count(&self) -> Result<u64> {
        let mut rows = self
            .conn()?
            .query("SELECT count(*) FROM messages", ())
            .await?;
        let row = rows.next().await?.context("count returned no rows")?;
        match row.get_value(0)? {
            Value::Integer(n) => Ok(n as u64),
            other => bail!("count: expected an integer, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::api::Role;

    #[tokio::test]
    async fn round_trips_messages_in_order() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("agent-store-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("t.db");
        let _ = std::fs::remove_file(&path);
        let path = path.to_str().unwrap();
        {
            let store = Store::open(path).await?;
            for text in ["one", "two"] {
                store
                    .append(&Message {
                        role: Role::User,
                        content: vec![json!({ "type": "text", "text": text })],
                    })
                    .await?;
            }
        }
        // Reopened, as after a restart.
        let store = Store::open(path).await?;
        let messages = store.load().await?;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content[0]["text"], "two");
        assert_eq!(store.count().await?, 2);
        Ok(())
    }

    #[tokio::test]
    async fn tracks_what_reached_the_agent() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("agent-cursor-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("c.db");
        let _ = std::fs::remove_file(&path);
        let store = Store::open(path.to_str().unwrap()).await?;
        assert_eq!(store.cursor("r").await?, None);
        store.advance("r", 5).await?;
        assert!(store.seen("r", 5).await?);
        assert!(!store.seen("r", 7).await?);
        // Pushed ahead of the catch-up.
        store.mark_seen("r", 7).await?;
        assert!(store.seen("r", 7).await?);
        store.advance("r", 8).await?;
        assert_eq!(store.cursor("r").await?, Some(8));
        assert!(store.seen("r", 7).await?);
        assert!(!store.seen("other", 7).await?);
        Ok(())
    }
}
