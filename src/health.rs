//! `GET /health`: an unauthenticated summary for an external monitor. It
//! reports whether the conversation database answers, how old the last successful backup
//! is (backup.sh writes a timestamp file), and how full the disk is: 200 when
//! all is well, 503 with the problems otherwise. It reveals no conversation content.

use std::{path::PathBuf, sync::Arc};

use axum::{Json, extract::State, http::StatusCode};
use serde::Serialize;

use crate::store::Store;

#[derive(Clone)]
pub struct Health(Arc<Config>);

pub struct Config {
    pub store: Store,
    /// Directory whose filesystem is checked for free space (the database's).
    pub data_dir: PathBuf,
    pub max_disk_percent: f64,
    /// File holding the RFC 3339 time of the last successful backup; no
    /// backup check when unset.
    pub backup_stamp: Option<PathBuf>,
    pub max_backup_age_hours: f64,
}

impl Health {
    pub fn new(config: Config) -> Self {
        Self(Arc::new(config))
    }
}

#[derive(Serialize, Debug, PartialEq)]
pub struct Report {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backup_age_hours: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_used_percent: Option<f64>,
    pub problems: Vec<String>,
}

pub async fn handler(State(Health(c)): State<Health>) -> (StatusCode, Json<Report>) {
    let mut problems = Vec::new();
    let messages = match c.store.count().await {
        Ok(n) => Some(n),
        Err(e) => {
            problems.push(format!("database: {e:#}"));
            None
        }
    };
    let disk = match disk_used_percent(&c.data_dir) {
        Ok(used) => Some(used),
        Err(e) => {
            problems.push(format!("disk: {e}"));
            None
        }
    };
    if let Some(used) = disk
        && used > c.max_disk_percent
    {
        problems.push(format!(
            "disk is {used:.0}% full (limit {:.0}%)",
            c.max_disk_percent
        ));
    }
    let mut backup_age = None;
    if let Some(path) = &c.backup_stamp {
        match backup_age_hours(path) {
            Ok(age) => {
                if age > c.max_backup_age_hours {
                    problems.push(format!(
                        "last backup was {age:.1} hours ago (limit {:.0})",
                        c.max_backup_age_hours
                    ));
                }
                backup_age = Some((age * 10.0).round() / 10.0);
            }
            Err(e) => problems.push(format!("backup: {e}")),
        }
    }
    let report = Report {
        ok: problems.is_empty(),
        messages,
        backup_age_hours: backup_age,
        disk_used_percent: disk.map(|d| d.round()),
        problems,
    };
    let status = if report.ok {
        StatusCode::OK
    } else {
        tracing::warn!(problems = ?report.problems, "health check failing");
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(report))
}

fn backup_age_hours(path: &std::path::Path) -> Result<f64, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("can't read {}: {e}", path.display()))?;
    let at = chrono::DateTime::parse_from_rfc3339(text.trim())
        .map_err(|e| format!("bad timestamp in {}: {e}", path.display()))?;
    Ok((chrono::Utc::now() - at.with_timezone(&chrono::Utc)).num_minutes() as f64 / 60.0)
}

fn disk_used_percent(dir: &std::path::Path) -> std::io::Result<f64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(dir.as_os_str().as_bytes())?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `path` is a valid NUL-terminated string and `stat` is a valid out-pointer.
    if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let total = stat.f_blocks as f64;
    let available = stat.f_bavail as f64;
    if total == 0.0 {
        return Ok(0.0);
    }
    Ok((total - available) / total * 100.0)
}
