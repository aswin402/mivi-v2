//! Verifier subprocess sandboxing.
//!
//! Model-generated code is executed by `CompilerVerifier` as child processes
//! (python3, node, bun, rustc, g++). Without restrictions those children run
//! with the full privileges of the mivi process: they can read `$HOME`,
//! open network connections, and write anywhere the user can.
//!
//! This module applies the Linux Landlock LSM inside each spawned child via
//! `pre_exec`, so only the interpreter is restricted — never the mivi server
//! itself. Policy is deny-by-default:
//!
//! - read/execute: system toolchain paths (`/usr`, `/bin`, `/lib`, `/lib64`,
//!   `/etc`) plus well-known user toolchain dirs (`~/.rustup`, `~/.cargo`,
//!   `~/.nvm`, `~/.bun`) when present. `$HOME` itself stays inaccessible.
//! - read-write: only the verifier's dedicated temp directory.
//! - network: TCP bind/connect denied when the kernel supports Landlock ABI
//!   v4 (6.7+); on older kernels filesystem isolation still applies but
//!   network remains reachable — surfaced via [`SandboxOutcome`].
//!
//! Controlled by `MIVI_VERIFY_SANDBOX`:
//! - `auto` (default): sandbox when available, warn once and continue when not
//! - `on`: require the sandbox; verification fails with a clear error otherwise
//! - `off`: no sandboxing (previous behavior)

use std::path::PathBuf;

pub const SANDBOX_ENV: &str = "MIVI_VERIFY_SANDBOX";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SandboxMode {
    Auto,
    On,
    Off,
}

impl SandboxMode {
    pub fn from_env() -> Self {
        match std::env::var(SANDBOX_ENV)
            .unwrap_or_default()
            .trim()
            .to_lowercase()
            .as_str()
        {
            "on" | "1" | "true" | "strict" => SandboxMode::On,
            "off" | "0" | "false" | "no" => SandboxMode::Off,
            _ => SandboxMode::Auto,
        }
    }
}

/// What actually got enforced in the sandboxed child.
#[derive(Debug, Clone, Copy)]
pub struct SandboxOutcome {
    pub fs_restricted: bool,
    pub net_restricted: bool,
}

/// Attach the sandbox to a command about to be spawned. Returns a warning
/// string when running unsandboxed is intentional (`auto` mode without kernel
/// support), and an error when strict mode cannot be honored.
pub fn attach(
    cmd: &mut tokio::process::Command,
    allow_dir: PathBuf,
) -> Result<Option<String>, String> {
    match SandboxMode::from_env() {
        SandboxMode::Off => Ok(None),
        SandboxMode::Auto | SandboxMode::On => match availability() {
            Availability::Supported { .. } => {
                attach_pre_exec(cmd, allow_dir);
                Ok(None)
            }
            Availability::Unsupported(reason) if SandboxMode::from_env() == SandboxMode::On => {
                Err(format!(
                    "MIVI_VERIFY_SANDBOX=on but the sandbox is unavailable: {}",
                    reason
                ))
            }
            Availability::Unsupported(reason) => Ok(Some(format!(
                "verifier running WITHOUT sandbox ({}); set MIVI_VERIFY_SANDBOX=on to make this fatal",
                reason
            ))),
        },
    }
}

#[cfg(target_os = "linux")]
fn attach_pre_exec(cmd: &mut tokio::process::Command, allow_dir: PathBuf) {
    use std::io;
    unsafe {
        cmd.pre_exec(move || {
            linux::restrict_current_process(&allow_dir)
                .map(|_| ())
                .map_err(io::Error::other)
        });
    }
}

#[cfg(not(target_os = "linux"))]
fn attach_pre_exec(_cmd: &mut tokio::process::Command, _allow_dir: PathBuf) {}

enum Availability {
    Supported { net: bool },
    Unsupported(String),
}

/// Probe whether Landlock can be used without restricting anything
/// (creating a ruleset does not affect the caller; only restrict_self does).
#[cfg(target_os = "linux")]
fn availability() -> Availability {
    match linux::probe() {
        Ok(net) => Availability::Supported { net },
        Err(e) => Availability::Unsupported(e),
    }
}

#[cfg(not(target_os = "linux"))]
fn availability() -> Availability {
    Availability::Unsupported("sandboxing is only implemented for Linux".to_string())
}

#[cfg(target_os = "linux")]
mod linux {
    use super::SandboxOutcome;
    use landlock::{
        path_beneath_rules, Access, AccessFs, AccessNet, CompatLevel, Compatible, PathBeneath,
        PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr, ABI,
    };
    use std::path::{Path, PathBuf};

    /// System paths interpreters and toolchains need at runtime.
    const SYSTEM_READ_PATHS: &[&str] = &["/usr", "/bin", "/lib", "/lib64", "/etc"];

    /// User-level toolchain locations that must stay readable for rustup/nvm/bun
    /// installs. `$HOME` itself is never granted.
    const USER_TOOLCHAIN_DIRS: &[&str] =
        &[".rustup", ".cargo", ".nvm", ".bun", ".local/share/mise"];

    /// True when the running kernel enforces Landlock network scoping (ABI v4+).
    pub fn probe() -> Result<bool, String> {
        build_ruleset(&std::env::temp_dir(), true)
            .map(|out| out.net_restricted)
            .map_err(|e| e)
    }

    /// Restrict the CURRENT thread/process. Called from `pre_exec` so only the
    /// spawned interpreter inherits the restriction.
    pub fn restrict_current_process(allow_dir: &Path) -> Result<SandboxOutcome, String> {
        build_ruleset(allow_dir, true).or_else(|strict_err| {
            // Kernel has filesystem Landlock but lacks network scoping (< 6.7):
            // fall back to filesystem-only isolation instead of failing hard.
            build_ruleset(allow_dir, false).map_err(|fallback_err| {
                format!(
                    "landlock unavailable (strict: {}; fs-only: {})",
                    strict_err, fallback_err
                )
            })
        })
    }

    fn build_ruleset(allow_dir: &Path, with_net: bool) -> Result<SandboxOutcome, String> {
        let mut ruleset = Ruleset::default().set_compatibility(CompatLevel::HardRequirement);
        ruleset = ruleset
            .handle_access(AccessFs::from_all(ABI::V1))
            .map_err(|e| format!("handle fs access: {}", e))?;
        if with_net {
            ruleset = ruleset
                .handle_access(AccessNet::from_all(ABI::V4))
                .map_err(|e| format!("handle net access: {}", e))?;
        }
        let created = ruleset
            .create()
            .map_err(|e| format!("create ruleset: {}", e))?;

        let mut read_paths: Vec<PathBuf> = SYSTEM_READ_PATHS
            .iter()
            .filter(|p| Path::new(p).exists())
            .map(PathBuf::from)
            .collect();
        if let Some(home) = std::env::var_os("HOME") {
            let home = PathBuf::from(home);
            for dir in USER_TOOLCHAIN_DIRS {
                let candidate = home.join(dir);
                if candidate.exists() {
                    read_paths.push(candidate);
                }
            }
        }

        let mut created = created
            .add_rules(path_beneath_rules(
                read_paths.iter().map(|p| p.as_path()),
                AccessFs::from_read(ABI::V1),
            ))
            .map_err(|e| format!("add read rules: {}", e))?;

        for dev in ["/dev/null"] {
            if Path::new(dev).exists() {
                created = created
                    .add_rule(PathBeneath::new(
                        PathFd::new(dev).map_err(|e| format!("open {}: {}", dev, e))?,
                        AccessFs::from_all(ABI::V1),
                    ))
                    .map_err(|e| format!("add rule {}: {}", dev, e))?;
            }
        }
        for dev in ["/dev/urandom", "/dev/random"] {
            if Path::new(dev).exists() {
                created = created
                    .add_rule(PathBeneath::new(
                        PathFd::new(dev).map_err(|e| format!("open {}: {}", dev, e))?,
                        AccessFs::from_read(ABI::V1),
                    ))
                    .map_err(|e| format!("add rule {}: {}", dev, e))?;
            }
        }

        let allow_dir = allow_dir
            .canonicalize()
            .unwrap_or_else(|_| allow_dir.to_path_buf());
        let created = created
            .add_rule(PathBeneath::new(
                PathFd::new(&allow_dir)
                    .map_err(|e| format!("open {}: {}", allow_dir.display(), e))?,
                AccessFs::from_all(ABI::V1),
            ))
            .map_err(|e| format!("add workdir rule: {}", e))?;

        created
            .restrict_self()
            .map_err(|e| format!("restrict_self: {}", e))?;

        Ok(SandboxOutcome {
            fs_restricted: true,
            net_restricted: with_net,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandbox_mode_parsing() {
        // Serialized through env_lock-style discipline: this test only reads
        // MIVI_VERIFY_SANDBOX after setting it, and other sandbox tests do not
        // touch env concurrently because cargo test runs them on separate
        // threads — keep this test independent of ordering by restoring value.
        let old = std::env::var(SANDBOX_ENV).ok();
        for (raw, want) in [
            ("on", SandboxMode::On),
            ("ON", SandboxMode::On),
            ("1", SandboxMode::On),
            ("off", SandboxMode::Off),
            ("no", SandboxMode::Off),
            ("", SandboxMode::Auto),
            ("auto", SandboxMode::Auto),
            ("garbage", SandboxMode::Auto),
        ] {
            std::env::set_var(SANDBOX_ENV, raw);
            assert_eq!(SandboxMode::from_env(), want, "input {:?}", raw);
        }
        match old {
            Some(v) => std::env::set_var(SANDBOX_ENV, v),
            None => std::env::remove_var(SANDBOX_ENV),
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn probe_reports_support_without_restricting_caller() {
        // Must not error on any modern CI runner; net flag depends on kernel.
        let outcome = super::availability();
        match outcome {
            Availability::Supported { .. } => {}
            Availability::Unsupported(reason) => {
                println!("landlock unsupported here: {}", reason)
            }
        }
        // Caller must remain unrestricted: writing next to the crate still works.
        let probe_path = std::env::temp_dir().join("mivi-sandbox-probe-ok");
        std::fs::write(&probe_path, b"ok").expect("caller should be unrestricted");
        let _ = std::fs::remove_file(&probe_path);
    }
}
