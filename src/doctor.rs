//! `mivi doctor`: hardware-aware environment diagnosis and runtime preset
//! recommendation (inspired by Colibrì's autotune/doctor and kimi-k3-in-c's
//! memory presets).
//!
//! Doctor is strictly read-only advisory: it inspects the machine, the current
//! effective `RuntimeConfig`, on-disk binaries/models, and prints a
//! recommended configuration. It never mutates process state.

use crate::model_catalog::ModelCatalog;
use crate::runtime::RuntimeConfig;

/// Everything the recommendation logic needs from the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemSnapshot {
    pub total_ram_mb: usize,
    pub available_ram_mb: usize,
    pub logical_cpus: usize,
    pub llama_server_present: bool,
    pub llama_cli_present: bool,
    pub gguf_count: usize,
}

/// The recommended MIVI configuration for a machine. Field names mirror the
/// `MIVI_*` env vars so the report can be exported directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecommendedPlan {
    pub runtime_mode: &'static str,
    pub ultra_low_ram: bool,
    pub context_budget: usize,
    pub ram_target_mb: usize,
    pub model_cache_max: usize,
    pub worker_idle_secs: u64,
    pub notes: Vec<String>,
}

/// Read total/available RAM in MB from `/proc/meminfo`. Returns `None` when
/// the file is missing (non-Linux) or unparsable.
pub fn read_meminfo_mb() -> Option<(usize, usize)> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb = 0usize;
    let mut avail_kb = 0usize;
    for line in content.lines() {
        if let Some(num) = line.strip_prefix("MemTotal:") {
            total_kb = num.split_whitespace().next()?.parse().ok()?;
        } else if let Some(num) = line.strip_prefix("MemAvailable:") {
            avail_kb = num.split_whitespace().next()?.parse().ok()?;
        }
    }
    if total_kb == 0 {
        return None;
    }
    Some((total_kb / 1024, avail_kb / 1024))
}

fn bin_present(name: &str) -> bool {
    std::path::Path::new("bin").join(name).is_file()
}

fn count_ggufs() -> usize {
    std::fs::read_dir("models")
        .map(|entries| {
            entries
                .filter_map(std::result::Result::ok)
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "gguf"))
                .count()
        })
        .unwrap_or(0)
}

pub fn read_system_snapshot() -> SystemSnapshot {
    let (total_ram_mb, available_ram_mb) = read_meminfo_mb().unwrap_or((0, 0));
    let logical_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    SystemSnapshot {
        total_ram_mb,
        available_ram_mb,
        logical_cpus,
        llama_server_present: bin_present("llama-server"),
        llama_cli_present: bin_present("llama-cli"),
        gguf_count: count_ggufs(),
    }
}

/// Pure tiering logic (unit-tested): pick a runtime shape from available RAM.
///
/// - `< 3000 MB` free: `spawn` + ultra-low-RAM streaming; keep only one model
///   resident and shrink the raw context so the KV cache fits the budget.
/// - `< 6000 MB` free: `worker-eco`; warm worker with default idle spin-down.
/// - otherwise: `worker-hot`; keep the worker warm longer to cut agent TTFT.
pub fn recommend(snapshot: &SystemSnapshot) -> RecommendedPlan {
    let mut notes = Vec::new();
    let avail = snapshot.available_ram_mb;

    let (runtime_mode, ultra_low_ram, context_budget, model_cache_max, worker_idle_secs) =
        if snapshot.total_ram_mb == 0 {
            notes.push("Could not read /proc/meminfo; assuming a small machine.".to_string());
            ("spawn", true, 2048, 1, 0)
        } else if avail < 3000 {
            notes.push(format!(
                "Only {avail} MB RAM available: spawn mode streams weights via mmap \
                 and keeps a single model resident."
            ));
            ("spawn", true, 2048, 1, 0)
        } else if avail < 6000 {
            notes.push(format!(
                "{avail} MB RAM available: worker-eco keeps a warm llama-server \
                 within the {DEFAULT} MB RAM target and spins it down when idle.",
                DEFAULT = crate::constants::DEFAULT_RAM_TARGET_MB
            ));
            (
                "worker-eco",
                false,
                crate::constants::DEFAULT_CONTEXT_TOKENS,
                2,
                120,
            )
        } else {
            notes.push(format!(
                "{avail} MB RAM available: worker-hot keeps inference warm for \
                 repeated agent requests."
            ));
            (
                "worker-hot",
                false,
                crate::constants::DEFAULT_CONTEXT_TOKENS,
                2,
                600,
            )
        };

    if snapshot.gguf_count == 0 {
        notes.push(
            "No GGUF files found in models/. Run download_models.py before serving.".to_string(),
        );
    }

    RecommendedPlan {
        runtime_mode,
        ultra_low_ram,
        context_budget,
        ram_target_mb: crate::constants::DEFAULT_RAM_TARGET_MB,
        model_cache_max,
        worker_idle_secs,
        notes,
    }
}

fn print_section(title: &str) {
    println!("\n=== {title} ===");
}

/// Entry point for `mivi doctor`. Prints hardware, dependency, model-catalog,
/// and config findings plus the recommended plan and its export form.
pub fn run_doctor() {
    println!("MIVI doctor — environment diagnosis");
    let snapshot = read_system_snapshot();

    print_section("Hardware");
    println!(
        "RAM total/available : {} MB / {} MB",
        snapshot.total_ram_mb, snapshot.available_ram_mb
    );
    println!("Logical CPUs        : {}", snapshot.logical_cpus);

    print_section("Dependencies");
    println!(
        "bin/llama-server    : {}",
        if snapshot.llama_server_present {
            "found"
        } else {
            "MISSING"
        }
    );
    println!(
        "bin/llama-cli       : {}",
        if snapshot.llama_cli_present {
            "found"
        } else {
            "MISSING"
        }
    );
    println!("models/*.gguf       : {} file(s)", snapshot.gguf_count);
    if !snapshot.llama_server_present && !snapshot.llama_cli_present {
        println!("  ! Neither llama.cpp binary found — only native/unit-test usage will work.");
    }

    print_section("Model catalog");
    match ModelCatalog::load_default() {
        Ok(catalog) => {
            for model in catalog.models.iter() {
                let status = if model.enabled {
                    if std::path::Path::new(&model.path).is_file() {
                        "enabled, file found"
                    } else {
                        "ENABLED BUT FILE MISSING"
                    }
                } else {
                    "disabled"
                };
                println!("{:<18} {:<40} {}", model.id, model.path, status);
            }
        }
        Err(err) => println!("Could not load configs/models.json: {err}"),
    }

    let config = RuntimeConfig::global();
    print_section("Effective runtime config");
    println!("mode            : {:?}", config.mode);
    println!(
        "context budget  : {} tokens (system {} / recent {})",
        config.context.max_input_tokens,
        config.context.system_tokens,
        config.context.recent_turn_tokens
    );
    println!("ram target      : {} MB", config.ram_target_mb);
    println!("threads         : {}", config.threads);
    println!("kv cache type   : {}", config.kv_cache_type);
    if config.lora_args.is_empty() {
        println!("lora adapters   : none (set MIVI_LORA_ADAPTERS=\"path[=scale],...\")");
    } else {
        println!(
            "lora adapters   : {}",
            config
                .lora_args
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    print_section("Recommended preset");
    let plan = recommend(&snapshot);
    println!("MIVI_RUNTIME_MODE={}", plan.runtime_mode);
    println!(
        "MIVI_ULTRA_LOW_RAM={}",
        if plan.ultra_low_ram { "1" } else { "0" }
    );
    println!("MIVI_CONTEXT_BUDGET={}", plan.context_budget);
    println!("MIVI_RAM_TARGET_MB={}", plan.ram_target_mb);
    println!("MIVI_MODEL_CACHE_MAX={}", plan.model_cache_max);
    if plan.worker_idle_secs > 0 {
        println!("MIVI_WORKER_IDLE_SECS={}", plan.worker_idle_secs);
    }
    for note in &plan.notes {
        println!("note: {note}");
    }
    println!("\nExport form:");
    println!(
        "export MIVI_RUNTIME_MODE={} MIVI_ULTRA_LOW_RAM={} MIVI_CONTEXT_BUDGET={} MIVI_RAM_TARGET_MB={} MIVI_MODEL_CACHE_MAX={}",
        plan.runtime_mode,
        if plan.ultra_low_ram { 1 } else { 0 },
        plan.context_budget,
        plan.ram_target_mb,
        plan.model_cache_max
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(total: usize, avail: usize) -> SystemSnapshot {
        SystemSnapshot {
            total_ram_mb: total,
            available_ram_mb: avail,
            logical_cpus: 8,
            llama_server_present: true,
            llama_cli_present: true,
            gguf_count: 4,
        }
    }

    #[test]
    fn tiny_machine_gets_spawn_ultra_low() {
        let plan = recommend(&snap(4000, 1800));
        assert_eq!(plan.runtime_mode, "spawn");
        assert!(plan.ultra_low_ram);
        assert_eq!(plan.model_cache_max, 1);
        // Context must stay above the hard floor.
        assert!(plan.context_budget >= crate::constants::MIN_CONTEXT_TOKENS);
        assert!(plan.notes.iter().any(|n| n.contains("spawn")));
    }

    #[test]
    fn mid_machine_gets_worker_eco() {
        let plan = recommend(&snap(8000, 4500));
        assert_eq!(plan.runtime_mode, "worker-eco");
        assert!(!plan.ultra_low_ram);
        assert_eq!(
            plan.worker_idle_secs,
            crate::constants::DEFAULT_WORKER_IDLE_SECS
        );
        assert_eq!(
            plan.context_budget,
            crate::constants::DEFAULT_CONTEXT_TOKENS
        );
    }

    #[test]
    fn big_machine_gets_worker_hot_with_longer_idle() {
        let plan = recommend(&snap(32000, 24000));
        assert_eq!(plan.runtime_mode, "worker-hot");
        assert!(plan.worker_idle_secs > crate::constants::DEFAULT_WORKER_IDLE_SECS);
    }

    #[test]
    fn unparsable_meminfo_degrades_to_safe_small_plan() {
        let mut s = snap(0, 0);
        s.llama_server_present = false;
        s.gguf_count = 0;
        let plan = recommend(&s);
        assert_eq!(plan.runtime_mode, "spawn");
        assert!(plan.ultra_low_ram);
        assert!(plan.notes.iter().any(|n| n.contains("download_models.py")));
    }

    #[test]
    fn meminfo_parser_reads_this_machine_or_bails_cleanly() {
        match read_meminfo_mb() {
            Some((total, avail)) => {
                assert!(total > 0);
                assert!(avail <= total);
            }
            None => {
                // Non-Linux or restricted /proc: parser must fail soft.
            }
        }
    }
}
