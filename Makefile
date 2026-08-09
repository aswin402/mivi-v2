.PHONY: check-agent check-agent-live build-native run-native check-native

check-agent:
	scripts/check_agent_compat.py --live off

check-agent-live:
	scripts/check_agent_compat.py --live auto

build-native:
	RUSTFLAGS="-C target-cpu=native" cargo build --release --features native

run-native:
	RUSTFLAGS="-C target-cpu=native" cargo run --release --features native -- serve

check-native:
	RUSTFLAGS="-C target-cpu=native" cargo test --release --features native
	RUSTFLAGS="-C target-cpu=native" cargo fmt --check
	RUSTFLAGS="-C target-cpu=native" cargo build --release --features native

