.PHONY: check-agent check-agent-live

check-agent:
	scripts/check_agent_compat.py --live off

check-agent-live:
	scripts/check_agent_compat.py --live auto
