# Development Workflow — ruvector_elixir

## Standard Verification Gates

```bash
mix format --check-formatted
mix compile --warnings-as-errors
mix test
```

### Phase Completion Verification and Checkpointing Protocol
Each phase ends with:
`- [ ] Task: Conductor - User Manual Verification '<Phase Name>' (Protocol in workflow.md)`
