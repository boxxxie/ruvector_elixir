# Specification: Upgrade ruvector_elixir to Elixir 1.20.5 & Erlang/OTP 29.1.2

> [!IMPORTANT]
> **Track Metadata:**
> - **Track ID:** `elixir_1_20_upgrade_20261118`
> - **Type:** `chore`
> - **Target Runtime:** Elixir `1.20.5-otp-29` / Erlang/OTP `29.1.2`

---

## 1. Overview & Current State

- **Repository:** `ruvector_elixir` (`/home/pauli/projects/elixir/ruvector`)
- **Current `mix.exs` constraint:** `~> 1.14` (target: `~> 1.18 (or ~> 1.20)`)
- **Current `.tool-versions`:** Not present (inherits global `~/.tool-versions`)
- **CI Workflows:** None detected
- **Dockerfiles:** None detected

### Deployment / Runtime Notes
Library/NIF repo: add `.tool-versions` (`erlang 29.1.2`, `elixir 1.20.5-otp-29`) and verify Rustler/NIF build on OTP 29.

---

## 2. Functional Requirements

1. **Toolchain & Dependency Alignment:**
   - Pin `erlang 29.1.2` and `elixir 1.20.5-otp-29` in `.tool-versions`.
   - Update `mix.exs` `:elixir` constraint to `~> 1.18 (or ~> 1.20)` and verify all `mix.lock` dependencies compile cleanly on Erlang/OTP `29.1.2` + Elixir `1.20.5`.
2. **Elixir 1.19–1.20 & OTP 28/29 Remediation:**
- Fix any Elixir 1.19/1.20 gradual type-system warnings (struct updates without prior `%Struct{} = var` pattern match, protocol dispatch errors, guard/whole-body/map inference errors, redundant/unreachable clauses).
- Pin dynamic bitstring size variables (`<<x::binary-size(^len), rest::binary>>`) per Elixir 1.20.
- Run `mix format --migrate` (reviewing `unless` -> `if !`, charlists -> `~c`, bitstring parens).
- Move any compile-time `~r/.../` module attributes or `defstruct` regex defaults into runtime functions or `~r/.../E` for OTP 28/29 NIF regex compatibility.
- Decide whether `mix.exs` keeps a wider minimum floor (e.g. `~> 1.19`) for downstream compatibility while testing & pinning `1.20.5-otp-29` in `.tool-versions` and CI, or bumps `elixir:` to `~> 1.20`.
3. **CI / Container Alignment:**
   - Update all CI workflow matrices/images (None detected) and container base images (None detected) to `1.20.5` / `29.1.2`.

---

## 3. Acceptance Criteria

- [ ] `.tool-versions` pins `erlang 29.1.2` and `elixir 1.20.5-otp-29`.
- [ ] `mix.exs`, CI workflows, and Dockerfiles (if present) reflect the target Elixir `1.20.5` / OTP `29.1.2` configuration.
- `mix format --check-formatted` passes with zero warnings/errors.
- `mix compile --warnings-as-errors` passes with zero warnings/errors.
- `mix test` passes with zero warnings/errors.
