# Context Graph POC - Repository Instructions

## Purpose

This repository implements the Context Graph POC described in the project specification.
The attached Context Graph specification is the functional source of truth for what the system must do.

Before making any change, read:
1. This `AGENTS.md`
2. The relevant Context Graph specification
3. The existing code and tests related to the change

Do not implement or change behavior that contradicts the specification unless the requirement has explicitly been changed.

## Code Quality

- Write clean, readable, production-quality Python.
- Keep the implementation minimal. Do not add unnecessary code, abstractions, frameworks, or dependencies.
- Prefer simple, understandable solutions over clever solutions.
- Follow OOP and SOLID principles where they genuinely improve structure and reusability.
- Reuse existing components instead of implementing the same functionality in multiple places.
- Keep each responsibility in one appropriate module/class.
- Maintain a single source of truth for business rules, configuration, schemas, policies, and constants.
- A small future change should normally require changes only in the relevant module(s), not duplicated changes across the repository.

## Typing and Models

- Use strong type hints throughout the code.
- Keep the code Pyright-clean.
- Use Pydantic for structured models, validation, configuration, and data/API boundaries wherever appropriate.
- Prefer enums and typed models over scattered string literals and untyped dictionaries.
- Keep schemas centralized and versioned when required.

## Architecture

- Keep orchestration separate from business logic.
- Keep providers, persistence, models, policies, configuration, and utilities separated by responsibility.
- External dependencies must be replaceable without rewriting core business logic.
- Do not duplicate business rules across layers.
- Do not introduce an abstraction unless there is a concrete need for it.

## Async and Performance

- Use async I/O wherever it provides a real benefit.
- Use concurrency for independent I/O-bound operations such as LLM calls, embedding calls, and database operations.
- Preserve required ordering and correctness, especially within the same Session.
- Batch expensive operations where practical.
- Avoid unnecessary LLM calls, database calls, network calls, and repeated computation.
- Keep memory usage bounded.

## Reliability

- Design operations to be safely retryable and idempotent where required.
- Never silently swallow errors.
- Handle failures explicitly and preserve useful error information.
- Do not introduce behavior that can silently skip or corrupt data.
- Validate inputs and outputs at system boundaries.

## Configuration

- Do not hardcode values that should be configurable.
- Keep configuration in a central location.
- Do not scatter thresholds, limits, model settings, prompts, or policies across unrelated files.
- Secrets and credentials must never be hardcoded.

## Testing

- Add tests for new behavior and meaningful changes.
- Test important edge cases and failure paths.
- Run relevant unit/integration tests after changes.
- Run Pyright after meaningful code changes.
- Do not consider a change complete until imports, dependencies, typing, tests, and integration points have been checked.

## Change Discipline

- Before changing code, understand the existing implementation and its dependencies.
- Make the smallest change that correctly satisfies the requirement.
- Do not refactor unrelated code during a focused change.
- Do not duplicate functionality that already exists.
- After a change, check whether the change affects other modules, interfaces, schemas, tests, or configuration.
- Keep the repository consistent and buildable after every change.

## Implementation Process

When implementing a large feature:

1. Break it into logical stages.
2. Implement one stage at a time where practical.
3. Verify dependencies and integration after each stage.
4. Run tests and Pyright after each meaningful stage.
5. Continue only after the current stage is correct.
6. Do not implement the entire feature in one large uncontrolled change.

## Repository Safety

Do not add new infrastructure, libraries, services, abstractions, or patterns unless they are required by the specification or provide a clear implementation benefit.

Do not silently change architectural decisions documented in the Context Graph specification.

If a requirement changes, update the affected source of truth and then make the smallest corresponding code changes.