# Python Model Pipeline Working Agreement

## Mission And Scope

`quantized-viet-asr-model` owns model preparation, quantization, validation, Qualcomm AI Hub compilation, evidence, and Android bundle synchronization for Zipformer and VPCD.

These instructions apply to the entire repository. Preserve unrelated work in a dirty worktree and never reset, discard, commit, or publish changes without explicit authorization.

## Required Work Lifecycle

Every request that changes a tracked source, test, documentation, configuration, or artifact file must follow this lifecycle:

1. Perform read-only discovery of the relevant code, tests, documentation, and current worktree state.
2. Create a dated plan in `docs/plans/active/` from `docs/plans/TEMPLATE.md` before making any other tracked change for the request.
3. Implement the selected approach task by task.
4. Immediately after a task's output and verification gate pass, check its box and append a concise progress-log entry to the active plan.
5. Update the relevant canonical documentation alongside source changes.
6. Run every plan-level verification gate.
7. Add closure evidence, set the plan status to `Completed`, and move the same plan file to `docs/plans/completed/`.

Read-only analysis, explanations, code review, and validation commands that do not change tracked files do not require a plan.

Never check a task before its proof passes. A blocked plan stays in `active/` with the blocker, attempted checks, and required next action recorded. Never move a plan with failing or skipped required gates into `completed/`.

## Plan And Decision Standard

Use the filename `YYYY-MM-DD-<short-kebab-case-topic>.md`. If that name already exists, add a meaningful disambiguating suffix.

Every active plan must record:

- goal, scope, and explicit exclusions;
- at least two genuinely viable approaches;
- advantages and trade-offs of each approach;
- the selected approach and why each alternative was rejected;
- repository surfaces and canonical docs expected to change;
- checkbox tasks with a concrete output and verification gate;
- a chronological progress log;
- completion evidence and repository update notes.

Small changes may use short entries, but they may not omit the alternatives or decision rationale. Prefer the smallest approach that satisfies the contract, preserves artifact truth, and avoids speculative generalization.

## Documentation Contract

An implementation plan is an operational record, not a substitute for canonical documentation.

Every source-code change must update at least one relevant canonical document outside `docs/plans/`. Update an existing document when the changed behavior falls within its boundary. Create a new canonical Markdown document only when no existing document owns the subject, and add the new document to the appropriate README or index in the same change.

Files under `docs/plans/` are the required exception to the new-document rule. They track work rather than define current product behavior.

Keep documentation portable: use repository-relative paths and placeholders, never machine-local absolute paths or secrets.

## Python Function Documentation Standard

Every handwritten function or method in `src/model_pipeline/**` and `test/**` must have a Google-style English docstring. This includes nested functions, constructors, properties, protocol methods, test functions, and test helpers.

Generated code, dependencies, lambdas, and anonymous callbacks are excluded.

Each docstring must contain:

- a one-sentence summary that explains the function's purpose;
- `Args:` for every parameter except `self` and `cls` when inputs exist;
- `Returns:` for every function, including functions that return `None`;
- `Yields:` instead of `Returns:` for generators;
- `Raises:` only for exceptions that are part of the caller-visible contract.

Describe semantics, side effects, units, accepted forms, and invariants. Do not merely repeat type annotations or narrate the implementation line by line.

Example:

```python
def resolve_recipe(model: str, configuration: str) -> RecipeSpec:
    """Resolve the canonical pipeline recipe for a model configuration.

    Args:
        model: Canonical model family name.
        configuration: Requested precision, shape, scope, and execution configuration.

    Returns:
        The validated recipe used by all pipeline stages.

    Raises:
        ValueError: If the model or configuration is unsupported.
    """
```

## Descriptive Naming Contract

Public and internal identifiers must describe observable technical properties: model family, quantization engine, weight and activation precision, fixed input shape, operator scope, or execution target. Write a technical abbreviation in full on first use in prose.

Do not use notebook names, numbered experiment labels, chronological stage labels, subjective quality labels, or deployment-campaign terminology as model identity. Do not preserve aliases for removed identities. Use `configuration` for recipe and command-line selection. Name intermediate values by state or role, such as `prepared_components`, `quantized_components`, `validated_components`, and `compiled_components`; name booleans by the condition they represent.

## Verification Discipline

Use the repository's configured Python environment for verification. Documentation should describe project-specific contracts and commands, not environment activation or interpreter-discovery steps.

Match verification to the changed surface, then run all gates named by the active plan. Fresh command output is required before claiming success. At minimum, source changes require focused tests, full pytest, `compileall src`, and `git diff --check`; model or artifact changes also require the graph, checksum, packaging, AI Hub, and Android gates defined by the canonical operations documentation.

Do not infer NPU success from filenames, manifests alone, or CPU execution. Preserve component-level precision, quantization scope, execution target, checksum, and provenance truth.

## Onboarding Order

1. `README.md`
2. `AGENTS.md`
3. `docs/README.md`
4. `docs/getting-started.md`
5. `docs/architecture.md`
6. `docs/source-code-guide.md`
7. The active plan for the current task
8. The model recipe and operations documents relevant to the task

Complete the common foundation before following the role-specific paths in `docs/README.md`. Read BKMeeting documentation only after reaching the Android handoff boundary; do not infer Android behavior from the Python package alone.
