# e30 — bare repo + git worktrees

Local-only repo using the **bare-repo + worktrees** layout so `aoe`
(agent-of-empires) creates per-session worktrees cleanly. Same pattern as
`~/dev/swoop`.

## Layout

```
~/dev/e30/
├── .bare/      # bare repository — ALL git data lives here
├── .git        # file: "gitdir: ./.bare"  (lets git run from ~/dev/e30)
├── main/       # primary worktree, branch `main`  (the RAG project)
└── worktrees/  # aoe creates feature worktrees here -> worktrees/<branch>/
```

## How it works

- `.bare` is a bare git repo (it has no working tree of its own).
- The top-level `.git` file points git at `.bare`, so plain `git ...` from
  `~/dev/e30` works.
- Every checkout is a **linked worktree**: `main/` and anything under `worktrees/`.
  `main` is just the first worktree.
- `origin` is **self-referential** (`origin -> ~/dev/e30/.bare`). This is
  deliberate: aoe runs `git fetch origin main` before creating a worktree, and on
  a local-only repo with no real remote that warns/fails. Fetching from itself is a
  harmless no-op that succeeds. **Do not remove this remote.**

## AOE settings — profile: e30

File: `~/.agent-of-empires/profiles/e30/config.toml`

```toml
[worktree]
enabled = true
path_template = "../worktrees/{branch}"           # NON-bare repos only (unused here)
bare_repo_path_template = "./worktrees/{branch}"  # THIS bare repo -> ~/dev/e30/worktrees/<branch>
```

`bare_repo_path_template` is resolved relative to the dir that contains `.bare`
(i.e. `~/dev/e30`), so `./worktrees/{branch}` → `~/dev/e30/worktrees/<branch>`.
Base branch is auto-detected as `main` (the bare repo's `HEAD` → `refs/heads/main`).

## Manual worktree commands

Run from `~/dev/e30` (or any worktree):

```bash
# new branch off main
git -C ~/dev/e30 worktree add worktrees/my-feature -b my-feature main

# existing branch
git -C ~/dev/e30 worktree add worktrees/my-feature my-feature

git -C ~/dev/e30 worktree list
git -C ~/dev/e30 worktree remove worktrees/my-feature
git -C ~/dev/e30 worktree prune
```
