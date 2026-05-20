# llm-multimodal

Standalone fork of `llm-multimodal`, extracted from `lightseekorg/smg` at
`crates/multimodal`.

This repository keeps the upstream crate history as a split subtree, with commit
messages rewritten so SMG pull request references use fully qualified links such
as `lightseekorg/smg#1459`. Local fork changes should stay as small top-level
commits on `main` so they can be replayed on top of a fresh split from SMG.

Current upstream split base:

- SMG source path: `crates/multimodal`
- Raw SMG split commit: `2c09471d9f26f10ebfeeba7a282e68d75fc8ba89`
- Local rewritten split commit: `d552f139ad89f61f43447fa804976a0f3d1294f6`
- Corresponding SMG PR at extraction time: `lightseekorg/smg#1459`

## License

Licensed under the Apache License, Version 2.0. This repository was extracted
from `lightseekorg/smg/crates/multimodal`; see `LICENSE` for details.

## Sync With Upstream

To refresh this fork from a newer SMG checkout:

```sh
cd ../smg
git fetch origin
git checkout main
git pull --ff-only
NEW_SPLIT=$(git subtree split --prefix=crates/multimodal HEAD)

cd ../llm-multimodal
git fetch smg "$NEW_SPLIT"
git checkout -B upstream-refresh FETCH_HEAD
git filter-repo --force --refs upstream-refresh \
  --message-callback 'import re
return re.sub(rb"(?<!lightseekorg/smg)#([0-9]+)", lambda m: b"lightseekorg/smg#" + m.group(1), message)'
git checkout main
git rebase --onto upstream-refresh d552f139ad89f61f43447fa804976a0f3d1294f6 main
```

After resolving any conflicts, run:

```sh
cargo check --all-targets
```

If the rebase succeeds, update both split base entries recorded above. Also
replace the old `d552f139ad89f61f43447fa804976a0f3d1294f6` base in the
`git rebase --onto` command with the new local rewritten split commit. Keep
semantic changes, dependency rewrites, and local integration fixes in separate
commits after the rewritten upstream split base.
