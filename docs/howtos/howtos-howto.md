# HOWTOs HOWTO

This document explains how the Flax HOWTO system works: How to submit
new HOWTOs, modify existing ones, and fix HOWTOs when base examples
change.

## How HOWTOs work

While HOWTOs are shown as changes on a branch, these branches are read-only for
the users. HOWTOs are represented as diff files in the repository as well. For 
instance, the "ensembling howto" is represented in two ways:

* In the branch `howto-ensembling` (only on Github).
* In the file `howtos/diffs/howto-ensembling.diff` (in the repository).

The branch `howto-ensembling` is simply the result of applying 
`howto/howto-ensembling.diff` to the master branch.

This setup allows us to ensure all local modifications to HOWTOs can be packed
in diff files locally, pushed to origin and applied to the master branch 
automatically.

This process is automatic, which means that users usually won't notice this, 
except if they want to add a new HOWTO, or their changes require making changes
to HOWTOs. We explain these two cases below.

## Adding a new HOWTO

In order to create a new HOWTO, you can simply make the desired changes local
changes and run:

```
./howtos/scripts/pack_howto_diffs.sh
```

This will pack the changes you made into a diff file (you can set the name of
the howto when executing this script), and stage it for committing. It will also
revert all changes you made, but you can directly recover these changes by
simply applying the diff:

```
git apply <diff_file>
```

After pushing your commit, the newly submitted diff file will be unpacked on
Github with a Github action and a new howto branch will be created for it.

The workflow script for the Github action can be found at 
`.github/workflows/apply-howto-branches.yml`

## Resolving a conflict with an existing HOWTO

This is still a work in progress. Currently, conflicts are resolved by manually
making the required fixes. The suggestions below can help speed up that process.

### Start with a clean and updated copy of `master`
As an example, assuming you have `https://github.com/google/flax` set as
`upstream`:
```bash
git checkout master
git fetch upstream
git rebase upstream/master
```
If your local copy of `flax` isn't clean, you may need to stash your changes
elsewhere (e.g., via `git stash`) or otherwise remove your changes.

### Option 1: Use vanilla `git apply`:

#### Automatically apply as many diff hunks as possible
The `--reject` flag tells `git apply` to apply whatever hunks in the supplied
diff file it can and outputs the rejected hunks to a `*.rej` file in the same
directory as the modified files (typically `examples/[EXAMPLE]`). The indices
of the rejected hunks are also printed to stdout.

```bash
git apply howtos/diffs/$name.diff --reject
```

#### Go through the remaining hunks manually
Examine the changes in each `*.rej` diff file for all hunks that couldn't be
merged automatically. If the example code hasn't changed significantly, the
work typically involves just finding the right place to insert and remove the
lines mentioned in the diff. If the example code _has_ changed significantly,
the work may also involve understanding the intent of the `howto` or contacting
the author(s) of the `howto`.

Tip: It's usually enough to insert/remove code from the hunks to match the
updated file and make sure that there are three unchanged lines before and
after the change (note that first line starts immediately after the `@@`
marker). Don't bother to update all the line numbers manually. Rather try to
reapply the patch (with command above) and once all hunks pass you can
regenerate the patch from the modified file, as described in the next section.

### Option 2: Use 3-way merge:
In case the vanilla apply doesn't work cleanly, you can use Git's 3-way merge,
which lets you resolve conflicts like you would with normal git merges.

```bash
git apply -3 $diff_file
```

This should apply the diff, where merge conflicts appear in the files
with `<<<`, `===`, `>>>` annotations. Edit the files that have a merge conflict
like you would with a normal Git merge conflict.

### Re-pack the HOWTO
Pack the changes on your branch into a `howto` via the script below. In this
case, we can overwrite the existing diff with the same name, where the name of
a `howto` is the filename without the extension (e.g., the name for the
`distributed-training.diff` `howto` is `distributed-training`).
```bash
./howtos/scripts/pack_howto_diffs.sh
```

## Modifying an existing HOWTO

1. Fetch upstream and rebase onto upstream/master (see above).
1. Apply the HOWTO : `git apply howtos/diffs/$name.diff`
2. Modify the modified files, without staging them.
3. Re-pack the HOWTO (supplying same `$name`).
