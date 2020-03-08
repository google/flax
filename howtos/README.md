# FLAX HOWTOs

## Introduction

FLAX HOWTOs explain how to implement standard techiques in FLAX. For instance,
the HOWTO for ensembling learning demonstrates what changes should be made to
the standard MNIST example in order to train an ensemble of models. As such, 
HOWTOs are showns as "diffs" from the standard FLAX implementation.

## List of HOWTOs

In essence, a HOWTO is a set of changes that are applied to the FLAX master 
branch on Github. HOWTOs are currently simply diff views between HOWTO branches 
and the master branch. Soon, they will be merged with the rest of our 
documentation through readthedocs.

Currently the following HOWTOs are available:

* [Distributed Training](https://github.com/marcvanzee/flax/compare/prerelease..howto-distributed-training?diff=split)
* [Ensembling](https://github.com/marcvanzee/flax/compare/prerelease..howto-ensembling?diff=split)
* [Polyak Averaging](https://github.com/marcvanzee/flax/compare/prerelease..howto-polyak-averaging?diff=split)
* [Training only a few layers](https://github.com/marcvanzee/flax/compare/prerelease..howto-training-subset-layers?diff=split)

## How HOWTOs work

While HOWTOs are shown as changes on a branch, these branches are read-only for
the users. HOWTOs are represented as diff files in each client as well. For 
instance, the "ensembling howto" is represented in two ways:

* In the branch `howto-ensembling`.
* In the file `howto/howto-ensembling.diff`, which exists in the master branch.

The branch `howto-ensembling` is simply the result of applying 
`howto/howto-ensembling.diff` to the master branch.

This setup allows us to ensure all local modifications to HOWTOs can be packed
in diff files locally, pushed to origin and applied to the master branch 
automatically.

This process is automatic, using local pre-commit hooks and automatic changes
that are applied after each push to master. This means that users usually won't
notice this, except if they want to add a new HOWTO, or their changes require
making changes to HOWTOs. We explain these two cases below.

## Adding a new HOWTO

In order to create a new HOWTO, you can simply commit your desired changes to a
new local branch called `howto-<name>`, and run

```
./howto/create_new_howto.sh howto-<name>
```

This will pack the changes made on `howto-<name>` into a diff file and commit 
this to the master branch.

> :warning: `create_new_howto.sh` is basically `pack_howto`, but I think using
  dedicated names makes it easier for the user.

### Example: inverse batching

Suppose that you want to create a howto for a machine learning technique called
"inverse batching".

First you make sure your repository is up to date with origin and you have no
open changes

```
$ git status
On branch prerelease
Your branch is up to date with 'origin/prerelease'.

nothing to commit, working tree clean
```

Then you create a new branch called `howto-inverse-batching`. It is important
that your branch starts with `howto-`.

```
$ git checkout -b howto-inverse-batching
Switched to a new branch 'howto-inverse-batching'
```

Suppose you make some changes to `mnist/train.py`.

```
$ git status
On branch howto-inverse-batching
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
  modified:   examples/mnist/train.py

no changes added to commit (use "git add" and/or "git commit -a")
```

Once you are happy with your change, you commit them to the howto branch.

```
$ git commit -am "Added howto for inverse batch"
```

Then pack the howto by running the following command:
```
$ ./howtos/create_new_howto.sh howto-inverse-batching
WARNING: this will pack your howto into a diff and remove the branch.
Are you sure? y

Packed howto-inverse-batching into howtos/howto-inverse-batching.diff
Creating commit
Removed branch howto-inverse-batching
Done!
```

> :warning: Note this means the howto branch is removed entirely, which is
  necessary to remove the user's commit.

You can now simply push your change to origin.

```
$ git push
Enumerating objects: 6, done.
Counting objects: 100% (6/6), done.
Delta compression using up to 12 threads
Compressing objects: 100% (4/4), done.
Writing objects: 100% (4/4), 519 bytes | 519.00 KiB/s, done.
Total 4 (delta 2), reused 0 (delta 0)
remote: Resolving deltas: 100% (2/2), completed with 2 local objects.
To https://github.com/marcvanzee/flax.git
   bc15979..736c50f  prerelease -> prerelease
```

After this push, the newly submitted diff file will be unpacked on Github and a
new howto branch will be created for it.

## Resolving a conflict with an existing HOWTO

In order to ensure local changes are not breaking existing HOWTOs, the following
procedure is executed automatically before every commit:

* Each diff file in `howto/` is applied to the current master branch 
  (containing the changes you'd like to commit).
* If all diffs can be applied unproblematically, nothing happens.
* If a diff cannot be applied, the user has to resolve the conflicts.

The result of each unsuccesfully applied diff is a 3-way merge on a temporary
branch. The user has to resolve the conflicts on this branch, and when all
conflicts are resolved, run `howto/resolve_howtos.sh`, which will pack the
new howtos into diff files and create a new commit.

### Example: Changing an example

Suppose you make some changes to the function `create_optimizer` in the MNIST
example. Suppose further that there are two HOWTOs, namely for ensembling and
distributed learning that are also making changes to this function. This means
that you will get an error when trying to commit your change.

```
$ git commit -am "Update optimizer to use FastOptimizer"
[prerelease f0a37e6] Update optimizer to use FastOptimizer
ERROR: Merge conflict(s) with HOWTOs. Please resolve conflicts in the following
branches and run howtos/resolve_howto_conflicts.sh:
howto-ensembling-temp
howto-distributed-training-temp
```

> :warning: The user should have no changes that are not added to the index,
  otherwise 3way merge fails with `Error ... does not match index`

Next, you switch to each branch and resolve the conflict. Let's look at the 
first branch.

```
$ git checkout howto-ensembling-temp
Switched to branch 'howto-ensembling-temp'
$ git mergetool
Normal merge conflict for 'examples/mnist/train.py':
  {local}: modified file
  {remote}: modified file
Hit return to start merge resolution tool (meld): 
```

You can now resolve the 3-way merge in your favorite editor. In the case of
`meld` it may look like the following:

![alt text](3way.png "Resolving a 3-way merge conflict for a HOWTO branch")

Once you resolved all conflicts on all branches, you run the following command:

```
$ ./howtos/resolve_howto_conflicts.sh
WARNING: this will pack all temporary howto branches into diff files and remove
the branches.
Are you sure? y
Packed howto-ensembling-temp into howtos/howto-ensembling.diff
Packed howto-distributed-training-temp into howtos/howto-distributed-training.diff
Creating commit
Removed branch howto-ensembling-temp
Removed branch howto-distributed-training-temp
Done!
```


# STUFF BELOW IS OUTDATED


## HOWTO Branches

A HOWTO is a commit on a branch called `howto-<name>`, which is always applied
on top of the master branch. This means that HOWTOs are always kept "up to 
date": they are compatible with the latest version of our code.

![alt text](img/howto_git1.png "HOWTO branches and the master branch")

**Automatic rebasing.** In order to ensure HOWTO branches are never "behind" the
master branch, each HOWTO branch is automatically rebased when Pa commit is being
pushed to master. In essence, the following command is executed for each howto
branch:

```
git rebase master howto-branch
```

This update always succeed because of a pre-commit process that is executed on
every commit.

### HOWTO diff files

Besides HOWTO branches, we also represent each HOWTO as a diff file on the
master branch. This is to ensure that a user is able to update HOWTOs as part
of a changes that affect them. This would not be possible if the HOWTOs were 
only stored on a branch, since if someone would like to make a change that 
affects a howto, this would require updating both the HOWTO branch and the
master branch, which is not possible to do in a single commit. To simplify this,
we use a pre-commit hook that checks whether a change affects a howto. If so, 
the user is able to update the HOWTO locally, and the HOWTO is automatically
packed into a diff file, which is part of the pull request. On Github, the new
diff is then applied to the master branch, and the result is applied to the
HOWTO branch.



Let's look at an example

Suppose you commit some local changes to master, but these changes conflict with
some changes that were made in `howto-1`.

![alt text](img/howto_git2.png "HOWTO branches, the master branch, and a local
change")

This means that you should make changes on the howto-1, but of course you would 
like your own changes in your local master branch also to be committed. 
Unfortunately, this is not possible in a single commit, making the process of 
updating howtos a bit cumbersome.

**A HOWTO is locally represented in a diff file.** In order to simplify the 
process of updating howtos, we also represent each HOWTO in diff....,


