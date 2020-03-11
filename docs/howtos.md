# FLAX HOWTOs

FLAX HOWTOs explain how to implement standard techiques in FLAX. For instance,
the HOWTO for ensembling learning demonstrates what changes should be made to
the standard MNIST example in order to train an ensemble of models. As such, 
HOWTOs are simply "diffs".

## List of HOWTOs

In essence, a HOWTO is a set of changes that are applied to the FLAX master 
branch on Github. HOWTOs are currently simply diff views between HOWTO branches 
and the master branch. Soon, they will be merged with the rest of our 
documentation through readthedocs.

Currently the following HOWTOs are available:

* [Ensembling](https://github.com/google-research/flax/compare/prerelease..howto-ensembling?diff=split)

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
./howtos/script/pack_howto_diff.sh
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

> :warning: TODO: This is still in progress. Currently conflicts are resolved by
            checking whether the Github action fails, and if so, manually making
            the required fixes.

## Modifying an exiting HOWTO

> :warning: TODO: Write this out. The summary is: apply a diff locally, make the
            changes, pack the diff again and commit. It would be good to show 
            this with an example. Note editing this diff is not a good idea 
            since it is extremely error-prone.

