# How to Contribute

Everyone can contribute to Flax, and we value everyone's contributions. 
You can contribute in many more ways than just writing code. Answering questions
on our [Discussions page](https://github.com/google/flax/discussions), helping
each other, and improving our documentation are extremely valuable to our
ecosystem.

We also appreciate if you spread the word, for instance by starring our Github
repo, or referencing Flax in blog posts of projects that used it.

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Ways to contribute

We welcome pull requests (PRs), in particular for those issues
[marked as PR-ready](https://github.com/google/flax/issues?q=is%3Aopen+is%3Aissue+label%3A%22Status%3A+pull+requests+welcome%22). For other proposals, we ask that you first open a Github Issue or
Github Discussion to discuss your planned contribution.

## Contributing code using Pull Requests

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

### Create a Pull Request in your own branch

1. Fork the Flax repository by clicking the 'Fork' button on the
   [repository page](http://www.github.com/google/flax). This creates a copy
   of the Flax repository in your own account.

2. Install Python >=3.6 and `svn` for running the tests (see below).

3. (Optional) Create a virtual environment or a Docker container. See 
   [`dev/README.md`](https://github.com/google/flax/blob/main/dev/README.md)
   for details on how to setup a Docker Container. To setup a virtual environment,
   run the following:

   ```bash
   python3.6 -m virtualenv env
   . env/bin/activate
   ```
  
   This ensures all your dependencies are installed in this environment.

4. `pip install` your fork from source. This allows you to modify the code
   and immediately test it out:

   ```bash
   git clone https://github.com/YOUR_USERNAME/flax
   cd flax
   pip install ".[testing]"
   pip install -e .
   pip install -r docs/requirements.txt
   ```

5. Setup pre-commit hooks, this will run some automated checks during each `git` commit and
   possibly update some files that require changes.
   
   ```bash
   pip install pre-commit
   pre-commit install
   ```

6. Add the Google Flax repo (not your fork) as an upstream remote, so you can use it to sync your
   changes.

   ```bash
   git remote add upstream http://www.github.com/google/flax
   ```


7. Create a branch where you will develop from:

   ```bash
   git checkout -b name-of-change
   ```

8. Implement your changes using your favorite editor (we recommend
   [Visual Studio Code](https://code.visualstudio.com/)).

   Make sure the tests pass by running the following command from the top of
   the repository:

   ```bash
   ./tests/run_all_tests.sh
   ```

9.  Once your change is done, create a commit as follows 
   ([how to write a commit message](https://chris.beams.io/posts/git-commit/)):

   ```bash
   git add file1.py file2.py ...
   git commit -m "Your commit message"
   ```

   Then sync your code with the main repo:

   ```bash
   git rebase upstream/main
   ```

11. Finally push your commit on your development branch and create a remote 
   branch in your fork that you can use to create a Pull Request from:

   ```bash
   git push --set-upstream origin name-of-change
   ```
   
   After running the command, you should see a Github link in your terminal output that you can click on to create a Pull Request.
   If you do not see this link in the terminal after doing a `git push`, go to the Github web UI; there should be a button there that lets you turn the commit into a Pull Request yourself.

11. Make sure your PR passes the 
   [PR checklist](https://github.com/google/flax/blob/main/.github/pull_request_template.md#checklist).
   If so, create a Pull Request from the Flax repository and send it for review.
   Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/)
   for more information on using pull requests.

### Update notebooks

We use [jupytext](https://jupytext.readthedocs.io/) to maintain two synced copies of the notebooks
in `docs/notebooks`: one in `ipynb` format, and one in `md` format. The advantage of the former
is that it can be opened and executed directly in Colab; the advantage of the latter is that
it makes it much easier to track diffs within version control.

#### Editing ipynb

For making large changes that substantially modify code and outputs, it is easiest to
edit the notebooks in Jupyter or in Colab. To edit notebooks in the Colab interface,
open <http://colab.research.google.com> and `Upload` from your local repo.
Update it as needed, `Run all cells` then `Download ipynb`.
You may want to test that it executes properly, using `sphinx-build` as explained above.

#### Editing md

For making smaller changes to the text content of the notebooks, it is easiest to edit the
`.md` versions using a text editor.

#### Syncing notebooks

After editing either the ipynb or md versions of the notebooks, you can sync the two versions
using [jupytext](https://jupytext.readthedocs.io/) by running `jupytext --sync` on the updated
notebooks; for example:

```
pip install jupytext==1.13.8
jupytext --sync docs/notebooks/quickstart.ipynb
```

The jupytext version should match that specified in
[.pre-commit-config.yaml](https://github.com/google/flax/blob/main/.pre-commit-config.yaml).

To check that the markdown and ipynb files are properly synced, you may use the
[pre-commit](https://pre-commit.com/) framework to perform the same check used
by the github CI:

```
git add docs -u  # pre-commit runs on files in git staging.
pre-commit run jupytext
```

#### Creating new notebooks

If you are adding a new notebook to the documentation and would like to use the `jupytext --sync`
command discussed here, you can set up your notebook for jupytext by using the following command:

```
jupytext --set-formats ipynb,md:myst path/to/the/notebook.ipynb
```

This works by adding a `"jupytext"` metadata field to the notebook file which specifies the
desired formats, and which the `jupytext --sync` command recognizes when invoked.

#### Notebooks within the sphinx build

Some of the notebooks are built automatically as part of the pre-submit checks and
as part of the [Read the docs](https://flax.readthedocs.io/en/latest) build.
The build will fail if cells raise errors. If the errors are intentional, you can either catch them,
or tag the cell with `raises-exceptions` metadata ([example PR](https://github.com/google/jax/pull/2402/files)).
You have to add this metadata by hand in the `.ipynb` file. It will be preserved when somebody else
re-saves the notebook.

We exclude some notebooks from the build, e.g., because they contain long computations.
See `exclude_patterns` in [conf.py](https://github.com/google/flax/blob/main/docs/conf.py).

### Updating the Pull Request contents

Every Pull Request should ideally be limited to just one commit, so if you have multiple commits please squash them.

Assuming you now have only one commit in your Pull Request, and want to add changes requested during review:

1. Make the changes locally in your editor.
2. Run `git commit -a --amend`. This updates the commit contents and allows you to edit the commit message.
3. At this point, `git push` alone will result in an error. Instead, use `git push --force`.
4. Check that it's done: The changes to your commit should be immediately reflected in the Github web UI.

## Troubleshooting

### Too many commits in a PR

If your PR has too many commits associated with it, then our build process will
fail with an error message. This is because of two reasons:

* We prefer to keep our commit history clean.

* Our source sync process will fail if our commit tree is too large.

If you encounter this error message, you should squash your commits. In order to
rebase your branch to main and creating a new commit containing all your
changes, please run the following command:

```bash
git rebase main && git reset --soft main && git commit
```

This will apply all your changes to the main branch. Note that if you had to
resolve any conflicts while working on your change (for instance, you did a
`pull upstream main` which led to conflict), then you will have to resolve these
conflicts agin.

Once you successfully rebased your branch, you should push your changes. Since
you are changing the commit history, you should use `git push --force`.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.
