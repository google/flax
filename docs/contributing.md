# How to contribute

Everyone can contribute to Flax, and the Flax development team values everyone's contributions!
You can contribute in many more ways than just writing code. Answering questions
on the [Flax GitHub Discussions page](https://github.com/google/flax/discussions), helping
each other, and improving Flax documentation are extremely valuable to the Flax
ecosystem.

We also appreciate if you spread the word, for instance by starring the [Flax GitHub repository](https://github.com/google/flax),
or referencing Flax in blog posts of projects that used it.

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Ways to contribute

We welcome pull requests (PRs), in particular for those issues
[marked as PR-ready](https://github.com/google/flax/issues?q=is%3Aopen+is%3Aissue+label%3A%22Status%3A+pull+requests+welcome%22).
For other proposals, you should first open a GitHub Issue or a GitHub Discussion to
start a conversation about your planned contribution.

## Contributing code using pull requests

The Flax development team performs all development using [Git](https://git-scm.com/). To contribute,
you should have basic knowledge of [Git](https://git-scm.com/) and [GitHub](https://docs.github.com).
(You can learn how to set up Git by following Git's official
[Getting Started - First-Time Git Setup](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup)
and GitHub's [Set Up Git](https://docs.github.com/en/get-started/quickstart/set-up-git) guides.)

To contribute code to Flax on GitHub, follow these steps:

### To create a pull request from a fork

1. Using GitHub's web UI, fork the Flax repository by clicking the 'Fork' button on the
   [`github.com/google/flax` repository page](http://www.github.com/google/flax). This creates a
   fork (a copy) of the Flax repository in your own GitHub.

   Reference: [Creating a pull request from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

2. Install [Python >=3.7](https://www.python.org/downloads/).

3. (Optional) Create a virtual environment or a Docker container. See
   [`dev/README.md`](https://github.com/google/flax/blob/main/dev/README.md)
   for details on how to set up a Docker Container. To set up a virtual environment,
   run the following:

   ```bash
   python3 -m virtualenv env
   . env/bin/activate
   ```

   This ensures all your dependencies are installed in this environment.

4. Clone your local forked Flax repo with `git clone`. Then, install the required packages
   with [PyPi](https://pip.pypa.io/en/stable/cli/pip_install/). This enables you to immediately
   test the code after modifying it:

   ```bash
   git clone https://github.com/YOUR_USERNAME/flax
   cd flax
   pip install -e .[all]
   pip install -e .[testing]
   pip install -r docs/requirements.txt
   ```

5. Set up pre-commit hooks, this will run some automated checks during each `git` commit and
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


7. Create a branch, such as `my_development_branch`, you will develop from:

   ```bash
   git checkout -b my_development_branch
   ```

8. Implement your changes using your favorite editor (we recommend
   [Visual Studio Code](https://code.visualstudio.com/)).

   Make sure the tests pass by running the following command from the top of
   the repository:

   ```bash
   ./tests/run_all_tests.sh
   ```

9. Once you finish making changes, don't forget to create commits
   ([learn how to write a commit message](https://chris.beams.io/posts/git-commit/)):

   ```bash
   git add file1.py file2.py ...
   # or use `git add .` to add all changed files
   git commit -m "Your commit message"
   ```

   Then sync your code with the main repository:

   ```bash
   git rebase upstream/main
   ```

10. Finally, push your commit on your `my_development_branch`, and create a remote
   branch in your fork that you can use to create a pull request from:

   ```bash
   git push --set-upstream origin my_development_branch
   ```

   After running the command, you should get a GitHub link in your (VS Code) terminal output for creating a pull request.
   If you don't receive a link after `git push`, use the [GitHub web UI](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request?tool=webui) to create a pull request.

11. Make sure your pull request passes the
   [Flax PR checklist](https://github.com/google/flax/blob/main/.github/pull_request_template.md#checklist).
   If so, create a pull request from the Flax repository and send it for review.
   Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/)
   for more information on using pull requests.

You can learn more in GitHub's [Creating a pull request from a fork
](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork). documentation.

### Updating Jupyter Notebooks

We use [jupytext](https://jupytext.readthedocs.io/) to maintain two synced copies of docs
in `docs/notebooks`: one in the Jupyter Notebook (`.ipynb`) format, and one in Markdown (`.md`).

The former can be opened and executed directly in [Google Colab](https://colab.research.google.com/).
Markdown makes it easier to track changes/diffs within version control and, for example, GitHub
web UI, since `.ipynb` files are based on JSON.

#### Editing Jupyter Notebooks (`.ipynb`)

For making large changes that substantially modify code and outputs, it's recommended to edit
the notebooks in [Jupyter](https://jupyter.org/install) or in [Colab](https://colab.research.google.com/).

If you choose to work in Colab, go to **File** and click **Upload notebook**, then pick your file.
After loading it into Colab and editing it, make sure you run the cells, and that there aren't any errors.
Click on **Runtime**, then select **Run all**. After you finish, click **File** > **Download** > **Download ipynb**.
You may also want to test that the file executes properly by using `sphinx-build`, as explained above.

After you make changes in your Jupyter Notebook, follow the steps _Syncing notebooks_ below.

#### Editing Markdown files (`.md`)

For making smaller changes to the text content of the notebooks, it is easiest to edit the
`.md` versions using a text editor.

After you make changes in your Markdown file, follow the steps _Syncing notebooks_ below.

#### Syncing notebooks

After editing either the `.ipynb` or `.md` versions of the docs, sync the two versions
using [jupytext](https://jupytext.readthedocs.io/) by running `jupytext --sync` on the updated
notebooks.

First, make sure you have jupytext installed. The jupytext version should match
the one specified in [.pre-commit-config.yaml](https://github.com/google/flax/blob/main/.pre-commit-config.yaml)
(currently, it is v1.13.8).

```bash
pip install jupytext==1.13.8
```

Then, after you have made your changes in the Jupyter Notebook, sync the contents with its Markdown-equivalent
file by running the following command:

```bash
jupytext --sync path/to/the/file.ipynb
```

Similarly, to sync your Markdown file with its Jupyter Notebook version, run:

```bash
jupytext --sync path/to/the/file.md
```

Note that if you receive an error, and it is the first time you worked in a Jupyter Notebook, you may need
to (re)create a synced copy of the document (which is explained in detail in _Creating new notebooks_ section below):

```bash
jupytext --set-formats ipynb,md:myst path/to/the/notebook.ipynb
```

Once you're finished with syncing the `.md` and `.ipynb` files, you can check that they are properly synced using the
[pre-commit](https://pre-commit.com/) framework to perform the same checks used
in the Flax GitHub CI:

```bash
git add docs -u  # pre-commit runs on files in git staging.
pre-commit run jupytext
```

#### Creating new notebooks

If you are adding a new Jupyter Notebook to the documentation, you can use `jupytext --set-formats`.
It can set up both the Jupyter Notebook (`.ipynb`) and Markdown (`.md`) versions of the file:

```bash
jupytext --set-formats ipynb,md:myst path/to/the/notebook.ipynb
```

This works by adding a `"jupytext"` metadata field to the notebook file which specifies the
desired formats. The `jupytext --sync` command can then recognize them when invoked.

After you make changes in your file(s), follow the steps from the _Syncing notebooks_
section above to keep the contents of both Markdown and Jupyter Notebook files in sync.

#### Notebooks within the Sphinx build

Some of the notebooks are built automatically as part of the pre-submit checks and
as part of the [Read the Docs](https://flax.readthedocs.io/en/latest) build.
The build will fail if cells raise errors. If the errors are intentional, you can either catch them,
or tag the cell with `raises-exceptions` metadata ([example PR](https://github.com/google/jax/pull/2402/files)).
You have to add this metadata by hand in the `.ipynb` file. It will be preserved when somebody else
re-saves the notebook.

We exclude some notebooks from the build because, for example, they contain long computations.
See `exclude_patterns` in [`conf.py`](https://github.com/google/flax/blob/main/docs/conf.py).

### Updating the pull request contents

Every pull request should ideally be limited to just one commit, so if you have multiple commits please squash them.

Assuming you now have only one commit in your pull request, and want to add changes requested during review:

1. Make the changes locally in your editor.
2. Run `git commit -a --amend`. This updates the commit contents and allows you to edit the commit message.
3. At this point, `git push` alone will result in an error. Instead, use `git push --force`.
4. Check that it's done: The changes to your commit should be immediately reflected in the Github web UI.

## Troubleshooting

### Too many commits in a pull request

If your PR has too many commits associated with it (for example, more than five),
you need to squash them. Otherwise, the Flax docs build process may fail with an
error message. This is because of the following reasons:

* There are more than five commits in your pull request; and
* The Flax source sync process fails when the commit tree is too large.

To squash your commits, you can rebase your branch to `main` and create a new
commit containing all your changes, run the following command:

```bash
git rebase main && git reset --soft main && git commit
```

This will apply all your changes to the main branch. Note that if you had to
resolve any conflicts while working on your change (for instance, you did a
`pull upstream main` which led to conflict), then you will have to resolve these
conflicts again.

After you have successfully rebased your branch, you should push your changes.
And because you changed the commit history, you may have to use `git push --force`.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.
