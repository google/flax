# How to Contribute

Everyone can contribute to Flax, and we value everyone's contributions. 
You can contribute in many more ways than just writing code. Anwering questions
on our [Discussions page](https://github.com/google/flax/discussions), helping
each other, and improving our documentation are extremely valuable to our
ecosystem.

We also appreciate if you spread the word, for instance by starring our Github
repo, referencing Flax in blog posts of projects that used it, or giving us a
Twitter shoutout by using the hashtag "#flax".

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).


## Ways to contribute

We welcome pull requests, in particular for those issues
[marked as PR-ready](https://github.com/google/flax/issues?q=is%3Aopen+is%3Aissue+label%3A%22Status%3A+pull+requests+welcome%22). For other proposals, we ask that you first open a Github Issue or
Github Discussion to discuss your planned contribution.

## Contributing code using Pull Requests

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Fork the Flax repository by clicking the 'Fork' button on the
   [repository page](http://www.github.com/google/flax). This create a copy
   of the Flax repository in your own account.

2. Install Python >=3.6 and `svn` for running the tests (see below).

3. (Optional) Create a virutal environment or a Docker container. See 
   [`dev/README.md`](https://github.com/google/flax/blob/master/dev/README.md)
   for details on how to setup a Docker Contaner. To setup a virual environment,
   run the following:

   ```bash
   python3.6 -m virtualenv env
   . env/bin/activate
   ```
  
   This ensures all your dependencies are installed in this environment.

4. `pip` installing your fork from source. This allows you to modify the code
   and immediately test it out:

   ```bash
   git clone https://github.com/YOUR_USERNAME/flax
   cd flax
   pip install -e . .[testing]
   ```

5. Add the Flax repo as an upstream remote, so you can use it to sync your
   changes.

   ```bash
   git add remote upstream http://www.github.com/google/flax
   ```


6. Create a branch where you will develop from:

   ```bash
   git checkout -b name-of-change
   ```

   And implement your changes using your favorite editor (we recomment
   [Visual Studio Code](https://code.visualstudio.com/).

7. Make sure the tests pass by running the following command from the top of
   the repository:

   ```bash
   ./tests/run_all_tests.sh
   ```

8. Once your change is done, create a commit as follows (
   [how to write a commit message](https://chris.beams.io/posts/git-commit/)):

   ```bash
   git add file1.py file2.py ...
   git commit -m "Your commit message"
   ```

   Then sync your code with the main repo:

   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

   Finally push your commit on your development branch and create a remote 
   branch in your fork that you can use to create a Pull Request from:

   ```bash
   git push --set-upstream origin name-of-change
   ```

9. Make sure your PR passes the 
   [PR checklist](https://github.com/google/flax/blob/master/.github/pull_request_template.md#checklist).
   If so, create a Pull Request from the Flax repository and send it for review.
   Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/)
   for more information on using pull requests.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.
