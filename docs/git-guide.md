# Your first Git commit and GitHub upload

**Git** records versions on your computer. A **commit** is a named snapshot.
**GitHub** hosts a copy online. Committing and uploading are separate actions.

## Start in the project folder

Open a terminal in the outer `silhouette-tools` folder: the one containing
`README.md`, `pyproject.toml`, and the inner `silhouette_tools` library folder.
This folder has its own Git repository, initialized with the branch name `main`.
No initial commit or GitHub upload was made during preparation.

Check the location Git is using:

```sh
git rev-parse --show-toplevel
```

The printed path should end in `silhouette-tools`. Then inspect the file list:

```sh
git status --short --untracked-files=all
```

The `.gitignore` file keeps environments, caches, and research data/results out
of this list. Ignoring a file leaves the local copy on your computer.

## Review and save the first snapshot

Read the README, license, citation, example, and code before committing.
When you are happy with the contents, select the files for the first snapshot:

```sh
git add .
git diff --cached --stat
```

`git add` **stages** files: it selects their current contents for the next
commit. The second command summarizes the selected files. For the full text
changes, use `git diff --cached`; press `q` to leave its viewer if it opens one.

If you edit a file after staging it, run `git add` again to include the edit.
If you decide a staged file should stay local before this first commit, use
`git rm --cached -- path/to/file` and add a suitable rule to `.gitignore`.
The `--cached` option removes it from the selection while keeping the file.

Save the selected snapshot locally:

```sh
git commit -m "Initial Silhouette Tools implementation"
git status
```

If Git asks for your identity, set `user.name` and `user.email` for this
repository, using your name and the commit email chosen in your GitHub account
settings. Then retry the commit. You do not need to change Git settings for
your other projects.

## Upload when you are ready

Create a new, empty repository named `silhouette-tools` on GitHub. Choose
private if you want to review it online before making it public. Leave GitHub's
options to add a README, license, and `.gitignore` unchecked: this project
already has those files.

Copy the repository URL shown by GitHub. After setting up GitHub authentication
for Git, run the following, replacing `REMOTE-URL` with that URL:

```sh
git remote add origin REMOTE-URL
git remote -v
git push -u origin main
```

`origin` is a nickname for the GitHub copy. `push` uploads your commits; `-u`
remembers the destination for subsequent pushes. These steps follow
[GitHub's guide to uploading local code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github),
which also links to authentication instructions and GitHub Desktop as an
alternative way to sign in and publish.

## Later changes

For a typical update, review the changes, select them, save a snapshot, and
upload it:

```sh
git diff
git add .
git diff --cached --stat
git commit -m "Describe what changed"
git push
```

GitHub hosting does not publish the package on PyPI. That can be a separate step
later; people can already install from a downloaded or cloned project folder.
