# Repo Analytics

To run the repo analytics follow the steps below:

1. You must have a Github token, if you don't have one you can create one by following [this guide](https://docs.github.com/en/enterprise-server@3.4/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).
2. Install the requirements:

```bash
pip install -r .github/analytics/requirements.txt
```
3. Run the analytics:

```bash
GITHUB_TOKEN=<token> \
python .github/analytics/get_repo_metrics.py \
  --repo-owner google \
  --repo-name flax
```