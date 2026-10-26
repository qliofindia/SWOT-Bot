# API keys

SWOT-Bot never stores keys in the repository. Put them in a `.env` file at the
repo root. `.env` is listed in `.gitignore`, so it is never committed.

```
# .env  (create it yourself; never commit it)
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...                  # optional: higher Hugging Face download limits
AWS_ACCESS_KEY_ID=...            # optional: Hugging Face bucket via s3.hf.co
AWS_SECRET_ACCESS_KEY=...
MONGODB_URI=mongodb://<user>:<password>@localhost:27017   # optional: notebooks/mongodb_atlas.ipynb
```

Restrict the file: `chmod 600 .env` (Linux/macOS).

| Variable | Used for | Where to get it |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude API calls | https://console.anthropic.com → API Keys |
| `HF_TOKEN` | Downloading BERT / sentence encoders with higher rate limits | https://huggingface.co/settings/tokens |
| `MONGODB_URI` | Connection string for the historical MongoDB notebook | Your MongoDB server or Atlas cluster |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Reading the `sln2737/Brain` bucket through `https://s3.hf.co/sln2737` | Bucket settings on Hugging Face |

## Checking a key without printing it

```bash
set -a; . ./.env; set +a
curl -s -o /dev/null -w '%{http_code}\n' https://api.anthropic.com/v1/models \
  -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01"
# 200 = valid, 401 = invalid or revoked
```

## If a key is exposed

If a key ever lands in a commit, chat log or screenshot, revoke it in the
provider's console and create a new one. Deleting the commit does not
help, because clones and forks keep the history.
