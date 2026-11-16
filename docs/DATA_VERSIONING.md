# Data and experiment versioning

The project contains an optional DVC stage for regenerating the deterministic synthetic validation results. DVC is not required to run the app.

```powershell
pip install -e ".[data-versioning]"
dvc init
dvc repro
```

For actual participant data, configure a private, access-controlled DVC remote appropriate to your organization's governance requirements. Do not put identifiable recordings, raw sensor files, personal identifiers or consent documents in a public repository. The application ZIP export contains hashes and processing metadata but intentionally omits raw source and sensor recordings.

Example local remote (replace with an approved storage location):

```powershell
dvc remote add -d research-storage ../thermal-data-cache
```

Do not use a local path in a collaborative deployment unless all collaborators share the same storage mount. For cloud remotes, follow the DVC backend guide and use environment variables/secret managers for credentials.

The `dvc.yaml` stage named `synthetic_validation` regenerates the bundled two-sequence software-validation examples. These examples test the code path and do not represent human physiological data.
