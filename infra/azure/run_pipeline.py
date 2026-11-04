"""Run the whole pipeline on Azure Databricks from the command line (route C in docs/azure.md).

Needs the Azure CLI (signed in with `az login`) and Terraform. No Databricks token: the script
uses your Azure sign-in to call the Databricks REST API.

    python infra/azure/run_pipeline.py up      # Terraform, then Unity Catalog storage, Git folder and job
    python infra/azure/run_pipeline.py run     # run the job and wait for it
    python infra/azure/run_pipeline.py status  # what exists, and the last run
    python infra/azure/run_pipeline.py down    # remove the Unity Catalog objects, then terraform destroy

State (names and URLs, no secrets) is kept in infra/azure/.run_state.json.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
STATE = HERE / ".run_state.json"
REPO_URL = "https://github.com/Shreya-Sebastian/online-retail-purchase-prediction"
DATABRICKS_APP = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d"  # Azure Databricks' fixed application id
SCHEMA = "custprop"
NOTEBOOKS = [("ingest", "01_ingest", []), ("snapshots", "02_snapshots", ["ingest"]),
             ("segments", "03_segments", ["snapshots"]), ("propensity", "04_propensity", ["segments"]),
             ("recommend", "05_recommend", ["segments"]), ("score", "06_score", ["propensity"]), ("figures", "07_figures", ["segments"])]


def sh(*args: str, capture: bool = True) -> str:
    exe = shutil.which(args[0]) or args[0]
    r = subprocess.run([exe, *args[1:]], cwd=HERE, text=True, capture_output=capture)
    if r.returncode != 0:
        sys.exit(f"{' '.join(args[:3])} failed:\n{r.stderr if capture else ''}")
    return (r.stdout or "").strip()


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=2))


class Api:
    def __init__(self, host: str, workspace_id: str = ""):
        self.host = host.rstrip("/")
        self.workspace_id = workspace_id  # Azure resource id: lets a first-time user in as admin
        self.token = sh("az", "account", "get-access-token", "--resource", DATABRICKS_APP,
                        "--query", "accessToken", "-o", "tsv")

    def call(self, method: str, path: str, body: dict | None = None, ok404: bool = False):
        req = urllib.request.Request(self.host + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/json",
                                              **({"X-Databricks-Azure-Workspace-Resource-Id": self.workspace_id}
                                                 if self.workspace_id else {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if ok404 and e.code == 404:
                return None
            raise RuntimeError(f"{method} {path} -> {e.code}: {msg[:400]}") from None


def tf_vars(s: dict) -> list[str]:
    return [f"-var=subscription_id={s['subscription_id']}", f"-var=storage_account_name={s['storage_account']}",
            f"-var=location={s.get('location', 'westeurope')}"]


def up() -> None:
    s = load()
    s.setdefault("subscription_id", sh("az", "account", "show", "--query", "id", "-o", "tsv"))
    s.setdefault("storage_account", f"custprop{random.randint(10000, 99999)}")
    # Some regions turn away new (free) subscriptions; pass another region as a second argument.
    if len(sys.argv) > 2:
        s["location"] = sys.argv[2]
    save(s)
    print("Terraform: creating the resource group, storage, workspace and access connector ...")
    sh("terraform", "init", "-input=false", capture=False)
    sh("terraform", "apply", "-input=false", "-auto-approve", *tf_vars(s), capture=False)
    out = json.loads(sh("terraform", "output", "-json"))
    s.update(host=out["workspace_url"]["value"], connector_id=out["access_connector_id"]["value"],
             raw_url=out["raw_container_url"]["value"],
             workspace_id=f"/subscriptions/{s['subscription_id']}/resourceGroups/rg-custprop"
                          "/providers/Microsoft.Databricks/workspaces/dbw-custprop")
    save(s)

    api = Api(s["host"], s.get("workspace_id", ""))
    for attempt in range(20):  # a new workspace takes a few minutes before its API answers
        try:
            me = api.call("GET", "/api/2.0/preview/scim/v2/Me")
            break
        except RuntimeError as e:
            if attempt == 19:
                raise
            print("workspace not reachable yet, retrying in 30 s:", str(e)[:100], flush=True)
            time.sleep(30)
    s["user"] = me["userName"]
    catalogs = [c["name"] for c in api.call("GET", "/api/2.1/unity-catalog/catalogs").get("catalogs", [])]
    s["catalog"] = next((c for c in catalogs if c not in ("system", "samples", "hive_metastore", "main")),
                        "main" if "main" in catalogs else None)
    if not s["catalog"]:
        sys.exit(f"No usable Unity Catalog catalog found (saw {catalogs}).")
    print("catalog:", s["catalog"])
    save(s)

    # External storage: credential -> external location -> schema -> external volume.
    s["external"] = False
    try:
        if not api.call("GET", "/api/2.1/unity-catalog/storage-credentials/custprop_lake", ok404=True):
            api.call("POST", "/api/2.1/unity-catalog/storage-credentials",
                     {"name": "custprop_lake", "azure_managed_identity": {"access_connector_id": s["connector_id"]}})
        for attempt in range(10):  # the role assignment can take a few minutes to apply
            try:
                if not api.call("GET", "/api/2.1/unity-catalog/external-locations/custprop_raw", ok404=True):
                    api.call("POST", "/api/2.1/unity-catalog/external-locations",
                             {"name": "custprop_raw", "url": s["raw_url"], "credential_name": "custprop_lake"})
                break
            except RuntimeError as e:
                if attempt == 9:
                    raise
                print("external location not ready yet, retrying in 30 s:", str(e)[:120])
                time.sleep(30)
        s["external"] = True
    except RuntimeError as e:
        print("Could not set up external storage; the notebook will use a managed volume instead:\n ", e)
    save(s)

    if not api.call("GET", f"/api/2.1/unity-catalog/schemas/{s['catalog']}.{SCHEMA}", ok404=True):
        api.call("POST", "/api/2.1/unity-catalog/schemas", {"name": SCHEMA, "catalog_name": s["catalog"]})
    if s["external"] and not api.call("GET", f"/api/2.1/unity-catalog/volumes/{s['catalog']}.{SCHEMA}.raw", ok404=True):
        api.call("POST", "/api/2.1/unity-catalog/volumes",
                 {"catalog_name": s["catalog"], "schema_name": SCHEMA, "name": "raw",
                  "volume_type": "EXTERNAL", "storage_location": s["raw_url"] + "online_retail"})
        print("external volume created in your storage account")

    # Code: a Git folder with the public repository.
    s["repo_path"] = f"/Repos/{s['user']}/online-retail-purchase-prediction"
    if not api.call("GET", "/api/2.0/workspace/get-status?path=" + s["repo_path"], ok404=True):
        api.call("POST", "/api/2.0/repos", {"url": REPO_URL, "provider": "gitHub", "path": s["repo_path"]})
    save(s)

    # The job: the same six tasks as databricks.yml, on serverless compute.
    job = {"name": "custprop-pipeline", "max_concurrent_runs": 1,
           "parameters": [{"name": "catalog", "default": s["catalog"]}, {"name": "schema", "default": SCHEMA}],
           "tasks": [{"task_key": k, "depends_on": [{"task_key": d} for d in deps],
                      "notebook_task": {"notebook_path": f"{s['repo_path']}/notebooks/{nb}", "source": "WORKSPACE"}}
                     for k, nb, deps in NOTEBOOKS]}
    if s.get("job_id"):
        api.call("POST", "/api/2.1/jobs/reset", {"job_id": s["job_id"], "new_settings": job})
    else:
        s["job_id"] = api.call("POST", "/api/2.1/jobs/create", job)["job_id"]
    save(s)
    print(f"ready: {s['host']}  job {s['job_id']}")


def run(attach: bool = False) -> None:
    s = load()
    api = Api(s["host"], s.get("workspace_id", ""))
    if attach:  # follow the run already started, e.g. after the watcher stopped
        run_id = s["run_id"]
    else:
        run_id = api.call("POST", "/api/2.1/jobs/run-now", {"job_id": s["job_id"]})["run_id"]
        s["run_id"] = run_id
        save(s)
        print("run", run_id, "started")
    while True:
        api = Api(s["host"], s.get("workspace_id", ""))  # fresh token: Azure tokens last about an hour
        r = api.call("GET", f"/api/2.1/jobs/runs/get?run_id={run_id}")
        states = {t["task_key"]: t["state"].get("result_state") or t["state"]["life_cycle_state"] for t in r["tasks"]}
        print(time.strftime("%H:%M:%S"), states, flush=True)
        if r["state"]["life_cycle_state"] in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        time.sleep(60)
    report(api, r)


def report(api: Api, r: dict) -> None:
    print("\nresult:", r["state"].get("result_state"), "-", r["state"].get("state_message", ""))
    for t in r["tasks"]:
        secs = (t.get("end_time", 0) - t.get("start_time", 0)) / 1000
        line = f"  {t['task_key']:<11} {t['state'].get('result_state', t['state']['life_cycle_state']):<10} {secs / 60:5.1f} min"
        if t["state"].get("result_state") not in (None, "SUCCESS"):
            out = api.call("GET", f"/api/2.1/jobs/runs/get-output?run_id={t['run_id']}")
            line += "\n    " + (out.get("error") or "")[:300]
        print(line)


def status() -> None:
    s = load()
    print(json.dumps({k: v for k, v in s.items()}, indent=2))
    if s.get("run_id"):
        api = Api(s["host"], s.get("workspace_id", ""))
        report(api, api.call("GET", f"/api/2.1/jobs/runs/get?run_id={s['run_id']}"))
        if s.get("external"):
            ls = api.call("GET", f"/api/2.0/fs/directories/Volumes/{s['catalog']}/{SCHEMA}/raw", ok404=True) or {}
            print("files in the external volume:", [f["name"] for f in ls.get("contents", [])])


def down() -> None:
    s = load()
    if s.get("host"):
        api = Api(s["host"], s.get("workspace_id", ""))
        for path in (f"/api/2.1/unity-catalog/volumes/{s.get('catalog')}.{SCHEMA}.raw",
                     "/api/2.1/unity-catalog/external-locations/custprop_raw?force=true",
                     "/api/2.1/unity-catalog/storage-credentials/custprop_lake?force=true"):
            try:
                api.call("DELETE", path, ok404=True)
            except RuntimeError as e:
                print("skipped:", e)
    sh("terraform", "destroy", "-input=false", "-auto-approve", *tf_vars(s), capture=False)
    STATE.unlink(missing_ok=True)
    print("everything removed")


if __name__ == "__main__":
    {"up": up, "run": run, "wait": lambda: run(attach=True), "status": status, "down": down}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
