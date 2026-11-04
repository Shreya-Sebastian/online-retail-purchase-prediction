# Running the pipeline on Azure Databricks

The notebooks need no code changes: they already take the Unity Catalog catalog and schema as
parameters and keep the raw file in a Unity Catalog volume. What changes on Azure is the setup
around them: your own Data Lake Storage account, a managed identity that Unity Catalog uses to reach
it without keys, and compute you choose yourself.

Time: about 1.5 to 2 hours. Cost: a few euros of the free Azure credit if you stop compute promptly.

## 0. Before you start

1. **Budget alert:** Portal > Cost Management > Budgets > Add, scope = your subscription, amount EUR 10,
   alert at 50% and 100% to your email.
2. **Spending limit:** a free account normally has one switched on, so it cannot charge beyond the free
   credit. Check it under Subscriptions > your subscription > Overview.
3. Use one region for everything: **West Europe**.
4. Everything goes into **one resource group**, so one delete cleans it all up (step 9).

## 1. Create the Azure resources

Choose **route A** (portal, no installs) or **route B** (Terraform). Both create the same four things.

### Route A: in the portal

1. **Resource group:** Resource groups > Create > name `rg-custprop`, region West Europe.
2. **Storage account (Data Lake Storage Gen2):** Storage accounts > Create
   - resource group `rg-custprop`, a globally unique name (lowercase letters and digits, e.g. `custprop<initials>01`),
     region West Europe, performance Standard, redundancy LRS;
   - tab **Advanced**: switch on **Enable hierarchical namespace** (this is what makes it Data Lake Storage Gen2);
   - after it is created: Data storage > Containers > + Container > name `raw`, access level Private.
3. **Azure Databricks workspace:** Azure Databricks > Create > resource group `rg-custprop`, name `dbw-custprop`,
   region West Europe, pricing tier **Premium** (or the 14-day Premium trial if offered; Premium is needed for
   Unity Catalog). Leave networking at the defaults.
4. **Access connector:** search "Access Connector for Azure Databricks" > Create > resource group `rg-custprop`,
   name `dbw-custprop-connector`. It gets a system-assigned managed identity automatically.
5. **Give the connector access to the storage account:** open the storage account > Access control (IAM) >
   Add role assignment > role **Storage Blob Data Contributor** > Assign access to: Managed identity >
   select the access connector > Review + assign.

### Route B: Terraform (`infra/azure`)

Needs the Azure CLI (`az login`) and Terraform.

```bash
cd infra/azure
az login
terraform init
terraform apply -var="subscription_id=<your subscription id>" -var="storage_account_name=<unique name>"
```

The outputs give the workspace URL, the access connector id and the container URL for step 3.

## 2. Open the workspace and find your catalog

Open the Databricks workspace (Launch Workspace). New workspaces usually have Unity Catalog enabled
automatically, with a **workspace catalog** named after the workspace. Open **Catalog** and note that
catalog's name: you pass it to the notebooks instead of `workspace`.

## 3. Connect Unity Catalog to your storage account

In **Catalog > External data**:

1. **Credentials > Create credential:** type Azure Managed Identity, name `custprop_lake`, access connector id =
   the connector's Resource ID (portal > the connector > Properties, or the Terraform output).
2. **External locations > Create:** name `custprop_raw`, URL
   `abfss://raw@<storage account>.dfs.core.windows.net/`, credential `custprop_lake`. Use **Test connection**.
3. In a SQL editor or notebook cell (replace `<catalog>` and `<storage account>`):

```sql
CREATE SCHEMA IF NOT EXISTS <catalog>.custprop;
CREATE EXTERNAL VOLUME IF NOT EXISTS <catalog>.custprop.raw
  LOCATION 'abfss://raw@<storage account>.dfs.core.windows.net/online_retail';
```

Notebook 01 runs `CREATE VOLUME IF NOT EXISTS ... raw`, finds your external volume and uses it, so the raw
file lands in **your** storage account.

If you are not allowed to create credentials or external locations, skip this step: notebook 01 then creates
a managed volume in the workspace's own storage. It still runs on Azure, but the data isn't in your account.

## 4. Get the code

Workspace > Create > **Git folder** > URL `https://github.com/Shreya-Sebastian/online-retail-purchase-prediction`.
The notebooks import the package from `../src`, so keep the repository layout as it is.
If the repository is private, first link GitHub under your user settings > Linked accounts.

## 5. Compute

- If **serverless** compute is offered for notebooks, use it (that is what the Free Edition used).
- Otherwise: Compute > Create compute > **Single node**, the smallest node offered (about 4 cores), a recent
  **ML** runtime (LTS), and **terminate after 15 minutes of inactivity**. Free subscriptions limit the number
  of cores per region, so a small single node is the safe choice.

## 6. Run the pipeline

1. Open `notebooks/01_ingest`, attach the compute, set the widget **catalog** to your catalog name (schema
   stays `custprop`) and run all. It downloads Online Retail II into the volume and writes `raw_transactions`
   and `transactions`.
2. Check in the portal: storage account > Containers > `raw` > `online_retail` should now contain
   `online_retail_II.xlsx`. That shows the data really lives in your Data Lake Storage.
3. Run 02 to 06 in order with the same catalog, or create a job with the six notebooks as tasks
   (as in `databricks.yml`) and pass `catalog` as a job parameter.

Optional, with the Databricks CLI: set the `azure` target's host in `databricks.yml`, then

```bash
databricks bundle deploy -t azure --var catalog=<catalog>
databricks bundle run custprop_pipeline -t azure --var catalog=<catalog>
```

## 7. What was different from the Free Edition

Write down what you actually ran into; typical differences are:

- the catalog name (the workspace catalog instead of `workspace`);
- storage access through a managed identity, a storage credential and an external location, so no keys in code;
- choosing compute and its auto-termination yourself, within the subscription's core limits;
- costs appearing in Azure Cost Management.

## 8. Record what you did

Add one line per step you completed to the README's "Running on Azure" section, and keep one or two
screenshots (the external location test, the file in the container, a finished run). Only claim the steps
you actually completed.

## 9. Clean up

Terminate the compute. When you are done, delete the resource group `rg-custprop` (Resource groups >
`rg-custprop` > Delete resource group); the workspace's managed resource group goes with it.
With Terraform: `terraform destroy` with the same variables.
