# Graph Analysis Tutorial 

## Environment Setup
### Creating Your Workbench Instance

Follow these steps to create your own notebook environment:

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to **Vertex AI** → **Workbench** → **Instances**
3. Click **Create New**
4. Configure the instance:
   - **Name**: `YOURNAME-graph` 
   - **Region**: `us-east1`
   - **Zone**: `us-east1-b`
   - Uncheck 'Enable Apache Spark and BigQuery Kernels'
5. Click **Advanced options**
6. Under **Environment**, select **Use custom container**
   - For 'Docker container image', choose 'select' and choose 'graph-tool' and select the one with 'latest' tag from the drop-down. 
7. Under **Machine type** n2-standard-96
   - Choose `n2-standard-96`
   - 'Enable Idle Shutdown' = 30
8. Click **Create**
9. Wait 10-15 minutes for provisioning
10. Once status shows **Running**, click **Open JupyterLab**
11. Once in the jupyter lab. Click the github tab on the left (3rd tab) -> clone a repository -> paste this repo link there. 
12. Now navigate inside `/graph_analysis` and we are all set!

### Using the Environment

Select the **"Python 3.10 (graph-tool)"** kernel

### Saving Your Work

- **Save notebooks frequently** 
- Files are preserved when you stop the instance
- Variables in memory are lost when you stop — re-run cells after restarting

### Managing Costs

- **Stop** your instance when not in use (you still pay for disk storage)
- **Start** to resume work
- **Delete** when you no longer need it