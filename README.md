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
7. Under **Machine type** choose n2-standard-32
   - `n2-standard-32`
7. Select the image URI:
```
   us-east1-docker.pkg.dev/sjcabs/notebook-images/graph-tool:latest
```
8. Click **Create**
9. Wait 10-15 minutes for provisioning
10. Once status shows **Running**, click **Open JupyterLab**

### Using the Environment

1. In the folder panel on the left, you should see sjcabs-graph-tutorial/graph_tutorial.ipynb. Open it.
2. Select the **"Python 3.10 (graph-tool)"** kernel

### Saving Your Work

- **Save notebooks frequently** 
- Files are preserved when you stop the instance
- Variables in memory are lost when you stop — re-run cells after restarting

### Managing Costs

- **Stop** your instance when not in use (you still pay for disk storage)
- **Start** to resume work
- **Delete** when you no longer need it