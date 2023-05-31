def query():

    from google.cloud import bigquery

    client = bigquery.Client()
    #create and initiate a client object

    data_ref = client.dataset('chicago_crime', project='bigquery-public-data')
    #create a reference to the dataset

    data_set = client.getdataset(data_ref)
    #fetch the dataset

    # 📌
    tables = list(client.list_tables(data_set))
    for table in tables:
        print(table.table_id)
    # checking the number of tables in the dataset

    # Write the code to figure out the answer
    table = dataset_ref.table("crime")
    new_table = client.get_table(table)

    # new_table.schema
    new_table.schema

    schemas = new_table.schema
    count = 0
    for schem in schemas:
        if schem.field_type == 'TIMESTAMP':
            count += 1
            print(count)

    client.list_rows(new_table, max_results=5).to_dataframe()
    # to gain access to the data.


def more_concise():

    # Create a "Client" object
    client = bigquery.Client()

    # Construct a reference to the "openaq" dataset
    dataset_ref = client.dataset("openaq", project="bigquery-public-data")

    # API request - fetch the dataset
    dataset = client.get_dataset(dataset_ref)

    # List all the tables in the "openaq" dataset
    tables = list(client.list_tables(dataset))

    # Print names of all tables in the dataset (there's only one!)
    for table in tables:
        print(table.table_id)

    # and then you get the table.
    # got back to reference
    table_ref = dataset_ref.table("global_air_quality")

    # API request - fetch the table
    table = client.get_table(table_ref)
    client.list_rows(table, max_results=5).to_dataframe()


    #then you create the query
    query = """
            SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
            """

    #then you run the query
    query_job = client.query(query)

    #after getting the data. you can convert it to a dataframe
    us_cities = query_job.to_dataframe()

    #then you can do whatever you want with the dataframe


     #trying to acertain the run cost.
    query = """
            SELECT score, title
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" 
            """

    # Create a QueryJobConfig object to estimate size of query without running it
    dry_run_config = bigquery.QueryJobConfig(dry_run=True)

    # API request - dry run query to estimate costs
    dry_run_query_job = client.query(query, job_config=dry_run_config)

    print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))