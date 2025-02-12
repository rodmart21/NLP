import pandas as pd
from sqlalchemy import create_engine, text

class NounExtractor:
    def __init__(self, db_username, db_host, db_pass, db_name):
        """
        Initialize the NounExtractor class.

        Parameters:
            db_username (str): Database username.
            db_host (str): Database host.
            db_pass (str): Database password.
            db_name (str): Database name.
        """
        self.db_username = db_username
        self.db_host = db_host
        self.db_pass = db_pass
        self.db_name = db_name
        self.engine = self.create_connection()

    def create_connection(self):
        """Create a database connection using SQLAlchemy."""
        try:
            connection_string = f'postgresql://{self.db_username}:{self.db_pass}@{self.db_host}:5432/{self.db_name}'
            engine = create_engine(connection_string)
            print("Database connection successful.")
            return engine
        except Exception as e:
            print(f"Error creating database connection: {e}")
            return None

    def import_view(self, view_name):
        """
        Import data from a specified database view into a pandas DataFrame.

        Parameters:
            view_name (str): Schema-qualified name of the view.

        Returns:
            DataFrame: The imported data.
        """
        try:
            query = f"SELECT * FROM {view_name}"
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            print(f"Data imported successfully from view: {view_name}")
            return df
        except Exception as e:
            print(f"Error importing data from view: {e}")
            return None

    def process_nouns(self, df, category_column):
        """
        Process the extracted DataFrame to rename columns for noun matching.

        Parameters:
            df (DataFrame): The extracted DataFrame.
            category_column (str): The column name in the DataFrame to use for the category.

        Returns:
            DataFrame: Processed DataFrame with renamed columns.
        """
        try:
            column_mapping = {'product_id': 'Product Id', category_column: 'predicted_category'}
            df.rename(columns=column_mapping, inplace=True)
            print("Columns renamed successfully for noun matching.")
            return df[['Product Id', 'predicted_category']]
        except KeyError as e:
            print(f"Error renaming columns: {e}")
            return None

    def upload_to_database(self, df, table_name, schema='processing'):
        """
        Upload the processed DataFrame to the specified database table.

        Parameters:
            df (DataFrame): The DataFrame to upload.
            table_name (str): Name of the target database table.
            schema (str): Schema name where the table exists.
        """
        try:
            df.to_sql(table_name, self.engine, if_exists='append', index=False, schema=schema)
            print(f"Data uploaded successfully to {schema}.{table_name}")
        except Exception as e:
            print(f"Error uploading data to database: {e}")

