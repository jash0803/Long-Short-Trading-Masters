import mysql.connector

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="your_database"
)
cursor = conn.cursor()

# Run structure.sql
with open('structure.sql', 'r') as f:
    structure_sql = f.read()
cursor.execute(structure_sql, multi=True)

# Run data.sql
with open('data.sql', 'r') as f:
    data_sql = f.read()
cursor.execute(data_sql, multi=True)

conn.commit()
cursor.close()
conn.close()
