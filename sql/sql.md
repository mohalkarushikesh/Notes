Below is an in-depth set of SQL notes tailored to provide a comprehensive understanding of Structured Query Language (SQL). These notes cover foundational concepts, key commands, and practical examples. SQL is used to manage and manipulate relational databases, so I’ll structure this to progress logically from basics to more advanced topics.

---

### **1. Introduction to SQL**
- **What is SQL?**
  - SQL stands for Structured Query Language.
  - It’s a standard language for interacting with relational database management systems (RDBMS) like MySQL, PostgreSQL, SQLite, Oracle, and SQL Server.
  - Used for tasks like querying data, updating records, and managing database structures.

- **Key Features:**
  - Declarative: You specify *what* you want, not *how* to get it (the RDBMS optimizes execution).
  - Case-insensitive for commands (e.g., `SELECT` = `select`), though data itself is case-sensitive depending on the system.
  - Works with tables (entities), columns (attributes), and rows (records).

- **Categories of SQL Commands:**
  1. **DDL (Data Definition Language)**: Define database structure (e.g., `CREATE`, `ALTER`, `DROP`).
  2. **DML (Data Manipulation Language)**: Manipulate data (e.g., `SELECT`, `INSERT`, `UPDATE`, `DELETE`).
  3. **DCL (Data Control Language)**: Control access (e.g., `GRANT`, `REVOKE`).
  4. **TCL (Transaction Control Language)**: Manage transactions (e.g., `COMMIT`, `ROLLBACK`).

---

### **2. Basic SQL Syntax**
- **General Structure of a Query:**
  ```sql
  SELECT column1, column2
  FROM table_name
  WHERE condition
  ORDER BY column_name;
  ```
  - Clauses are executed in this order: `FROM` → `WHERE` → `SELECT` → `ORDER BY`.

- **Comments:**
  - Single-line: `-- This is a comment`
  - Multi-line: `/* This is a multi-line comment */`

---

### **3. Core SQL Commands**

#### **3.1 SELECT (Retrieve Data)**
- **Purpose**: Query data from one or more tables.
- **Syntax**:
  ```sql
  SELECT column1, column2
  FROM table_name
  WHERE condition;
  ```
- **Examples**:
  - Select all columns: 
    ```sql
    SELECT * FROM employees;
    ```
  - Select specific columns:
    ```sql
    SELECT first_name, last_name FROM employees;
    ```
  - Distinct values (remove duplicates):
    ```sql
    SELECT DISTINCT department FROM employees;
    ```

- **Filtering with WHERE**:
  - Conditions: `=`, `<>`, `>`, `<`, `>=`, `<=`, `LIKE`, `IN`, `BETWEEN`.
  - Example:
    ```sql
    SELECT * FROM employees
    WHERE salary > 50000;
    ```
  - Using `LIKE` for patterns:
    ```sql
    SELECT first_name FROM employees
    WHERE first_name LIKE 'A%'; -- Names starting with 'A'
    ```

- **Sorting with ORDER BY**:
  - Ascending (default): `ORDER BY column_name ASC`
  - Descending: `ORDER BY column_name DESC`
  - Example:
    ```sql
    SELECT first_name, salary FROM employees
    ORDER BY salary DESC;
    ```

#### **3.2 INSERT (Add Data)**
- **Purpose**: Add new rows to a table.
- **Syntax**:
  ```sql
  INSERT INTO table_name (column1, column2)
  VALUES (value1, value2);
  ```
- **Example**:
  ```sql
  INSERT INTO employees (first_name, last_name, salary)
  VALUES ('John', 'Doe', 60000);
  ```
- **Insertmultiple rows**:
  ```sql
  INSERT INTO employees (first_name, last_name, salary)
  VALUES ('Jane', 'Smith', 55000), ('Mike', 'Brown', 70000);
  ```

#### **3.3 UPDATE (Modify Data)**
- **Purpose**: Update existing rows.
- **Syntax**:
  ```sql
  UPDATE table_name
  SET column1 = value1
  WHERE condition;
  ```
- **Example**:
  ```sql
  UPDATE employees
  SET salary = 65000
  WHERE first_name = 'John' AND last_name = 'Doe';
  ```
- **Caution**: Without `WHERE`, all rows are updated!

#### **3.4 DELETE (Remove Data)**
- **Purpose**: Delete rows from a table.
- **Syntax**:
  ```sql
  DELETE FROM table_name
  WHERE condition;
  ```
- **Example**:
  ```sql
  DELETE FROM employees
  WHERE salary < 40000;
  ```
- **Delete all rows**: 
  ```sql
  DELETE FROM employees; -- No WHERE clause
  ```

#### **3.5 CREATE (Define Structure)**
- **Purpose**: Create databases or tables.
- **Syntax**:
  ```sql
  CREATE TABLE table_name (
      column1 datatype constraints,
      column2 datatype constraints
  );
  ```
- **Example**:
  ```sql
  CREATE TABLE employees (
      id INT PRIMARY KEY,
      first_name VARCHAR(50) NOT NULL,
      last_name VARCHAR(50) NOT NULL,
      salary DECIMAL(10, 2),
      hire_date DATE
  );
  ```
- **Common Data Types**:
  - `INT` / `INTEGER`: Whole numbers.
  - `VARCHAR(n)`: Variable-length string (n = max length).
  - `DECIMAL(p, s)`: Fixed-point number (p = precision, s = scale).
  - `DATE`: Date (e.g., '2025-02-28').
  - `BOOLEAN`: True/False.

- **Constraints**:
  - `PRIMARY KEY`: Unique identifier for each row.
  - `FOREIGN KEY`: Links to another table’s primary key.
  - `NOT NULL`: Ensures a value is provided.
  - `UNIQUE`: Ensures all values are distinct.
  - `DEFAULT`: Sets a default value.

#### **3.6 ALTER (Modify Structure)**
- **Purpose**: Change table structure.
- **Syntax**:
  ```sql
  ALTER TABLE table_name
  ADD column_name datatype;
  ```
- **Examples**:
  - Add column:
    ```sql
    ALTER TABLE employees
    ADD email VARCHAR(100);
    ```
  - Modify column:
    ```sql
    ALTER TABLE employees
    MODIFY COLUMN salary DECIMAL(12, 2);
    ```
  - Drop column:
    ```sql
    ALTER TABLE employees
    DROP COLUMN email;
    ```

#### **3.7 DROP (Remove Structure)**
- **Purpose**: Delete tables or databases.
- **Syntax**:
  ```sql
  DROP TABLE table_name;
  ```
- **Example**:
  ```sql
  DROP TABLE employees;
  ```

---

### **4. Advanced SQL Concepts**

#### **4.1 Joins**
- **Purpose**: Combine data from multiple tables based on related columns.
- **Types of Joins**:
  1. **INNER JOIN**: Only matching rows from both tables.
     ```sql
     SELECT e.first_name, d.department_name
     FROM employees e
     INNER JOIN departments d
     ON e.department_id = d.id;
     ```
  2. **LEFT (OUTER) JOIN**: All rows from left table, NULLs if no match in right table.
     ```sql
     SELECT e.first_name, d.department_name
     FROM employees e
     LEFT JOIN departments d
     ON e.department_id = d.id;
     ```
  3. **RIGHT (OUTER) JOIN**: All rows from right table, NULLs if no match in left table.
  4. **FULL (OUTER) JOIN**: All rows from both tables, NULLs where no match.

- **Alias**: Shorten table names (e.g., `employees e`).

#### **4.2 Aggregate Functions**
- **Purpose**: Perform calculations on multiple rows.
- **Common Functions**:
  - `COUNT()`: Number of rows.
  - `SUM()`: Total of numeric values.
  - `AVG()`: Average of numeric values.
  - `MIN()` / `MAX()`: Minimum/Maximum value.
- **Example**:
  ```sql
  SELECT department_id, AVG(salary)
  FROM employees
  GROUP BY department_id;
  ```

- **GROUP BY**: Groups rows with identical values.
- **HAVING**: Filters groups (like `WHERE` for aggregates).
  ```sql
  SELECT department_id, COUNT(*) as emp_count
  FROM employees
  GROUP BY department_id
  HAVING COUNT(*) > 5;
  ```

#### **4.3 Subqueries**
- **Purpose**: Nest a query inside another query.
- **Example**:
  ```sql
  SELECT first_name, salary
  FROM employees
  WHERE salary > (SELECT AVG(salary) FROM employees);
  ```
- **Types**:
  - Single-row subquery: Returns one value (e.g., `>` comparison).
  - Multi-row subquery: Returns multiple values (e.g., `IN`).

#### **4.4 Transactions**
- **Purpose**: Ensure data integrity for multiple operations.
- **Commands**:
  - `BEGIN` / `START TRANSACTION`: Start a transaction.
  - `COMMIT`: Save changes.
  - `ROLLBACK`: Undo changes.
- **Example**:
  ```sql
  BEGIN;
  UPDATE employees SET salary = salary + 1000 WHERE id = 1;
  DELETE FROM employees WHERE id = 2;
  COMMIT;
  ```
  - If an error occurs, use `ROLLBACK` to revert.

#### **4.5 Indexes**
- **Purpose**: Improve query performance.
- **Syntax**:
  ```sql
  CREATE INDEX idx_name ON table_name (column_name);
  ```
- **Example**:
  ```sql
  CREATE INDEX idx_salary ON employees (salary);
  ```
- **Drop Index**:
  ```sql
  DROP INDEX idx_salary;
  ```

---

### **5. Best Practices**
- **Naming Conventions**: Use descriptive, consistent names (e.g., `employee_id` not `eid`).
- **Normalization**: Design tables to reduce redundancy (1NF, 2NF, 3NF).
- **Use Comments**: Document complex queries.
- **Test Queries**: Run on small datasets first to avoid unintended updates/deletes.
- **Backup**: Always back up databases before major changes.

---

### **6. Example Database Schema**
Let’s tie it together with a sample schema:
```sql
CREATE TABLE departments (
    id INT PRIMARY KEY,
    department_name VARCHAR(50) NOT NULL
);

CREATE TABLE employees (
    id INT PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    department_id INT,
    salary DECIMAL(10, 2),
    FOREIGN KEY (department_id) REFERENCES departments(id)
);

INSERT INTO departments (id, department_name)
VALUES (1, 'HR'), (2, 'Engineering');

INSERT INTO employees (id, first_name, last_name, department_id, salary)
VALUES (1, 'Alice', 'Smith', 1, 60000),
       (2, 'Bob', 'Jones', 2, 75000);

SELECT e.first_name, e.salary, d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.id
WHERE e.salary > 65000
ORDER BY e.salary DESC;
```
**Output**:
```
first_name | salary  | department_name
-----------+---------+----------------
Bob        | 75000   | Engineering
```

---

These notes cover the essentials and some advanced topics to get you started with SQL. Let me know if you’d like deeper dives into specific areas like optimization, stored procedures, or triggers!
