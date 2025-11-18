# Installation Guide


## Python & Environment

### 1. Package Manager
We use modern [uv](https://github.com/astral-sh/uv) for managing python packages.

To install it run:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```
(it will save `uv` binary into the <span style="color:#6ee7b7; background:rgba(110, 231, 183, 0.05); padding: 2px 5px; border-radius: 6px;">~/.local/bin</span>)

Check the installation by running:
```
uv --version
```
(you should see something like <span style="color:#f2c94c; background:rgba(242, 201, 76, 0.05); padding: 2px 5px; border-radius: 6px;">uv 0.9.9</span>)

### 2. Python
We use `Python 3.12`. You can try other Python versions, but we recommend installing `Python 3.12`, as the project was tested with it. You can do it using:
```
uv python install 3.12
uv python list
```
(this installs `CPython 3.12` into `uv`’s own directory under <span style="color:#6ee7b7; background:rgba(110, 231, 183, 0.05); padding: 2px 5px; border-radius: 6px;">~/.local/share/uv/python/...</span>)

### 3. Environment
It is time to create and activate a virtual environment.

Move into the repo root:
```
cd <path-to-repository>/smart-search-engine
```

Run:
```
uv sync
```
(`uv` creates `.venv` directory)


### 4. Precommit (optional)
If you ever want to collaborate, run:
```
uv run pre-commit install
```


## Database

### 1. PostgreSQL

```
sudo apt install -y postgresql postgresql-contrib build-essential
```

Check the installation by running:
```
psql --version
```
(you should see something like <span style="color:#f2c94c; background:rgba(242, 201, 76, 0.05); padding: 2px 5px; border-radius: 6px;">psql (PostgreSQL) 16.10 (Ubuntu 16.10-0ubuntu0.24.04.1)</span>)

Start service:
```
sudo systemctl enable postgresql
sudo systemctl start postgresql
```

### 2. pgvector

It is required to install development headers to install `pgvector` (or there will be an error about `postgres.h` which is missing):
```
sudo apt install postgresql-server-dev-<N>
```
(where $N$ is the `psql` version you can find by running `psql --version`)

```
cd /tmp
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

If everything is alright, you can clear the tmp:
```
rm -rf /tmp/pgvector
```

### 3. Database & Users Creation

Enter Postgres as the default `postgres` superuser:
```
sudo -u postgres psql
```

Then, in the `psql` shell run:
```
CREATE ROLE telegram_smart_search_user WITH LOGIN PASSWORD '<password>';
CREATE DATABASE telegram_smart_search_database OWNER telegram_smart_search_user;
GRANT ALL PRIVILEGES ON DATABASE telegram_smart_search_database TO telegram_smart_search_user;
```

Enable `pgvector`:
```
CREATE EXTENSION IF NOT EXISTS vector;
```

Exit:
```
\q
```

To create tables, run:
```
psql -h localhost -U telegram_smart_search_user -d telegram_smart_search_database -f database/schema.sql
```

If you would ever need to connect as your app user (optional):
```
psql -h localhost -U telegram_smart_search_user -d telegram_smart_search_database
# enter <password>

# DO

\q
```


## Client & Web Config

### 1. Linux Packages Installation

```
sudo apt install -y redis-server
```

```
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

Check that `redis` is working with:
```
redis-cli ping  # PONG
```

### 2. Django Migrations (Optional)
```
uv run python manage.py migrate
```

### 3. Running
```
uv run python3 manage.py runserver 0.0.0.0:8000
```