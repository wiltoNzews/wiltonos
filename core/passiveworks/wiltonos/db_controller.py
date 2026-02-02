"""
WiltonOS DB Controller
Centralized database access layer with integrated secrets management
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from contextlib import contextmanager
import datetime
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [DB_CONTROLLER] %(message)s",
    handlers=[
        logging.FileHandler("logs/db_controller.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("db_controller")

# Thread-local storage for connection management
local = threading.local()

class DBController:
    """
    Centralized database controller for WiltonOS
    
    Provides secure access to the PostgreSQL database with proper connection
    management, schema handling, and integrated secrets.
    """
    
    def __init__(self):
        # Database connection parameters from environment
        self.db_params = {
            'dbname': os.getenv('PGDATABASE'),
            'user': os.getenv('PGUSER'),
            'password': os.getenv('PGPASSWORD'),
            'host': os.getenv('PGHOST'),
            'port': os.getenv('PGPORT')
        }
        
        # Verify database connection parameters
        missing_params = [k for k, v in self.db_params.items() if v is None]
        if missing_params:
            logger.warning(f"Missing database connection parameters: {', '.join(missing_params)}")
        
        # Initialize connection pool
        self.connection_pool = None
        
        # Table schemas
        self.schemas = {
            'users': """
                CREATE TABLE IF NOT EXISTS users (
                    id VARCHAR(50) PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE,
                    first_name VARCHAR(100),
                    last_name VARCHAR(100),
                    bio TEXT,
                    profile_image_url VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            'sessions': """
                CREATE TABLE IF NOT EXISTS sessions (
                    sid VARCHAR(255) PRIMARY KEY,
                    sess JSON NOT NULL,
                    expire TIMESTAMP(6) NOT NULL
                )
            """,
            'cognitive_resonance_logs': """
                CREATE TABLE IF NOT EXISTS cognitive_resonance_logs (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) REFERENCES users(id),
                    memory_waves FLOAT NOT NULL,
                    emotional_viscosity FLOAT NOT NULL,
                    perturbation FLOAT NOT NULL,
                    cognitive_resonance FLOAT NOT NULL,
                    note TEXT,
                    source VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            'fractal_visualizations': """
                CREATE TABLE IF NOT EXISTS fractal_visualizations (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) REFERENCES users(id),
                    type VARCHAR(50) NOT NULL,
                    parameters JSON NOT NULL,
                    image_url VARCHAR(255),
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            'agent_state': """
                CREATE TABLE IF NOT EXISTS agent_state (
                    agent_id VARCHAR(50) PRIMARY KEY,
                    agent_type VARCHAR(50) NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    description TEXT,
                    state JSON NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    last_active TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            'discover_logs': """
                CREATE TABLE IF NOT EXISTS discover_logs (
                    id SERIAL PRIMARY KEY,
                    scan_type VARCHAR(50) NOT NULL,
                    findings JSON NOT NULL,
                    recommendations JSON,
                    scan_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            'secrets': """
                CREATE TABLE IF NOT EXISTS secrets (
                    key VARCHAR(255) PRIMARY KEY,
                    value TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        }
        
        # Initialize database
        self._init_database()
        
        logger.info("Database controller initialized")
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool"""
        # Check if we have a connection in thread-local storage
        conn = getattr(local, 'conn', None)
        
        if conn is None:
            # Create a new connection
            try:
                conn = psycopg2.connect(**self.db_params)
                local.conn = conn
                logger.debug("Created new database connection")
            except Exception as e:
                logger.error(f"Error connecting to database: {str(e)}")
                raise
        
        try:
            # Return the connection for use in the with block
            yield conn
        except Exception as e:
            # If there's an error, rollback
            conn.rollback()
            logger.error(f"Database error, rolled back: {str(e)}")
            raise
        else:
            # If no error, commit the transaction
            conn.commit()
    
    def _init_database(self):
        """Initialize the database with required tables"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Create tables if they don't exist
                    for table_name, schema in self.schemas.items():
                        try:
                            cur.execute(schema)
                            logger.info(f"Initialized table: {table_name}")
                        except Exception as e:
                            logger.error(f"Error creating table {table_name}: {str(e)}")
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
    
    def execute_query(self, query: str, params: Optional[Union[Tuple, Dict[str, Any]]] = None, 
                    fetch: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a SQL query with parameters
        
        Args:
            query: SQL query to execute
            params: Query parameters (tuple or dict)
            fetch: Type of fetch to perform ('one', 'all', or None for no fetch)
            
        Returns:
            Query results if fetch is specified, otherwise None
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                try:
                    cur.execute(query, params or ())
                    
                    if fetch == 'one':
                        return dict(cur.fetchone()) if cur.rowcount > 0 else None
                    elif fetch == 'all':
                        return [dict(row) for row in cur.fetchall()]
                    else:
                        return None
                except Exception as e:
                    logger.error(f"Query execution error: {str(e)}")
                    logger.error(f"Query: {query}")
                    logger.error(f"Params: {params}")
                    raise
    
    def execute_batch(self, query: str, params_list: List[Tuple]) -> int:
        """
        Execute a batch query with multiple parameter sets
        
        Args:
            query: SQL query template to execute
            params_list: List of parameter tuples
            
        Returns:
            Number of rows affected
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    execute_values(cur, query, params_list)
                    return cur.rowcount
                except Exception as e:
                    logger.error(f"Batch execution error: {str(e)}")
                    logger.error(f"Query: {query}")
                    raise
    
    def close_all_connections(self):
        """Close all database connections"""
        if hasattr(local, 'conn'):
            try:
                local.conn.close()
                delattr(local, 'conn')
                logger.info("Closed database connection")
            except Exception as e:
                logger.error(f"Error closing database connection: {str(e)}")
    
    # Secrets management methods
    def get_secret(self, key: str) -> Optional[str]:
        """
        Get a secret by key
        
        Args:
            key: Secret key to retrieve
            
        Returns:
            Secret value or None if not found
        """
        query = "SELECT value FROM secrets WHERE key = %s"
        result = self.execute_query(query, (key,), fetch='one')
        return result['value'] if result else None
    
    def set_secret(self, key: str, value: str, description: Optional[str] = None) -> bool:
        """
        Set a secret value
        
        Args:
            key: Secret key
            value: Secret value
            description: Optional description of the secret
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if secret exists
            existing = self.get_secret(key)
            
            if existing is not None:
                # Update existing secret
                query = """
                    UPDATE secrets
                    SET value = %s, description = %s, updated_at = %s
                    WHERE key = %s
                """
                self.execute_query(query, (value, description, datetime.datetime.now(), key))
            else:
                # Insert new secret
                query = """
                    INSERT INTO secrets (key, value, description)
                    VALUES (%s, %s, %s)
                """
                self.execute_query(query, (key, value, description))
            
            logger.info(f"Secret '{key}' saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting secret '{key}': {str(e)}")
            return False
    
    def delete_secret(self, key: str) -> bool:
        """
        Delete a secret
        
        Args:
            key: Secret key to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            query = "DELETE FROM secrets WHERE key = %s"
            self.execute_query(query, (key,))
            logger.info(f"Secret '{key}' deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting secret '{key}': {str(e)}")
            return False
    
    def list_secrets(self, include_values: bool = False) -> List[Dict[str, Any]]:
        """
        List all stored secrets
        
        Args:
            include_values: Whether to include secret values in the result
            
        Returns:
            List of secret information
        """
        fields = "key, description, created_at, updated_at"
        if include_values:
            fields += ", value"
        
        query = f"SELECT {fields} FROM secrets ORDER BY key"
        return self.execute_query(query, fetch='all') or []
    
    # User management methods
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user by ID"""
        query = "SELECT * FROM users WHERE id = %s"
        return self.execute_query(query, (user_id,), fetch='one')
    
    def upsert_user(self, user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert or update a user"""
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'bio', 'profile_image_url']
        values = [user_data.get(field) for field in fields]
        
        query = f"""
            INSERT INTO users ({', '.join(fields)})
            VALUES ({', '.join(['%s'] * len(fields))})
            ON CONFLICT (id) DO UPDATE SET
                username = EXCLUDED.username,
                email = EXCLUDED.email,
                first_name = EXCLUDED.first_name,
                last_name = EXCLUDED.last_name,
                bio = EXCLUDED.bio,
                profile_image_url = EXCLUDED.profile_image_url,
                updated_at = CURRENT_TIMESTAMP
            RETURNING *
        """
        
        return self.execute_query(query, tuple(values), fetch='one')
    
    # Cognitive resonance log methods
    def log_cognitive_resonance(self, user_id: Optional[str], memory_waves: float, 
                               emotional_viscosity: float, perturbation: float,
                               cognitive_resonance: float, note: Optional[str] = None,
                               source: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Log a cognitive resonance event"""
        query = """
            INSERT INTO cognitive_resonance_logs
            (user_id, memory_waves, emotional_viscosity, perturbation, cognitive_resonance, note, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING *
        """
        
        params = (user_id, memory_waves, emotional_viscosity, perturbation, 
                 cognitive_resonance, note, source)
        
        return self.execute_query(query, params, fetch='one')
    
    def get_cognitive_resonance_logs(self, user_id: Optional[str] = None, 
                                    limit: int = 100) -> List[Dict[str, Any]]:
        """Get cognitive resonance logs for a user"""
        query = """
            SELECT * FROM cognitive_resonance_logs
            WHERE user_id IS NOT DISTINCT FROM %s
            ORDER BY created_at DESC
            LIMIT %s
        """
        
        return self.execute_query(query, (user_id, limit), fetch='all') or []
    
    # Agent state methods
    def save_agent_state(self, agent_id: str, agent_type: str, name: str,
                        description: str, state: Dict[str, Any],
                        status: str) -> Optional[Dict[str, Any]]:
        """Save an agent's state"""
        query = """
            INSERT INTO agent_state
            (agent_id, agent_type, name, description, state, status, last_active, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (agent_id) DO UPDATE SET
                agent_type = EXCLUDED.agent_type,
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                state = EXCLUDED.state,
                status = EXCLUDED.status,
                last_active = EXCLUDED.last_active,
                updated_at = EXCLUDED.updated_at
            RETURNING *
        """
        
        now = datetime.datetime.now()
        params = (agent_id, agent_type, name, description, json.dumps(state),
                 status, now, now)
        
        return self.execute_query(query, params, fetch='one')
    
    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent's state"""
        query = "SELECT * FROM agent_state WHERE agent_id = %s"
        result = self.execute_query(query, (agent_id,), fetch='one')
        
        if result and 'state' in result:
            # Parse JSON state
            try:
                result['state'] = json.loads(result['state'])
            except Exception as e:
                logger.error(f"Error parsing agent state JSON: {str(e)}")
        
        return result
    
    def list_agents(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all agents with optional status filter"""
        if status:
            query = "SELECT * FROM agent_state WHERE status = %s ORDER BY last_active DESC"
            return self.execute_query(query, (status,), fetch='all') or []
        else:
            query = "SELECT * FROM agent_state ORDER BY last_active DESC"
            return self.execute_query(query, fetch='all') or []
    
    # Discover logs methods
    def save_discover_log(self, scan_type: str, findings: Dict[str, Any],
                         recommendations: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Save a discover scan log"""
        query = """
            INSERT INTO discover_logs
            (scan_type, findings, recommendations)
            VALUES (%s, %s, %s)
            RETURNING *
        """
        
        params = (
            scan_type,
            json.dumps(findings),
            json.dumps(recommendations) if recommendations else None
        )
        
        return self.execute_query(query, params, fetch='one')
    
    def get_latest_discover_log(self, scan_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the latest discover scan log"""
        if scan_type:
            query = """
                SELECT * FROM discover_logs
                WHERE scan_type = %s
                ORDER BY scan_timestamp DESC
                LIMIT 1
            """
            result = self.execute_query(query, (scan_type,), fetch='one')
        else:
            query = """
                SELECT * FROM discover_logs
                ORDER BY scan_timestamp DESC
                LIMIT 1
            """
            result = self.execute_query(query, fetch='one')
        
        if result:
            # Parse JSON fields
            try:
                result['findings'] = json.loads(result['findings'])
                if result['recommendations']:
                    result['recommendations'] = json.loads(result['recommendations'])
            except Exception as e:
                logger.error(f"Error parsing discover log JSON: {str(e)}")
        
        return result

# Create a singleton instance
db = DBController()