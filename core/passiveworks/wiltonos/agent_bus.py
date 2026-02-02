"""
WiltonOS Agent Logic Bus - Central Orchestration Layer
Powered by LangChain for agent management and routing
"""

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from typing import List, Dict, Any, Optional
import os
import json
import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [AGENT_BUS] %(message)s",
    handlers=[
        logging.FileHandler("logs/agent_bus.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agent_bus")

class AgentProfile:
    """Profile definition for an agent in the WiltonOS ecosystem"""
    
    def __init__(
        self, 
        agent_id: str,
        name: str,
        description: str,
        capabilities: List[str],
        instruction_template: str,
        model: str = "gpt-4o",
        temperature: float = 0.7
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.instruction_template = instruction_template
        self.model = model
        self.temperature = temperature
        self.creation_timestamp = datetime.datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "instruction_template": self.instruction_template,
            "model": self.model,
            "temperature": self.temperature,
            "creation_timestamp": self.creation_timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentProfile':
        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            description=data["description"],
            capabilities=data["capabilities"],
            instruction_template=data["instruction_template"],
            model=data.get("model", "gpt-4o"),
            temperature=data.get("temperature", 0.7)
        )
    
    @classmethod
    def from_file(cls, filepath: str) -> 'AgentProfile':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

class WiltonOSAgentBus:
    """
    Central orchestration system for WiltonOS agents
    
    Manages agent creation, routing, message passing, and state management
    for the entire WiltonOS ecosystem.
    """
    
    def __init__(self):
        self.agents = {}
        self.agent_memory = {}
        self.tools = self._initialize_tools()
        logger.info("Agent Bus initialized")
        
    def _initialize_tools(self) -> List[Tool]:
        """Initialize standard tools available to all agents"""
        return [
            Tool(
                name="search_documents",
                func=self._search_documents,
                description="Search through WiltonOS documents and knowledge base"
            ),
            Tool(
                name="execute_query",
                func=self._execute_query,
                description="Execute a database query to retrieve or store information"
            ),
            Tool(
                name="agent_communication",
                func=self._agent_communication,
                description="Send a message to another agent in the ecosystem"
            ),
            Tool(
                name="environment_query",
                func=self._environment_query,
                description="Query the environment for system state, time, or other context"
            )
        ]
    
    def _search_documents(self, query: str) -> str:
        """Search through WiltonOS documents"""
        # This will be implemented to use vector search
        logger.info(f"Document search request: {query}")
        return "Document search functionality pending integration"
    
    def _execute_query(self, query: str) -> str:
        """Execute a database query"""
        # This will connect to the database
        logger.info(f"Database query request: {query}")
        return "Database query functionality pending integration"
    
    def _agent_communication(self, message: str) -> str:
        """Handle inter-agent communication"""
        try:
            data = json.loads(message)
            target_agent = data.get("target_agent")
            content = data.get("content")
            
            if target_agent and target_agent in self.agents:
                logger.info(f"Message sent to agent {target_agent}")
                # In a full implementation, this would queue the message
                return f"Message delivered to {target_agent}"
            else:
                return "Invalid target agent specified"
        except Exception as e:
            return f"Error in agent communication: {str(e)}"
    
    def _environment_query(self, query: str) -> str:
        """Query the environment for context"""
        queries = {
            "time": datetime.datetime.now().isoformat(),
            "active_agents": list(self.agents.keys()),
            "system_status": "operational",
            "quantum_coherence": 0.97  # Simulated value
        }
        
        if query in queries:
            return str(queries[query])
        
        return "Unknown environment query"
    
    def register_agent(self, profile: AgentProfile) -> str:
        """Register a new agent with the bus"""
        agent_id = profile.agent_id
        
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already exists, updating profile")
        
        # Initialize LLM for the agent
        llm = ChatOpenAI(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model=profile.model,
            temperature=profile.temperature
        )
        
        # Create memory for the agent
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.agent_memory[agent_id] = memory
        
        # Initialize the agent with tools
        agent = initialize_agent(
            self.tools,
            llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True
        )
        
        self.agents[agent_id] = {
            "profile": profile,
            "agent": agent,
            "status": "active",
            "last_active": datetime.datetime.now().isoformat()
        }
        
        logger.info(f"Agent {agent_id} ({profile.name}) registered")
        return agent_id
    
    def send_message(self, agent_id: str, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Send a message to a specific agent"""
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return f"Error: Agent {agent_id} not found"
        
        agent_data = self.agents[agent_id]
        agent = agent_data["agent"]
        
        # Update last active timestamp
        agent_data["last_active"] = datetime.datetime.now().isoformat()
        
        # Add context to message if provided
        if context:
            message = f"{message}\n\nContext: {json.dumps(context)}"
        
        try:
            # Run the agent with the message
            response = agent.run(message)
            logger.info(f"Message processed by agent {agent_id}")
            return response
        except Exception as e:
            logger.error(f"Error in agent {agent_id}: {str(e)}")
            return f"Error in agent processing: {str(e)}"
    
    def broadcast_message(self, message: str, exclude_agents: List[str] = None) -> Dict[str, str]:
        """Broadcast a message to all active agents"""
        exclude_agents = exclude_agents or []
        responses = {}
        
        for agent_id, agent_data in self.agents.items():
            if agent_id not in exclude_agents and agent_data["status"] == "active":
                responses[agent_id] = self.send_message(agent_id, message)
        
        return responses
    
    def deactivate_agent(self, agent_id: str) -> bool:
        """Deactivate an agent temporarily"""
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        self.agents[agent_id]["status"] = "inactive"
        logger.info(f"Agent {agent_id} deactivated")
        return True
    
    def activate_agent(self, agent_id: str) -> bool:
        """Reactivate an agent"""
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        self.agents[agent_id]["status"] = "active"
        logger.info(f"Agent {agent_id} activated")
        return True
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents with their status"""
        result = []
        for agent_id, agent_data in self.agents.items():
            profile = agent_data["profile"]
            result.append({
                "agent_id": agent_id,
                "name": profile.name,
                "description": profile.description,
                "status": agent_data["status"],
                "last_active": agent_data["last_active"],
                "capabilities": profile.capabilities
            })
        return result
    
    def save_state(self, filepath: str = "state/agent_bus_state.json") -> bool:
        """Save the current state of the agent bus"""
        state = {
            "agents": {
                agent_id: {
                    "profile": agent_data["profile"].to_dict(),
                    "status": agent_data["status"],
                    "last_active": agent_data["last_active"]
                }
                for agent_id, agent_data in self.agents.items()
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Agent Bus state saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving Agent Bus state: {str(e)}")
            return False
    
    def load_state(self, filepath: str = "state/agent_bus_state.json") -> bool:
        """Load a previously saved state"""
        if not os.path.exists(filepath):
            logger.error(f"State file {filepath} not found")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Reset current state
            self.agents = {}
            
            # Load agents from saved state
            for agent_id, agent_data in state["agents"].items():
                profile = AgentProfile.from_dict(agent_data["profile"])
                self.register_agent(profile)
                
                # Restore status
                self.agents[agent_id]["status"] = agent_data["status"]
                self.agents[agent_id]["last_active"] = agent_data["last_active"]
            
            logger.info(f"Agent Bus state loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading Agent Bus state: {str(e)}")
            return False

# Create a singleton instance
agent_bus = WiltonOSAgentBus()