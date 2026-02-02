"""
WiltonOS Signal Mesh - Real-time Event Communication System
Powered by Socket.IO for distributed real-time messaging
"""

import socketio
import asyncio
import json
import logging
import datetime
import uuid
from typing import Dict, Any, List, Callable, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [SIGNAL_MESH] %(message)s",
    handlers=[
        logging.FileHandler("logs/signal_mesh.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("signal_mesh")

class SignalMeshNode:
    """
    A node in the WiltonOS Signal Mesh network
    
    Handles real-time event distribution across the WiltonOS ecosystem
    using Socket.IO for high-performance message passing.
    """
    
    def __init__(
        self,
        node_id: str = None,
        node_type: str = "standard",
        server_url: str = "http://localhost:5000",
        namespace: str = "/ws"
    ):
        # Generate a UUID if node_id not provided
        self.node_id = node_id or str(uuid.uuid4())
        self.node_type = node_type
        self.server_url = server_url
        self.namespace = namespace
        
        # Initialize Socket.IO client
        self.sio = socketio.AsyncClient()
        
        # Event handlers
        self.event_handlers = {}
        
        # Connection status
        self.connected = False
        self.last_heartbeat = None
        
        # Message queue for offline operation
        self.message_queue = []
        
        # Setup event handlers
        self._setup_socketio_handlers()
        
        logger.info(f"Signal Mesh Node initialized: {self.node_id} (Type: {self.node_type})")
    
    def _setup_socketio_handlers(self):
        """Setup internal Socket.IO event handlers"""
        
        @self.sio.event
        async def connect():
            self.connected = True
            self.last_heartbeat = datetime.datetime.now()
            logger.info(f"Connected to Signal Mesh: {self.server_url}{self.namespace}")
            
            # Register node with the mesh
            await self.sio.emit('register', {
                'node_id': self.node_id,
                'node_type': self.node_type,
                'capabilities': self._get_capabilities()
            }, namespace=self.namespace)
            
            # Process any queued messages
            await self._process_message_queue()
        
        @self.sio.event
        async def disconnect():
            self.connected = False
            logger.info("Disconnected from Signal Mesh")
        
        @self.sio.event(namespace=self.namespace)
        async def signal(data):
            await self._handle_signal(data)
        
        @self.sio.event(namespace=self.namespace)
        async def heartbeat(data):
            self.last_heartbeat = datetime.datetime.now()
            await self.sio.emit('heartbeat_ack', {
                'node_id': self.node_id,
                'timestamp': datetime.datetime.now().isoformat()
            }, namespace=self.namespace)
    
    def _get_capabilities(self) -> List[str]:
        """Get the capabilities of this node based on registered event handlers"""
        return list(self.event_handlers.keys())
    
    async def _handle_signal(self, data: Dict[str, Any]):
        """Handle an incoming signal event"""
        try:
            signal_type = data.get('type')
            payload = data.get('payload', {})
            source = data.get('source')
            
            logger.info(f"Received signal: {signal_type} from {source}")
            
            if signal_type in self.event_handlers:
                for handler in self.event_handlers[signal_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(payload, source)
                        else:
                            handler(payload, source)
                    except Exception as e:
                        logger.error(f"Error in handler for {signal_type}: {str(e)}")
            else:
                logger.warning(f"No handler registered for signal type: {signal_type}")
        
        except Exception as e:
            logger.error(f"Error handling signal: {str(e)}")
    
    async def _process_message_queue(self):
        """Process queued messages after connecting"""
        if not self.message_queue:
            return
        
        logger.info(f"Processing {len(self.message_queue)} queued messages")
        
        for message in self.message_queue:
            signal_type = message.get('type')
            payload = message.get('payload', {})
            target = message.get('target')
            await self.emit_signal(signal_type, payload, target)
        
        # Clear the queue
        self.message_queue = []
    
    async def connect(self):
        """Connect to the Signal Mesh server"""
        try:
            await self.sio.connect(
                self.server_url,
                namespaces=[self.namespace],
                transports=['websocket']
            )
        except Exception as e:
            logger.error(f"Error connecting to Signal Mesh: {str(e)}")
            raise
    
    async def disconnect(self):
        """Disconnect from the Signal Mesh server"""
        if self.connected:
            await self.sio.disconnect()
    
    def register_handler(self, signal_type: str, handler: Callable[[Dict[str, Any], str], None]):
        """Register a handler for a specific signal type"""
        if signal_type not in self.event_handlers:
            self.event_handlers[signal_type] = []
        
        self.event_handlers[signal_type].append(handler)
        logger.info(f"Registered handler for signal type: {signal_type}")
    
    def unregister_handler(self, signal_type: str, handler: Callable[[Dict[str, Any], str], None]) -> bool:
        """Unregister a handler for a specific signal type"""
        if signal_type not in self.event_handlers:
            return False
        
        if handler in self.event_handlers[signal_type]:
            self.event_handlers[signal_type].remove(handler)
            logger.info(f"Unregistered handler for signal type: {signal_type}")
            return True
        
        return False
    
    async def emit_signal(
        self,
        signal_type: str,
        payload: Dict[str, Any],
        target: Optional[Union[str, List[str]]] = None
    ):
        """Emit a signal to the mesh"""
        message = {
            'type': signal_type,
            'payload': payload,
            'source': self.node_id,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if target:
            message['target'] = target
        
        # If not connected, queue the message
        if not self.connected:
            self.message_queue.append(message)
            logger.info(f"Queued signal: {signal_type} (not connected)")
            return
        
        try:
            await self.sio.emit('signal', message, namespace=self.namespace)
            logger.info(f"Emitted signal: {signal_type}")
        except Exception as e:
            logger.error(f"Error emitting signal: {str(e)}")
            # Queue the message for retry
            self.message_queue.append(message)
    
    async def request_response(
        self,
        signal_type: str,
        payload: Dict[str, Any],
        target: str,
        timeout: float = 10.0
    ) -> Optional[Dict[str, Any]]:
        """Send a signal and wait for a response"""
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        payload['request_id'] = request_id
        
        # Create a future to wait for the response
        response_future = asyncio.Future()
        
        # Register a temporary handler for the response
        async def response_handler(data, source):
            if data.get('request_id') == request_id:
                response_future.set_result(data)
        
        response_signal_type = f"{signal_type}_response"
        self.register_handler(response_signal_type, response_handler)
        
        # Send the request
        await self.emit_signal(signal_type, payload, target)
        
        try:
            # Wait for the response with timeout
            response = await asyncio.wait_for(response_future, timeout)
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for response to {signal_type}")
            return None
        finally:
            # Unregister the temporary handler
            self.unregister_handler(response_signal_type, response_handler)

class SignalMeshServer:
    """
    Signal Mesh Server for WiltonOS
    
    Manages Socket.IO server for distributed communication across nodes.
    """
    
    def __init__(self, port: int = 5000):
        self.port = port
        self.sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
        self.app = socketio.ASGIApp(self.sio, socketio_path='socket.io')
        
        # Track connected nodes
        self.nodes = {}
        
        # Setup event handlers
        self._setup_socketio_handlers()
        
        logger.info(f"Signal Mesh Server initialized on port {port}")
    
    def _setup_socketio_handlers(self):
        @self.sio.event
        async def connect(sid, environ):
            logger.info(f"Client connected: {sid}")
        
        @self.sio.event
        async def disconnect(sid):
            # Find and remove the disconnected node
            node_id = None
            for nid, node_data in self.nodes.items():
                if node_data.get('sid') == sid:
                    node_id = nid
                    break
            
            if node_id:
                logger.info(f"Node disconnected: {node_id}")
                del self.nodes[node_id]
            else:
                logger.info(f"Client disconnected: {sid}")
        
        @self.sio.event
        async def register(sid, data):
            node_id = data.get('node_id')
            if not node_id:
                logger.warning(f"Registration without node_id from {sid}")
                return
            
            self.nodes[node_id] = {
                'sid': sid,
                'node_type': data.get('node_type', 'unknown'),
                'capabilities': data.get('capabilities', []),
                'last_heartbeat': datetime.datetime.now().isoformat()
            }
            
            logger.info(f"Node registered: {node_id} ({data.get('node_type')})")
            
            # Acknowledge registration
            await self.sio.emit('register_ack', {
                'node_id': node_id,
                'status': 'registered',
                'mesh_nodes': len(self.nodes)
            }, room=sid)
        
        @self.sio.event
        async def signal(sid, data):
            source = data.get('source')
            target = data.get('target')
            signal_type = data.get('type')
            
            logger.info(f"Signal from {source}: {signal_type}")
            
            # If there's a specific target or targets
            if target:
                if isinstance(target, list):
                    for t in target:
                        await self._forward_to_target(t, data)
                else:
                    await self._forward_to_target(target, data)
            else:
                # Broadcast to all nodes except the source
                for node_id, node_data in self.nodes.items():
                    if node_id != source:
                        await self.sio.emit('signal', data, room=node_data['sid'])
        
        @self.sio.event
        async def heartbeat_ack(sid, data):
            node_id = data.get('node_id')
            if node_id in self.nodes:
                self.nodes[node_id]['last_heartbeat'] = data.get('timestamp')
    
    async def _forward_to_target(self, target: str, data: Dict[str, Any]):
        """Forward a signal to a specific target node"""
        if target in self.nodes:
            target_sid = self.nodes[target]['sid']
            await self.sio.emit('signal', data, room=target_sid)
        else:
            logger.warning(f"Target node not found: {target}")
    
    async def start_heartbeat(self, interval: int = 30):
        """Start sending heartbeat signals to nodes"""
        while True:
            current_time = datetime.datetime.now()
            nodes_to_remove = []
            
            for node_id, node_data in self.nodes.items():
                # Send heartbeat
                try:
                    await self.sio.emit('heartbeat', {
                        'timestamp': current_time.isoformat()
                    }, room=node_data['sid'])
                    
                    # Check if node hasn't responded for a while
                    last_hb = node_data.get('last_heartbeat')
                    if last_hb:
                        last_hb_time = datetime.datetime.fromisoformat(last_hb)
                        if (current_time - last_hb_time).total_seconds() > interval * 3:
                            logger.warning(f"Node {node_id} hasn't responded, marking for removal")
                            nodes_to_remove.append(node_id)
                except Exception as e:
                    logger.error(f"Error sending heartbeat to {node_id}: {str(e)}")
            
            # Remove unresponsive nodes
            for node_id in nodes_to_remove:
                if node_id in self.nodes:
                    logger.info(f"Removing unresponsive node: {node_id}")
                    del self.nodes[node_id]
            
            await asyncio.sleep(interval)
    
    def get_node_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all connected nodes"""
        return self.nodes

# Create a singleton server instance
signal_mesh_server = SignalMeshServer()

# Helper function to create a new node
def create_signal_node(
    node_id: str = None,
    node_type: str = "standard",
    server_url: str = "http://localhost:5000",
    namespace: str = "/ws"
) -> SignalMeshNode:
    """Create and return a new Signal Mesh Node"""
    return SignalMeshNode(node_id, node_type, server_url, namespace)