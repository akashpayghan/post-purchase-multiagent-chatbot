"""
LangGraph Workflow Orchestration
Main multi-agent workflow using LangGraph
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from .state_management import ConversationState, StateManager
from .routing_logic import AgentRouter

class AgentOrchestrator:
    """Orchestrate multi-agent workflow using LangGraph"""
    
    def __init__(self, agents: Dict[str, Any]):
        """
        Initialize orchestrator
        
        Args:
            agents: Dictionary of agent instances
        """
        self.agents = agents
        self.state_manager = StateManager()
        self.router = AgentRouter()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        
        # Create graph with state
        workflow = StateGraph(ConversationState)
        
        # Add nodes for each agent
        workflow.add_node("controller", self._controller_node)
        workflow.add_node("monitor", self._monitor_node)
        workflow.add_node("visual", self._visual_node)
        workflow.add_node("exchange", self._exchange_node)
        workflow.add_node("resolution", self._resolution_node)
        
        # Set entry point
        workflow.set_entry_point("controller")
        
        # Add conditional routing from controller
        workflow.add_conditional_edges(
            "controller",
            self._route_from_controller,
            {
                "monitor": "monitor",
                "visual": "visual",
                "exchange": "exchange",
                "resolution": "resolution",
                END: END  # Fixed: Use END directly, not string "end"
            }
        )
        
        # Add edges back to controller or END
        for agent_name in ["monitor", "visual", "exchange", "resolution"]:
            workflow.add_conditional_edges(
                agent_name,
                self._route_after_agent,
                {
                    "controller": "controller",
                    END: END  # Fixed: Use END directly
                }
            )
        
        return workflow.compile()
    
    def _controller_node(self, state: ConversationState) -> ConversationState:
        """Controller agent node"""
        messages = state['messages']
        if not messages:
            return state
        
        last_message = messages[-1]['content']
        
        # Route to appropriate agent
        routing_decision = self.router.route(last_message, state['context'])
        
        # Update state
        state['current_agent'] = routing_decision['agent']
        state['next_action'] = routing_decision['agent']
        state['context']['routing_confidence'] = routing_decision['confidence']
        
        return state
    
    def _monitor_node(self, state: ConversationState) -> ConversationState:
        """Monitor agent node"""
        monitor_agent = self.agents.get('monitor')
        
        if not monitor_agent:
            state['next_action'] = END
            return state
        
        order_id = state.get('order_id') or state['context'].get('order_id')
        
        if order_id:
            try:
                result = monitor_agent.check_order_status(order_id, state['context'])
                
                state['messages'].append({
                    'role': 'assistant',
                    'content': result['status_message'],
                    'agent_type': 'monitor',
                    'timestamp': None
                })
                
                if result.get('issues_detected'):
                    state['context']['issues_detected'] = result['issues_detected']
                    state['next_action'] = 'resolution'
                else:
                    state['next_action'] = END
            except:
                state['next_action'] = END
        else:
            state['next_action'] = END
        
        return state
    
    def _visual_node(self, state: ConversationState) -> ConversationState:
        """Visual agent node"""
        state['next_action'] = END  # Simplified for now
        return state
    
    def _exchange_node(self, state: ConversationState) -> ConversationState:
        """Exchange agent node"""
        state['next_action'] = END  # Simplified for now
        return state
    
    def _resolution_node(self, state: ConversationState) -> ConversationState:
        """Resolution agent node"""
        state['next_action'] = END  # Simplified for now
        return state
    
    def _route_from_controller(self, state: ConversationState) -> Literal["monitor", "visual", "exchange", "resolution"] | type(END):
        """Route from controller to next agent"""
        next_action = state.get('next_action', END)
        
        if next_action in ['monitor', 'visual', 'exchange', 'resolution']:
            return next_action
        
        return END  # Return END directly, not string
    
    def _route_after_agent(self, state: ConversationState) -> Literal["controller"] | type(END):
        """Route after agent completes task"""
        next_action = state.get('next_action', END)
        
        if next_action == END or state['context'].get('turn_count', 0) > 10:
            return END  # Return END directly
        
        if next_action in ['monitor', 'visual', 'exchange', 'resolution']:
            return 'controller'
        
        return END  # Return END directly
    
    def run(self, conversation_id: str, user_message: str, 
            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run orchestration workflow"""
        # Simplified version - just return direct response
        return {
            'response': "I'm processing your request...",
            'agent': 'controller',
            'state': {}
        }
