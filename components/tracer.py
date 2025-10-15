"""
Metis Agentic System Tracer

This module provides comprehensive tracing and observability for the Metis
agentic orchestration system. It tracks execution flows, tool usage,
decision points, and performance metrics.

Save as: single_agent/components/tracer.py
"""

import json
import time
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import os

class TraceLevel(Enum):
    """Trace detail levels"""
    MINIMAL = "minimal"      # Only major events
    STANDARD = "standard"    # Standard workflow tracking
    DETAILED = "detailed"    # Detailed step tracking
    DEBUG = "debug"         # Full debug information

class EventType(Enum):
    """Types of trace events"""
    QUERY_START = "query_start"
    QUERY_END = "query_end"
    INTENT_CLASSIFICATION = "intent_classification"
    PLANNING_START = "planning_start"
    PLANNING_END = "planning_end"
    TASK_EXECUTION = "task_execution"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    EVALUATION_START = "evaluation_start"
    EVALUATION_END = "evaluation_end"
    IMPROVEMENT_ATTEMPT = "improvement_attempt"
    ERROR = "error"
    WARNING = "warning"
    CUSTOM = "custom"

@dataclass
class TraceEvent:
    """Individual trace event"""
    event_id: str
    trace_id: str
    session_id: str
    timestamp: float
    event_type: EventType
    component: str
    message: str
    data: Dict[str, Any]
    duration_ms: Optional[float] = None
    parent_event_id: Optional[str] = None
    tags: List[str] = None
    level: TraceLevel = TraceLevel.STANDARD
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass 
class TraceSession:
    """Complete trace session information"""
    session_id: str
    trace_id: str
    user_query: str
    start_time: float
    end_time: Optional[float] = None
    events: List[TraceEvent] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.events is None:
            self.events = []
        if self.metadata is None:
            self.metadata = {}

class MetisTracer:
    """
    Main tracer class for the Metis agentic system
    """
    
    def __init__(self, 
                 trace_level: TraceLevel = TraceLevel.STANDARD,
                 output_dir: str = "traces",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 max_trace_size: int = 10000):
        """
        Initialize the tracer
        
        Args:
            trace_level: Level of detail to capture
            output_dir: Directory for trace files
            enable_console: Enable console output
            enable_file: Enable file output
            max_trace_size: Maximum number of events per trace
        """
        self.trace_level = trace_level
        self.output_dir = Path(output_dir)
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.max_trace_size = max_trace_size
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Current session tracking
        self.current_session: Optional[TraceSession] = None
        self.active_traces: Dict[str, TraceSession] = {}
        
        # Event stack for nested operations
        self.event_stack: List[TraceEvent] = []
        
        # Statistics
        self.stats = {
            "total_sessions": 0,
            "total_events": 0,
            "error_count": 0,
            "avg_session_duration": 0.0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        print(f"🔍 MetisTracer initialized (level: {trace_level.value}, output: {output_dir})")
    
    def start_session(self, user_query: str, session_id: str = None, metadata: Dict[str, Any] = None) -> str:
        """
        Start a new trace session
        
        Args:
            user_query: The user's query that started this session
            session_id: Optional session ID (auto-generated if not provided)
            metadata: Additional session metadata
            
        Returns:
            Session ID for this trace
        """
        with self._lock:
            session_id = session_id or str(uuid.uuid4())
            trace_id = str(uuid.uuid4())
            
            session = TraceSession(
                session_id=session_id,
                trace_id=trace_id,
                user_query=user_query,
                start_time=time.time(),
                metadata=metadata or {}
            )
            
            self.current_session = session
            self.active_traces[session_id] = session
            self.stats["total_sessions"] += 1
            
            # Log session start
            self._add_event(
                event_type=EventType.QUERY_START,
                component="MetisTracer",
                message=f"Started session for query: {user_query[:100]}...",
                data={
                    "query": user_query,
                    "session_id": session_id,
                    "trace_id": trace_id
                }
            )
            
            if self.enable_console:
                print(f"🚀 [TRACE-{session_id[:8]}] Session started: {user_query[:80]}...")
            
            return session_id
    
    def end_session(self, session_id: str = None, result: Any = None):
        """
        End a trace session
        
        Args:
            session_id: Session to end (current session if not specified)
            result: Final result of the session
        """
        with self._lock:
            session_id = session_id or (self.current_session.session_id if self.current_session else None)
            
            if session_id not in self.active_traces:
                print(f"⚠️ Warning: Trying to end unknown session {session_id}")
                return
            
            session = self.active_traces[session_id]
            session.end_time = time.time()
            duration = session.end_time - session.start_time
            
            # Log session end
            self._add_event(
                event_type=EventType.QUERY_END,
                component="MetisTracer", 
                message=f"Session completed in {duration:.2f}s",
                data={
                    "duration_seconds": duration,
                    "total_events": len(session.events),
                    "result_summary": str(result)[:200] if result else None
                }
            )
            
            # Update statistics
            self._update_stats(session)
            
            # Save to file if enabled
            if self.enable_file:
                self._save_session_to_file(session)
            
            if self.enable_console:
                print(f"🏁 [TRACE-{session_id[:8]}] Session ended: {duration:.2f}s, {len(session.events)} events")
            
            # Clean up
            if session_id in self.active_traces:
                del self.active_traces[session_id]
            
            if self.current_session and self.current_session.session_id == session_id:
                self.current_session = None
    
    def trace_event(self, 
                   event_type: EventType,
                   component: str,
                   message: str,
                   data: Dict[str, Any] = None,
                   level: TraceLevel = TraceLevel.STANDARD,
                   tags: List[str] = None):
        """
        Add a trace event
        
        Args:
            event_type: Type of event
            component: Component generating the event
            message: Human-readable message
            data: Additional event data
            level: Trace level for this event
            tags: Optional tags for filtering
        """
        if self._should_trace(level):
            self._add_event(event_type, component, message, data or {}, level, tags)
    
    def trace_tool_selection(self, task: str, selected_tool: str, method: str, candidates: List[str] = None):
        """Trace tool selection decisions"""
        self.trace_event(
            event_type=EventType.TOOL_SELECTION,
            component="ToolSelector",
            message=f"Selected {selected_tool} for task using {method}",
            data={
                "task": task,
                "selected_tool": selected_tool,
                "selection_method": method,
                "candidates": candidates or [],
                "task_hash": hash(task) % 10000
            },
            tags=["tool_selection", selected_tool.lower() if selected_tool else "none"]
        )
    
    def trace_tool_execution(self, tool_name: str, task: str, result: Any, duration_ms: float, success: bool = True):
        """Trace tool execution"""
        self.trace_event(
            event_type=EventType.TOOL_EXECUTION,
            component=tool_name,
            message=f"{'Successfully executed' if success else 'Failed to execute'} {tool_name}",
            data={
                "tool_name": tool_name,
                "task": task,
                "success": success,
                "result_length": len(str(result)) if result else 0,
                "result_preview": str(result)[:200] if result else None,
                "duration_ms": duration_ms
            },
            tags=["tool_execution", tool_name.lower(), "success" if success else "failure"]
        )
    
    def trace_evaluation(self, query: str, response: Any, score: float, passed: bool, suggestions: List[str] = None):
        """Trace response evaluation"""
        self.trace_event(
            event_type=EventType.EVALUATION_END,
            component="ResponseEvaluator",
            message=f"Evaluated response: {score:.1f}/10 ({'PASS' if passed else 'FAIL'})",
            data={
                "query": query,
                "score": score,
                "passed": passed,
                "response_length": len(str(response)) if response else 0,
                "improvement_suggestions": suggestions or [],
                "evaluation_timestamp": time.time()
            },
            tags=["evaluation", "pass" if passed else "fail"]
        )
    
    def trace_planning(self, original_task: str, subtasks: List[str], duration_ms: float):
        """Trace task planning"""
        self.trace_event(
            event_type=EventType.PLANNING_END,
            component="Planner",
            message=f"Created plan with {len(subtasks)} subtasks",
            data={
                "original_task": original_task,
                "subtasks": subtasks,
                "subtask_count": len(subtasks),
                "planning_duration_ms": duration_ms
            },
            tags=["planning", f"subtasks_{len(subtasks)}"]
        )
    
    def trace_error(self, component: str, error: Exception, context: Dict[str, Any] = None):
        """Trace errors"""
        self.trace_event(
            event_type=EventType.ERROR,
            component=component,
            message=f"Error in {component}: {str(error)}",
            data={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
                "timestamp": time.time()
            },
            level=TraceLevel.MINIMAL,  # Always trace errors
            tags=["error", component.lower()]
        )
        
        self.stats["error_count"] += 1
    
    def trace_custom(self, component: str, event_name: str, data: Dict[str, Any], message: str = None):
        """Trace custom events"""
        self.trace_event(
            event_type=EventType.CUSTOM,
            component=component,
            message=message or f"Custom event: {event_name}",
            data={"event_name": event_name, **data},
            tags=["custom", event_name.lower()]
        )
    
    def get_session_trace(self, session_id: str = None) -> Optional[TraceSession]:
        """Get trace for a specific session"""
        session_id = session_id or (self.current_session.session_id if self.current_session else None)
        return self.active_traces.get(session_id) or self._load_session_from_file(session_id)
    
    def get_session_summary(self, session_id: str = None) -> Dict[str, Any]:
        """Get summary of a session"""
        session = self.get_session_trace(session_id)
        if not session:
            return {"error": "Session not found"}
        
        events_by_type = {}
        tools_used = set()
        errors = []
        
        for event in session.events:
            event_type = event.event_type.value
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
            
            if event.event_type == EventType.TOOL_EXECUTION:
                tools_used.add(event.data.get("tool_name", "unknown"))
            
            if event.event_type == EventType.ERROR:
                errors.append(event.data.get("error_message", "Unknown error"))
        
        duration = (session.end_time or time.time()) - session.start_time
        
        return {
            "session_id": session.session_id,
            "query": session.user_query,
            "duration_seconds": duration,
            "total_events": len(session.events),
            "events_by_type": events_by_type,
            "tools_used": list(tools_used),
            "error_count": len(errors),
            "errors": errors,
            "status": "completed" if session.end_time else "active"
        }
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get overall tracing statistics"""
        active_sessions = len(self.active_traces)
        
        return {
            **self.stats,
            "active_sessions": active_sessions,
            "trace_level": self.trace_level.value,
            "output_directory": str(self.output_dir),
            "uptime_seconds": time.time() - (self.stats.get("start_time", time.time()))
        }
    
    def export_trace_data(self, session_id: str = None, format: str = "json") -> str:
        """Export trace data in various formats"""
        session = self.get_session_trace(session_id)
        if not session:
            return "Session not found"
        
        if format.lower() == "json":
            return json.dumps(asdict(session), indent=2, default=str)
        elif format.lower() == "csv":
            return self._export_to_csv(session)
        elif format.lower() == "markdown":
            return self._export_to_markdown(session)
        else:
            return f"Unsupported format: {format}"
    
    # Context manager support
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # End any active sessions
        for session_id in list(self.active_traces.keys()):
            self.end_session(session_id)
    
    # Private methods
    
    def _should_trace(self, level: TraceLevel) -> bool:
        """Check if we should trace at this level"""
        level_order = {
            TraceLevel.MINIMAL: 0,
            TraceLevel.STANDARD: 1, 
            TraceLevel.DETAILED: 2,
            TraceLevel.DEBUG: 3
        }
        return level_order[level] <= level_order[self.trace_level]
    
    def _add_event(self, 
                  event_type: EventType,
                  component: str,
                  message: str,
                  data: Dict[str, Any],
                  level: TraceLevel = TraceLevel.STANDARD,
                  tags: List[str] = None):
        """Add an event to the current session"""
        if not self.current_session:
            return
        
        event = TraceEvent(
            event_id=str(uuid.uuid4()),
            trace_id=self.current_session.trace_id,
            session_id=self.current_session.session_id,
            timestamp=time.time(),
            event_type=event_type,
            component=component,
            message=message,
            data=data,
            level=level,
            tags=tags or [],
            parent_event_id=self.event_stack[-1].event_id if self.event_stack else None
        )
        
        self.current_session.events.append(event)
        self.stats["total_events"] += 1
        
        # Console output for important events
        if self.enable_console and self._should_trace(level):
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S.%f")[:-3]
            session_short = event.session_id[:8]
            
            # Color coding based on event type
            color_map = {
                EventType.ERROR: "🔴",
                EventType.WARNING: "🟡", 
                EventType.TOOL_EXECUTION: "🔧",
                EventType.EVALUATION_END: "📊",
                EventType.QUERY_START: "🚀",
                EventType.QUERY_END: "🏁"
            }
            
            icon = color_map.get(event_type, "📝")
            print(f"{icon} [{timestamp}] [{session_short}] {component}: {message}")
    
    def _update_stats(self, session: TraceSession):
        """Update tracing statistics"""
        if session.end_time and session.start_time:
            duration = session.end_time - session.start_time
            
            # Update average duration
            total_sessions = self.stats["total_sessions"]
            current_avg = self.stats["avg_session_duration"]
            self.stats["avg_session_duration"] = ((current_avg * (total_sessions - 1)) + duration) / total_sessions
    
    def _save_session_to_file(self, session: TraceSession):
        """Save session trace to file"""
        try:
            timestamp = datetime.fromtimestamp(session.start_time).strftime("%Y%m%d_%H%M%S")
            filename = f"trace_{timestamp}_{session.session_id[:8]}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(asdict(session), f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Failed to save trace file: {str(e)}")
    
    def _load_session_from_file(self, session_id: str) -> Optional[TraceSession]:
        """Load session from file (simplified implementation)"""
        # In a full implementation, this would search through trace files
        return None
    
    def _export_to_csv(self, session: TraceSession) -> str:
        """Export session to CSV format"""
        lines = ["timestamp,event_type,component,message,duration_ms"]
        
        for event in session.events:
            timestamp = datetime.fromtimestamp(event.timestamp).isoformat()
            duration = event.duration_ms or ""
            
            # Escape commas in message
            message = event.message.replace(",", ";")
            
            lines.append(f"{timestamp},{event.event_type.value},{event.component},{message},{duration}")
        
        return "\n".join(lines)
    
    def _export_to_markdown(self, session: TraceSession) -> str:
        """Export session to Markdown format"""
        duration = (session.end_time or time.time()) - session.start_time
        
        md = f"""# Trace Report: {session.session_id}

**Query:** {session.user_query}  
**Duration:** {duration:.2f} seconds  
**Events:** {len(session.events)}  
**Status:** {'Completed' if session.end_time else 'Active'}

## Event Timeline

| Time | Component | Event | Message |
|------|-----------|-------|---------|
"""
        
        for event in session.events:
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S.%f")[:-3]
            md += f"| {timestamp} | {event.component} | {event.event_type.value} | {event.message} |\n"
        
        return md

# Decorator for automatic tracing
def trace_method(component_name: str = None, event_type: EventType = EventType.CUSTOM):
    """Decorator to automatically trace method calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get tracer from global or first argument
            tracer = get_global_tracer()
            if not tracer:
                return func(*args, **kwargs)
            
            comp_name = component_name or (args[0].__class__.__name__ if args else "Unknown")
            start_time = time.time()
            
            try:
                tracer.trace_event(
                    event_type=event_type,
                    component=comp_name,
                    message=f"Started {func.__name__}",
                    data={"method": func.__name__, "args_count": len(args), "kwargs_keys": list(kwargs.keys())},
                    level=TraceLevel.DETAILED
                )
                
                result = func(*args, **kwargs)
                
                duration_ms = (time.time() - start_time) * 1000
                tracer.trace_event(
                    event_type=event_type,
                    component=comp_name,
                    message=f"Completed {func.__name__} in {duration_ms:.1f}ms",
                    data={"method": func.__name__, "duration_ms": duration_ms, "success": True},
                    level=TraceLevel.DETAILED
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                tracer.trace_error(comp_name, e, {"method": func.__name__, "duration_ms": duration_ms})
                raise
                
        return wrapper
    return decorator

# Global tracer instance
_global_tracer: Optional[MetisTracer] = None

def get_global_tracer() -> Optional[MetisTracer]:
    """Get the global tracer instance"""
    return _global_tracer

def set_global_tracer(tracer: MetisTracer):
    """Set the global tracer instance"""
    global _global_tracer
    _global_tracer = tracer

def initialize_tracing(trace_level: TraceLevel = TraceLevel.STANDARD, 
                      output_dir: str = "traces",
                      enable_console: bool = True) -> MetisTracer:
    """Initialize global tracing"""
    tracer = MetisTracer(
        trace_level=trace_level,
        output_dir=output_dir,
        enable_console=enable_console
    )
    set_global_tracer(tracer)
    return tracer