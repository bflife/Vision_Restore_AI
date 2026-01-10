from typing import Any, Dict, List, Optional, Callable
import copy
from vision_restore.core.logger import get_logger

logger = get_logger(__name__)

class HistoryManager:
    """Manages undo/redo history for application state."""
    
    def __init__(self, max_depth: int = 50):
        self.max_depth = max_depth
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []
        self._current_state: Optional[Dict[str, Any]] = None
        
    def set_initial_state(self, state: Dict[str, Any]):
        """Set the initial state (clears history)."""
        self._current_state = copy.deepcopy(state)
        self._undo_stack.clear()
        self._redo_stack.clear()
        logger.debug("History initialized")

    def push_state(self, state: Dict[str, Any]):
        """Push a new state to history.
        
        Only pushes if state is different from current.
        """
        if self._current_state is None:
            self.set_initial_state(state)
            return

        # Simple equality check (can be optimized)
        if state == self._current_state:
            return

        self._undo_stack.append(self._current_state)
        self._current_state = copy.deepcopy(state)
        self._redo_stack.clear()
        
        # Limit depth
        if len(self._undo_stack) > self.max_depth:
            self._undo_stack.pop(0)
            
        logger.debug(f"State pushed. Undo stack: {len(self._undo_stack)}")

    def undo(self) -> Optional[Dict[str, Any]]:
        """Undo last operation."""
        if not self._undo_stack:
            return None
            
        self._redo_stack.append(self._current_state)
        previous_state = self._undo_stack.pop()
        self._current_state = previous_state
        logger.debug(f"Undo performed. Stack: {len(self._undo_stack)}")
        return copy.deepcopy(previous_state)

    def redo(self) -> Optional[Dict[str, Any]]:
        """Redo last undone operation."""
        if not self._redo_stack:
            return None
            
        self._undo_stack.append(self._current_state)
        next_state = self._redo_stack.pop()
        self._current_state = next_state
        logger.debug(f"Redo performed. Stack: {len(self._undo_stack)}")
        return copy.deepcopy(next_state)

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0
