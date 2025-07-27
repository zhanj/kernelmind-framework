import unittest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from kernelmind.core import Point, Line, _Transition


class TestPoint(unittest.TestCase):
    """Test cases for the Point class"""

    def setUp(self):
        """Set up test fixtures"""
        self.memory = {"data": "test_data", "result": None}
        self.simple_point = Point()
        self.async_point = Point(is_async=True)
        self.retry_point = Point(retries=3, wait=0.1)

    def test_point_initialization(self):
        """Test Point initialization with different parameters"""
        # Test default initialization
        point = Point()
        self.assertEqual(point.retries, 1)
        self.assertEqual(point.wait, 0)
        self.assertFalse(point.parallel)
        self.assertFalse(point.is_async)
        self.assertEqual(point.next_points, {})

        # Test custom initialization
        point = Point(retries=5, wait=2.0, parallel=True, is_async=True)
        self.assertEqual(point.retries, 5)
        self.assertEqual(point.wait, 2.0)
        self.assertTrue(point.parallel)
        self.assertTrue(point.is_async)

    def test_next_points_connection(self):
        """Test connecting points with next() method"""
        point1 = Point()
        point2 = Point()
        
        # Test default connection
        result = point1.next(point2)
        self.assertEqual(result, point2)
        self.assertEqual(point1.next_points["default"], point2)
        
        # Test custom action connection
        result = point1.next(point2, "success")
        self.assertEqual(point1.next_points["success"], point2)

    def test_next_points_warning(self):
        """Test warning when overwriting existing connection"""
        point1 = Point()
        point2 = Point()
        point3 = Point()
        
        point1.next(point2, "action")
        with self.assertWarns(UserWarning):
            point1.next(point3, "action")
        
        self.assertEqual(point1.next_points["action"], point3)

    def test_rshift_operator(self):
        """Test >> operator for connecting points"""
        point1 = Point()
        point2 = Point()
        
        result = point1 >> point2
        self.assertEqual(result, point2)
        self.assertEqual(point1.next_points["default"], point2)

    def test_sub_operator(self):
        """Test - operator for creating transitions"""
        point1 = Point()
        
        # Test valid action
        transition = point1 - "success"
        self.assertIsInstance(transition, _Transition)
        self.assertEqual(transition.src, point1)
        self.assertEqual(transition.action, "success")
        
        # Test invalid action type
        with self.assertRaises(TypeError):
            point1 - 123

    def test_basic_point_operations(self):
        """Test basic load, process, save operations"""
        class TestPoint(Point):
            def load(self, memory):
                return memory.get("data")
            
            def process(self, item):
                return f"processed_{item}"
            
            def save(self, memory, prep_result, exec_result):
                memory["result"] = exec_result
                return "default"
        
        test_point = TestPoint()
        result = test_point.run(self.memory)
        
        self.assertEqual(result, "default")
        self.assertEqual(self.memory["result"], "processed_test_data")

    def test_point_with_none_load_result(self):
        """Test point behavior when load returns None"""
        class NoneLoadPoint(Point):
            def load(self, memory):
                return None
            
            def save(self, memory, prep_result, exec_result):
                memory["result"] = "none_result"
                return "default"
        
        test_point = NoneLoadPoint()
        result = test_point.run(self.memory)
        
        self.assertEqual(result, "default")
        self.assertEqual(self.memory["result"], "none_result")

    def test_sync_batch_processing(self):
        """Test batch processing with sync operations"""
        class BatchPoint(Point):
            def load(self, memory):
                return ["item1", "item2", "item3"]
            
            def process(self, item):
                return f"processed_{item}"
            
            def save(self, memory, prep_result, exec_result):
                memory["results"] = exec_result
                return "default"
        
        test_point = BatchPoint()
        result = test_point.run(self.memory)
        
        self.assertEqual(result, "default")
        expected_results = ["processed_item1", "processed_item2", "processed_item3"]
        self.assertEqual(self.memory["results"], expected_results)

    def test_retry_mechanism(self):
        """Test retry mechanism with failures"""
        class FailingPoint(Point):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.call_count = 0
            
            def load(self, memory):
                return "test_item"
            
            def process(self, item):
                self.call_count += 1
                if self.call_count < 3:  # Fail first 2 times
                    raise ValueError("Simulated failure")
                return "success"
            
            def save(self, memory, prep_result, exec_result):
                memory["result"] = exec_result
                return "default"
        
        # Test with retries
        test_point = FailingPoint(retries=3, wait=0.01)
        result = test_point.run(self.memory)
        
        self.assertEqual(result, "default")
        self.assertEqual(self.memory["result"], "success")
        self.assertEqual(test_point.call_count, 3)

    def test_fallback_mechanism(self):
        """Test fallback mechanism when all retries fail"""
        class AlwaysFailingPoint(Point):
            def load(self, memory):
                return "test_item"
            
            def process(self, item):
                raise ValueError("Always fails")
            
            def on_fail(self, item, exc):
                return "fallback_result"
            
            def save(self, memory, prep_result, exec_result):
                memory["result"] = exec_result
                return "default"
        
        test_point = AlwaysFailingPoint(retries=2, wait=0.01)
        result = test_point.run(self.memory)
        
        self.assertEqual(result, "default")
        self.assertEqual(self.memory["result"], "fallback_result")

    def test_async_point_operations(self):
        """Test async point operations"""
        class AsyncTestPoint(Point):
            def load(self, memory):
                return "async_item"
            
            async def process(self, item):
                await asyncio.sleep(0.01)  # Simulate async work
                return f"async_processed_{item}"
            
            def save(self, memory, prep_result, exec_result):
                memory["result"] = exec_result
                return "default"
        
        test_point = AsyncTestPoint(is_async=True)
        result = test_point.run(self.memory)
        
        self.assertEqual(result, "default")
        self.assertEqual(self.memory["result"], "async_processed_async_item")

    def test_async_batch_processing(self):
        """Test async batch processing"""
        class AsyncBatchPoint(Point):
            def load(self, memory):
                return ["item1", "item2", "item3"]
            
            async def process(self, item):
                await asyncio.sleep(0.01)  # Simulate async work
                return f"async_processed_{item}"
            
            def save(self, memory, prep_result, exec_result):
                memory["results"] = exec_result
                return "default"
        
        test_point = AsyncBatchPoint(is_async=True)
        result = test_point.run(self.memory)
        
        self.assertEqual(result, "default")
        expected_results = ["async_processed_item1", "async_processed_item2", "async_processed_item3"]
        self.assertEqual(self.memory["results"], expected_results)

    def test_parallel_async_processing(self):
        """Test parallel async processing"""
        class ParallelAsyncPoint(Point):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.start_times = []
            
            def load(self, memory):
                return ["item1", "item2", "item3"]
            
            async def process(self, item):
                self.start_times.append(time.time())
                await asyncio.sleep(0.1)  # Simulate longer async work
                return f"parallel_processed_{item}"
            
            def save(self, memory, prep_result, exec_result):
                memory["results"] = exec_result
                return "default"
        
        test_point = ParallelAsyncPoint(is_async=True, parallel=True)
        start_time = time.time()
        result = test_point.run(self.memory)
        end_time = time.time()
        
        self.assertEqual(result, "default")
        expected_results = ["parallel_processed_item1", "parallel_processed_item2", "parallel_processed_item3"]
        self.assertEqual(self.memory["results"], expected_results)
        
        # Check that processing was actually parallel (should take less than 0.3 seconds for 3 items)
        self.assertLess(end_time - start_time, 0.3)

    def test_async_warning(self):
        """Test warning when is_async=True but process is not async"""
        class SyncPointWithAsyncFlag(Point):
            def load(self, memory):
                return "test"
            
            def process(self, item):  # Not async
                return "result"
            
            def save(self, memory, prep_result, exec_result):
                return "default"
        
        with self.assertWarns(UserWarning):
            test_point = SyncPointWithAsyncFlag(is_async=True)
        
        # Should still work despite the warning
        result = test_point.run(self.memory)
        self.assertEqual(result, "default")

    def test_async_in_sync_context_error(self):
        """Test error when calling async process in sync context"""
        class AsyncPointInSyncContext(Point):
            def load(self, memory):
                return "test"
            
            async def process(self, item):
                return "result"
            
            def save(self, memory, prep_result, exec_result):
                return "default"
        
        test_point = AsyncPointInSyncContext()  # is_async=False by default
        with self.assertRaises(RuntimeError):
            test_point.run(self.memory)

    def test_point_with_successors_warning(self):
        """Test warning when running point with successors independently"""
        point1 = Point()
        point2 = Point()
        point1.next(point2)
        
        with self.assertWarns(UserWarning):
            point1.run(self.memory)


class TestTransition(unittest.TestCase):
    """Test cases for the _Transition class"""

    def test_transition_creation(self):
        """Test _Transition object creation"""
        point = Point()
        transition = _Transition(point, "success")
        
        self.assertEqual(transition.src, point)
        self.assertEqual(transition.action, "success")

    def test_transition_rshift(self):
        """Test >> operator for transitions"""
        point1 = Point()
        point2 = Point()
        transition = _Transition(point1, "success")
        
        result = transition >> point2
        self.assertEqual(result, point2)
        self.assertEqual(point1.next_points["success"], point2)


class TestLine(unittest.TestCase):
    """Test cases for the Line class"""

    def setUp(self):
        """Set up test fixtures"""
        self.memory = {"data": "test_data", "result": None}

    def test_line_initialization(self):
        """Test Line initialization"""
        point = Point()
        line = Line(entry=point)
        
        self.assertEqual(line.entry, point)
        self.assertEqual(line.retries, 1)  # Inherited from Point
        self.assertEqual(line.wait, 0)

    def test_simple_line_execution(self):
        """Test simple line with two points"""
        class FirstPoint(Point):
            def load(self, memory):
                return memory.get("data")
            
            def process(self, item):
                return f"first_{item}"
            
            def save(self, memory, prep_result, exec_result):
                memory["first_result"] = exec_result
                return "default"
        
        class SecondPoint(Point):
            def load(self, memory):
                return memory.get("first_result")
            
            def process(self, item):
                return f"second_{item}"
            
            def save(self, memory, prep_result, exec_result):
                memory["final_result"] = exec_result
                return "default"
        
        first = FirstPoint()
        second = SecondPoint()
        first >> second
        
        line = Line(entry=first)
        result = line.run(self.memory)
        
        self.assertEqual(result, "default")
        self.assertEqual(self.memory["first_result"], "first_test_data")
        self.assertEqual(self.memory["final_result"], "second_first_test_data")

    def test_line_with_custom_actions(self):
        """Test line with custom actions"""
        class DecisionPoint(Point):
            def load(self, memory):
                return memory.get("decision")
            
            def process(self, item):
                return "success" if item == "yes" else "failure"
            
            def save(self, memory, prep_result, exec_result):
                memory["decision_result"] = exec_result
                return exec_result
        
        class SuccessPoint(Point):
            def load(self, memory):
                return "success_data"
            
            def process(self, item):
                return "success_processed"
            
            def save(self, memory, prep_result, exec_result):
                memory["success_result"] = exec_result
                return "default"
        
        class FailurePoint(Point):
            def load(self, memory):
                return "failure_data"
            
            def process(self, item):
                return "failure_processed"
            
            def save(self, memory, prep_result, exec_result):
                memory["failure_result"] = exec_result
                return "default"
        
        decision = DecisionPoint()
        success = SuccessPoint()
        failure = FailurePoint()
        
        decision - "success" >> success
        decision - "failure" >> failure
        
        line = Line(entry=decision)
        
        # Test success path
        self.memory["decision"] = "yes"
        result = line.run(self.memory)
        self.assertEqual(result, "default")
        self.assertEqual(self.memory["decision_result"], "success")
        self.assertEqual(self.memory["success_result"], "success_processed")
        self.assertNotIn("failure_result", self.memory)
        
        # Test failure path
        self.memory.clear()
        self.memory["decision"] = "no"
        result = line.run(self.memory)
        self.assertEqual(result, "default")
        self.assertEqual(self.memory["decision_result"], "failure")
        self.assertEqual(self.memory["failure_result"], "failure_processed")
        self.assertNotIn("success_result", self.memory)

    def test_line_termination(self):
        """Test line termination when no successor is found"""
        class TerminalPoint(Point):
            def load(self, memory):
                return "terminal_data"
            
            def process(self, item):
                return "terminal_result"
            
            def save(self, memory, prep_result, exec_result):
                memory["terminal"] = exec_result
                return "unknown_action"  # No successor for this action
        
        terminal = TerminalPoint()
        line = Line(entry=terminal)
        
        # Should print termination message and return the action
        with patch('builtins.print') as mock_print:
            result = line.run(self.memory)
        
        self.assertEqual(result, "unknown_action")
        self.assertEqual(self.memory["terminal"], "terminal_result")
        mock_print.assert_called_once()

    def test_line_with_no_entry(self):
        """Test line with no entry point"""
        line = Line()
        result = line.run(self.memory)
        self.assertIsNone(result)

    def test_line_with_looping(self):
        """Test line with looping behavior"""
        class CounterPoint(Point):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.count = 0
            
            def load(self, memory):
                return memory.get("count", 0)
            
            def process(self, item):
                return item + 1
            
            def save(self, memory, prep_result, exec_result):
                memory["count"] = exec_result
                self.count = exec_result
                return "continue" if exec_result < 3 else "done"
        
        class FinalPoint(Point):
            def load(self, memory):
                return memory.get("count")
            
            def process(self, item):
                return f"final_{item}"
            
            def save(self, memory, prep_result, exec_result):
                memory["final"] = exec_result
                return "default"
        
        counter = CounterPoint()
        final = FinalPoint()
        
        counter - "continue" >> counter
        counter - "done" >> final
        
        line = Line(entry=counter)
        result = line.run(self.memory)
        
        self.assertEqual(result, "default")
        self.assertEqual(self.memory["count"], 3)
        self.assertEqual(self.memory["final"], "final_3")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def setUp(self):
        """Set up test fixtures"""
        self.memory = {"data": "test_data", "result": None}

    def test_complex_workflow(self):
        """Test a complex workflow with multiple points and actions"""
        class DataLoader(Point):
            def load(self, memory):
                return memory.get("input_data", "default_input")
            
            def process(self, item):
                return f"loaded_{item}"
            
            def save(self, memory, prep_result, exec_result):
                memory["loaded_data"] = exec_result
                return "default"
        
        class DataProcessor(Point):
            def load(self, memory):
                return memory.get("loaded_data")
            
            def process(self, item):
                if "error" in item:
                    raise ValueError("Processing error")
                return f"processed_{item}"
            
            def on_fail(self, item, exc):
                # Return a fallback result instead of raising the exception
                return f"error_processed_{item}"
            
            def save(self, memory, prep_result, exec_result):
                memory["processed_data"] = exec_result
                return "success"
        
        class ErrorHandler(Point):
            def load(self, memory):
                return memory.get("loaded_data")
            
            def process(self, item):
                return f"error_handled_{item}"
            
            def save(self, memory, prep_result, exec_result):
                memory["error_result"] = exec_result
                return "default"
        
        class Finalizer(Point):
            def load(self, memory):
                return memory.get("processed_data") or memory.get("error_result")
            
            def process(self, item):
                return f"finalized_{item}"
            
            def save(self, memory, prep_result, exec_result):
                memory["final_result"] = exec_result
                return "default"
        
        # Create points
        loader = DataLoader()
        processor = DataProcessor()
        error_handler = ErrorHandler()
        finalizer = Finalizer()
        
        # Connect points
        loader >> processor
        processor - "success" >> finalizer
        processor - "default" >> error_handler
        error_handler >> finalizer
        
        # Create line
        line = Line(entry=loader)
        
        # Test normal flow
        memory = {"input_data": "test_data"}
        result = line.run(memory)
        
        self.assertEqual(result, "default")
        self.assertEqual(memory["final_result"], "finalized_processed_loaded_test_data")
        
        # Test error flow
        memory = {"input_data": "error_data"}
        result = line.run(memory)
        
        self.assertEqual(result, "default")
        self.assertEqual(memory["final_result"], "finalized_error_processed_loaded_error_data")

    def test_async_integration(self):
        """Test async integration with multiple points"""
        class AsyncLoader(Point):
            def load(self, memory):
                return ["item1", "item2"]
            
            async def process(self, item):
                await asyncio.sleep(0.01)
                return f"async_loaded_{item}"
            
            def save(self, memory, prep_result, exec_result):
                memory["loaded_items"] = exec_result
                return "default"
        
        class AsyncProcessor(Point):
            def load(self, memory):
                return memory.get("loaded_items")
            
            async def process(self, item):
                await asyncio.sleep(0.01)
                return f"async_processed_{item}"
            
            def save(self, memory, prep_result, exec_result):
                memory["processed_items"] = exec_result
                return "default"
        
        loader = AsyncLoader(is_async=True)
        processor = AsyncProcessor(is_async=True)
        loader >> processor
        
        line = Line(entry=loader)
        result = line.run(self.memory)
        
        self.assertEqual(result, "default")
        expected_loaded = ["async_loaded_item1", "async_loaded_item2"]
        expected_processed = ["async_processed_async_loaded_item1", "async_processed_async_loaded_item2"]
        self.assertEqual(self.memory["loaded_items"], expected_loaded)
        self.assertEqual(self.memory["processed_items"], expected_processed)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2) 