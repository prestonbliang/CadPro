import pytest
from OCP.TopoDS import TopoDS_Shape

from cad_diff.boolean_diff import _run_boolean, boolean_cross_check


class _NullResultOperation:
    def __init__(self, left, right):
        pass

    def IsDone(self):
        return True

    def Shape(self):
        return TopoDS_Shape()


def test_incomplete_boolean_operation_raises_a_contextual_error():
    null_shape = TopoDS_Shape()

    with pytest.raises(RuntimeError, match="Boolean added-volume cut did not complete"):
        boolean_cross_check(null_shape, null_shape)


def test_completed_boolean_with_null_result_is_rejected():
    null_shape = TopoDS_Shape()

    with pytest.raises(RuntimeError, match="Boolean test operation produced a null shape"):
        _run_boolean(_NullResultOperation, null_shape, null_shape, "test operation")
