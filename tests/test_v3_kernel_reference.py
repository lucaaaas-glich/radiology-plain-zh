import functools
from radplain.kernel_probe_v3 import reference_wrapper


def test_kernel_probe_reference_does_not_accidentally_compare_fla_with_itself():
    def actual_reference(a,scale=1):return a*scale
    @functools.wraps(actual_reference)
    def accelerated(*args,**kwargs):return 'accelerated'
    reference=reference_wrapper(accelerated)
    assert reference(3,scale=2,unrelated_model_kwarg=True)==6
