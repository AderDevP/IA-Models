import inspect
import re

class DummyComponent:
    def __init__(self, label=None, value=None):
        self.label = label
        self.value = value

def make_safe_init(original_init):
    def safe_init(self, *args, **kwargs):
        while True:
            try:
                return original_init(self, *args, **kwargs)
            except TypeError as e:
                msg = str(e)
                if "unexpected keyword argument" in msg:
                    m = re.search(r"unexpected keyword argument '([^']+)'", msg)
                    if m and m.group(1) in kwargs:
                        kwargs.pop(m.group(1), None)
                        continue
                raise e
    return safe_init

DummyComponent.__init__ = make_safe_init(DummyComponent.__init__)

obj = DummyComponent(label="MammoAI", height=200, show_copy_button=True, datatype=["str"])
print("SUCCESS! Label:", obj.label)
print("Instance created without error!")
