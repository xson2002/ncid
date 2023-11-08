
class NullDistributionStrategy:
    def scope(self):
        return NullContextManager()

class NullContextManager:

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Do not surpress any raised exceptions in the with scope
        return False