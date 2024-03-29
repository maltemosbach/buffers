from enum import IntEnum


class VerificationMode(IntEnum):
    NO_CHECKS = 0
    BASIC_CHECKS = 1
    EXTENDED_CHECKS = 2
    DEBUGGING_CHECKS = 3


def basic_check(func):
    """Decorator for basic checks that ensure the dataset functions correctly. Can be disabled for performance."""
    def wrapper(self, *args, **kwargs):
        if self.verification_mode >= VerificationMode.BASIC_CHECKS:
            return func(self, *args, **kwargs)
        else:
            return None
    return wrapper


def extended_check(func):
    """Decorator for more extensive checks."""
    def wrapper(self, *args, **kwargs):
        if self.verification_mode >= VerificationMode.EXTENDED_CHECKS:
            return func(self, *args, **kwargs)
        else:
            return None
    return wrapper


def debugging_check(func):
    """Decorator for more debugging checks to infer what is happening in a dataset."""
    def wrapper(self, *args, **kwargs):
        if self.verification_mode == VerificationMode.DEBUGGING_CHECKS:
            return func(self, *args, **kwargs)
        else:
            return None
    return wrapper
