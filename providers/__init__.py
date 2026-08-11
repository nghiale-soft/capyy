"""Provider plugins cho AI Gateway.

Mỗi provider implement `providers.base.Provider` và đăng ký trong Registry.
"""

from .base import Provider

__all__ = ["Provider"]
