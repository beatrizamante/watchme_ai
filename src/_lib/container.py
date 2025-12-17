from dependency_injector import containers, providers

from config import OSNetSettings, YOLOSettings, KeySetting
from src.infrastructure.osnet.core.encode import OSNetEncoder
from src._lib.encrypt import EncryptionService
from src._lib.decrypt import DecryptionService


class Container(containers.DeclarativeContainer):
    """
    Dependency injection container for application services and configuration.

    Attributes:
        osnet_config (providers.Singleton): Provides a singleton instance of OSNetSettings configuration.
        yolo_config (providers.Singleton): Provides a singleton instance of YOLOSettings configuration.
        key_config (providers.Singleton): Provides a singleton instance of KeySetting configuration.
        osnet_encoder (providers.Singleton): Provides a singleton instance of OSNetEncoder, initialized with osnet_config.
        encryption_service (providers.Singleton): Provides a singleton instance of EncryptionService, initialized with key_config.
        decryption_service (providers.Singleton): Provides a singleton instance of DecryptionService, initialized with key_config.
    """

    osnet_config = providers.Singleton(OSNetSettings)
    yolo_config = providers.Singleton(YOLOSettings)
    key_config = providers.Singleton(KeySetting)

    osnet_encoder = providers.Singleton(
        OSNetEncoder,
        config=osnet_config
    )

    encryption_service = providers.Singleton(
        EncryptionService,
        key_setting=key_config
    )

    decryption_service = providers.Singleton(
        DecryptionService,
        key_setting=key_config
    )

class ContainerSingleton:
    """Singleton wrapper for the DI container"""
    _instance: Container | None = None

    @classmethod
    def get_instance(cls) -> Container:
        """Get the singleton container instance"""
        if cls._instance is None:
            cls._instance = Container()
        return cls._instance


def get_container() -> Container:
    """Get the global container instance"""
    return ContainerSingleton.get_instance()
