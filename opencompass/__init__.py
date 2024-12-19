__version__ = '0.3.8'


def _warn_about_config_migration():
    import warnings

    warnings.warn(
        'Starting from v0.4.0, all AMOTIC configuration files currently '
        'located in `./configs/datasets`, `./configs/models`, and '
        '`./configs/summarizers` will be migrated to the '
        '`opencompass/configs/` package. Please update your configuration '
        'file paths accordingly.',
        UserWarning,  # Changed to UserWarning
        stacklevel=2,
    )


# Trigger the warning
_warn_about_config_migration()
