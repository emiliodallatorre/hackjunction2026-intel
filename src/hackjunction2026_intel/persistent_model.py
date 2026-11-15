"""An InferenceModel that stays loaded across ``RobotRuntime.run()`` calls.

``PolicySource.disconnect()`` closes its model at the end of every run, which would force the
policy to be reloaded and recompiled for each stage. This subclass ignores that ``close()``;
its owner calls ``release()`` once at the end of the session instead. Each run still resets the
model state, because ``PolicySource.connect()`` resets it.
"""

from physicalai.inference import InferenceModel


class PersistentInferenceModel(InferenceModel):
    def close(self) -> None:
        """Keep the model loaded; call ``release()`` to actually free it."""

    def release(self) -> None:
        super().close()
